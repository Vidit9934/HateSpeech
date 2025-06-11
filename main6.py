# main6_updated.py - Updated Test Suite for HybridRetrieverAgent
#!/usr/bin/env python3
"""
Updated test suite that works with the fixed HybridRetrieverAgent implementation
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
import json

# Import your fixed HybridRetrieverAgent
from app.agents.hybrid_retriever import HybridRetrieverAgent

# Set logging level to reduce noise
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("HybridRetrieverAgent").setLevel(logging.INFO)

def print_header(title):
    """Print a clean header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_policy_summary(policies: List[Dict[str, Any]], max_content_length: int = 100):
    """Print a clean summary of retrieved policies"""
    if not policies:
        print("   ❌ No policies retrieved")
        return
    
    for i, policy in enumerate(policies, 1):
        content = policy.get('content', '')[:max_content_length]
        if len(policy.get('content', '')) > max_content_length:
            content += "..."
        
        score = policy.get('similarity_score', 0)
        final_score = policy.get('final_score', score)
        source = policy.get('source', 'unknown')
        policy_type = policy.get('policy_type', 'general')
        
        print(f"   {i}. 📄 Policy (Score: {final_score:.3f})")
        print(f"      Source: {source} | Type: {policy_type}")
        print(f"      Content: {content}")
        print()

async def test_initialization():
    """Test HybridRetrieverAgent initialization"""
    print_header("INITIALIZATION TEST")
    
    try:
        print("1. Testing HybridRetrieverAgent initialization...")
        retriever = HybridRetrieverAgent()
        
        print("   ✅ Agent initialized successfully")
        print(f"   ✅ Vector store connected: {retriever.vector_store is not None}")
        print(f"   ✅ Embedding service ready: {retriever.embedding_service is not None}")
        
        # Test collection stats
        stats = retriever.get_collection_stats()
        print(f"   ✅ Collection stats: {stats}")
        
        return retriever
        
    except Exception as e:
        print(f"   ❌ Initialization failed: {str(e)}")
        print("   💡 Check your Qdrant connection and embedding service")
        import traceback
        traceback.print_exc()
        return None

async def ensure_test_data(retriever):
    """Ensure we have test data in the collection"""
    print_header("DATA PREPARATION")
    
    if not retriever:
        print("   ⚠️  Skipping - retriever not initialized")
        return False
    
    try:
        stats = retriever.get_collection_stats()
        vectors_count = stats.get('vectors_count', 0)
        
        print(f"Current collection size: {vectors_count} documents")
        
        if vectors_count == 0:
            print("   📝 No data found. Adding sample test documents...")
            
            # Add minimal test documents
            test_docs = [
                {
                    "id": "test_hate_policy",
                    "content": "Hate speech and discriminatory language targeting specific groups is prohibited...",
                    "title": "Hate Speech Policy",
                    "policy_type": "hate_speech",  # Be specific about policy type
                    "source": "community_guidelines",
                    "metadata": {
                        "version": "1.0",
                        "last_updated": "2025-06-12",
                        "severity": "high"
                    }
                },
                {
                    "id": "test_toxic_policy", 
                    "content": "Toxic behavior including threats, harassment and abusive language creates a hostile environment and violates our standards.",
                    "title": "Toxic Behavior Policy",
                    "policy_type": "toxic",
                    "source": "test_guidelines"
                },
                {
                    "id": "test_general_policy",
                    "content": "General community standards require respectful interaction and constructive dialogue between all members.",
                    "title": "General Community Standards",
                    "policy_type": "general", 
                    "source": "test_guidelines"
                }
            ]
            
            result = await retriever.index_policy_documents(test_docs)
            
            if result.get("success", True):
                print(f"   ✅ Added {len(test_docs)} test documents")
                return True
            else:
                print(f"   ❌ Failed to add test documents: {result.get('error')}")
                return False
        else:
            print(f"   ✅ Found {vectors_count} existing documents")
            return True
            
    except Exception as e:
        print(f"   ❌ Data preparation failed: {str(e)}")
        return False

async def test_basic_retrieval(retriever):
    """Test basic policy retrieval functionality"""
    print_header("BASIC RETRIEVAL TEST")
    
    if not retriever:
        print("   ⚠️  Skipping - retriever not initialized")
        return
    
    test_queries = [
        "hate speech policy",
        "harassment guidelines", 
        "toxic content moderation",
        "community standards",
        "discriminatory language"
    ]
    
    print("Testing basic policy retrieval with various queries:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        try:
            start_time = time.time()
            policies = await retriever.retrieve_relevant_policies(query, top_k=3)
            duration = time.time() - start_time
            
            print(f"   ✅ Retrieved {len(policies)} policies in {duration:.3f}s")
            print_policy_summary(policies, max_content_length=80)
            
        except Exception as e:
            print(f"   ❌ Query failed: {str(e)}")

async def test_hybrid_search(retriever):
    """Test hybrid search functionality"""
    print_header("HYBRID SEARCH TEST")
    
    if not retriever:
        print("   ⚠️  Skipping - retriever not initialized")
        return
    
    test_cases = [
        {
            "query": "user posted discriminatory content",
            "classification": "Hate",
            "description": "Hate speech classification"
        },
        {
            "query": "aggressive threatening message",
            "classification": "Toxic", 
            "description": "Toxic content classification"
        },
        {
            "query": "regular community discussion",
            "classification": "Neutral",
            "description": "Neutral content classification"
        }
    ]
    
    print("Testing hybrid search with different classifications:")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print(f"   Query: '{case['query']}'")
        print(f"   Classification: {case['classification']}")
        
        try:
            start_time = time.time()
            policies = await retriever.hybrid_search(
                query=case['query'],
                classification=case['classification'],
                top_k=3
            )
            duration = time.time() - start_time
            
            print(f"   ✅ Found {len(policies)} policies in {duration:.3f}s")
            
            # Show relevance boosting
            for policy in policies:
                boost = policy.get('relevance_boost', 0)
                if boost > 0:
                    print(f"   📈 Relevance boost applied: +{boost:.2f}")
                    break
            
            print_policy_summary(policies, max_content_length=60)
            
        except Exception as e:
            print(f"   ❌ Hybrid search failed: {str(e)}")

async def test_policy_context(retriever):
    """Test policy context generation"""
    print_header("POLICY CONTEXT TEST")
    
    if not retriever:
        print("   ⚠️  Skipping - retriever not initialized")
        return
    
    test_classifications = [
        {
            "classification": "Hate",
            "reason": "Contains discriminatory language targeting specific group",
            "description": "Hate speech case"
        },
        {
            "classification": "Toxic", 
            "reason": "Threatening and abusive language used",
            "description": "Toxic content case"
        },
        {
            "classification": "Neutral",
            "reason": "Regular community discussion",
            "description": "Neutral content case"
        }
    ]
    
    print("Testing policy context generation:")
    
    for i, case in enumerate(test_classifications, 1):
        print(f"\n{i}. {case['description']}")
        print(f"   Classification: {case['classification']}")
        print(f"   Reason: {case['reason']}")
        
        try:
            start_time = time.time()
            context = await retriever.get_policy_context(case)
            duration = time.time() - start_time
            
            print(f"   ✅ Context generated in {duration:.3f}s")
            print(f"   ✅ Total policies found: {context.get('total_policies_found', 0)}")
            print(f"   ✅ Search query used: '{context.get('search_query', '')}'")
            print(f"   ✅ Classification: {context.get('classification', 'N/A')}")
            
            policies = context.get('relevant_policies', [])
            if policies:
                print(f"   📋 Top policy preview:")
                top_policy = policies[0]
                content_preview = top_policy.get('content', '')[:100] + "..."
                print(f"      {content_preview}")
            
        except Exception as e:
            print(f"   ❌ Context generation failed: {str(e)}")

async def test_edge_cases(retriever):
    """Test edge cases and error handling"""
    print_header("EDGE CASES & ERROR HANDLING TEST")
    
    if not retriever:
        print("   ⚠️  Skipping - retriever not initialized")
        return
    
    edge_cases = [
        {
            "name": "Empty query",
            "query": "",
            "classification": "Neutral"
        },
        {
            "name": "Very short query", 
            "query": "a",
            "classification": "Hate"
        },
        {
            "name": "Special characters",
            "query": "!@#$%^&*()",
            "classification": "Toxic"
        },
        {
            "name": "Unknown classification",
            "query": "test query",
            "classification": "UnknownType"
        }
    ]
    
    print("Testing edge cases:")
    
    for i, case in enumerate(edge_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Query: '{case['query']}'")
        
        try:
            policies = await retriever.hybrid_search(
                query=case['query'],
                classification=case['classification'],
                top_k=2
            )
            
            print(f"   ✅ Handled gracefully - {len(policies)} policies returned")
            
        except Exception as e:
            print(f"   ⚠️  Exception raised: {str(e)}")

async def test_filtering_logic(retriever):
    """Test classification-based filtering logic"""
    print_header("FILTERING LOGIC TEST")
    
    if not retriever:
        print("   ⚠️  Skipping - retriever not initialized")
        return
    
    print("Testing classification keyword filtering:")
    
    # Create mock policies to test filtering
    mock_policies = [
        {
            "content": "This policy covers hate speech and discrimination in our community",
            "similarity_score": 0.8,
            "metadata": {"source": "community_guidelines", "policy_type": "hate"}
        },
        {
            "content": "Toxic behavior including threats and harassment is not allowed",
            "similarity_score": 0.7,
            "metadata": {"source": "behavior_policy", "policy_type": "toxic"}
        },
        {
            "content": "General community standards and guidelines for all users",
            "similarity_score": 0.6,
            "metadata": {"source": "general_rules", "policy_type": "general"}
        }
    ]
    
    classifications = ["Hate", "Toxic", "Neutral"]
    
    for classification in classifications:
        print(f"\n📊 Testing filtering for: {classification}")
        
        filtered = retriever._filter_by_classification(mock_policies.copy(), classification)
        
        print(f"   ✅ Policies processed: {len(filtered)}")
        
        for i, policy in enumerate(filtered, 1):
            boost = policy.get('relevance_boost', 0)
            final_score = policy.get('final_score', policy.get('similarity_score', 0))
            
            print(f"   {i}. Score: {final_score:.3f} (boost: +{boost:.2f})")
            content_preview = policy['content'][:60] + "..."
            print(f"      {content_preview}")

async def main():
    """Run all tests"""
    print("🔍 HybridRetrieverAgent Test Suite (Updated)")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # Initialize the retriever
        retriever = await test_initialization()
        
        if retriever:
            # Ensure we have test data
            data_ready = await ensure_test_data(retriever)
            
            if data_ready:
                # Run all tests
                await test_basic_retrieval(retriever)
                await test_hybrid_search(retriever)
                await test_policy_context(retriever)
                await test_edge_cases(retriever)
                await test_filtering_logic(retriever)
                
                print_header("ALL TESTS COMPLETED! ✅")
            else:
                print_header("TESTS INCOMPLETE - DATA PREPARATION FAILED ❌")
        else:
            print_header("TESTS INCOMPLETE - INITIALIZATION FAILED ❌")
            print("💡 Please check:")
            print("   - Qdrant server is running")
            print("   - Config settings are correct")
            print("   - Embedding service is available")
        
        print(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print_header("TEST SUITE FAILED ❌")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())