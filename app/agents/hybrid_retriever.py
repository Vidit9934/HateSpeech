# hybrid_retriever.py - Fixed version
#!/usr/bin/env python3
"""
Fixed HybridRetrieverAgent that properly integrates with your existing EmbeddingService
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from app.services.vector_service import VectorStoreService

class HybridRetrieverAgent:
    def __init__(self, 
                 collection_name: str = "embeddings",
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize HybridRetrieverAgent with VectorStoreService
        """
        self.logger = logging.getLogger("HybridRetrieverAgent")
        
        try:
            # Initialize vector store service (similar to main1.py)
            self.vector_store = VectorStoreService(
                collection_name="embeddings",
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                model_name=model_name
            )
            
            # Access the embedding service from vector store
            self.embedding_service = self.vector_store.embedding_service
            
            self.logger.info("HybridRetrieverAgent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HybridRetrieverAgent: {str(e)}")
            raise

    async def retrieve_relevant_policies(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant policies using semantic search
        """
        try:
            # Add deduplication by content
            seen_contents = set()
            policies = []
            
            search_result = self.vector_store.search(query, top_k=top_k * 2)  # Get more results to account for duplicates
            
            for result in search_result.get("results", []):
                content = result.get("content", "")
                
                # Skip if we've seen this content before
                if content in seen_contents:
                    continue
                
                seen_contents.add(content)
                
                policy = {
                    "id": result.get("id"),
                    "content": content,
                    "similarity_score": result.get("score", 0.0),
                    "final_score": result.get("score", 0.0),
                    "source": result.get("metadata", {}).get("source", "test_guidelines"),
                    "policy_type": result.get("metadata", {}).get("policy_type", "policy"),
                    "metadata": result.get("metadata", {})
                }
                policies.append(policy)
                
                if len(policies) >= top_k:
                    break
            
            self.logger.info(f"Retrieved {len(policies)} unique policies for query: '{query}'")
            return policies
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve policies: {str(e)}")
            return []

    async def hybrid_search(self, 
                          query: str, 
                          classification: str, 
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with classification-based filtering and boosting
        """
        try:
            # First get basic semantic search results
            policies = await self.retrieve_relevant_policies(query, top_k * 2)  # Get more to filter
            
            # Apply classification-based filtering and boosting
            filtered_policies = self._filter_by_classification(policies, classification)
            
            # Return top_k results
            return filtered_policies[:top_k]
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {str(e)}")
            return []

    def _filter_by_classification(self, policies: List[Dict[str, Any]], classification: str) -> List[Dict[str, Any]]:
        """
        Filter and boost policies based on classification
        """
        # Classification keyword mappings
        classification_keywords = {
            "Hate": ["hate", "discrimination", "racist", "sexist", "homophobic", "bias"],
            "Toxic": ["toxic", "threat", "harassment", "abuse", "violent", "aggressive"],
            "Offensive": ["offensive", "vulgar", "inappropriate", "profanity", "crude"],
            "Neutral": ["community", "standard", "guideline", "general", "rule"],
            "Ambiguous": ["unclear", "context", "moderate", "review", "complex"]
        }
        
        keywords = classification_keywords.get(classification, [])
        
        for policy in policies:
            content_lower = policy.get("content", "").lower()
            
            # Calculate relevance boost based on keyword matches
            boost = 0.0
            for keyword in keywords:
                if keyword in content_lower:
                    boost += 0.1  # Add 0.1 for each keyword match
            
            # Apply boost to final score
            original_score = policy.get("similarity_score", 0.0)
            policy["relevance_boost"] = boost
            policy["final_score"] = min(1.0, original_score + boost)  # Cap at 1.0
        
        # Sort by final score
        policies.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        return policies

    async def get_policy_context(self, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive policy context for a classification result
        """
        try:
            classification = classification_result.get("classification", "Neutral")
            reason = classification_result.get("reason", "")
            
            # Create search query from classification and reason
            search_query = f"{classification.lower()} {reason}".strip()
            
            # Perform hybrid search
            relevant_policies = await self.hybrid_search(
                query=search_query,
                classification=classification,
                top_k=5
            )
            
            # Build context
            context = {
                "classification": classification,
                "reason": reason,
                "search_query": search_query,
                "total_policies_found": len(relevant_policies),
                "relevant_policies": relevant_policies,
                "timestamp": "2024-01-01T00:00:00Z"  # You can use actual timestamp
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get policy context: {str(e)}")
            return {
                "classification": classification_result.get("classification", "Unknown"),
                "reason": classification_result.get("reason", ""),
                "search_query": "",
                "total_policies_found": 0,
                "relevant_policies": [],
                "error": str(e)
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics
        """
        try:
            return self.vector_store.get_collection_stats()
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}

    async def index_policy_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index policy documents into the vector store
        """
        try:
            # Format documents for indexing
            formatted_docs = []
            for i, doc in enumerate(documents):
                formatted_doc = {
                    "id": doc.get("id", f"policy_{i}"),
                    "content": doc.get("content", ""),
                    "title": doc.get("title", f"Policy {i}"),
                    "document_type": doc.get("policy_type", "policy"),
                    "source": doc.get("source", "unknown"),
                    "metadata": doc.get("metadata", {})
                }
                formatted_docs.append(formatted_doc)
            
            # Index using vector store
            result = self.vector_store.index_documents(formatted_docs)
            
            self.logger.info(f"Indexed {len(formatted_docs)} policy documents")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to index documents: {str(e)}")
            return {"success": False, "error": str(e)}
    # Add this method to your HybridRetrieverAgent class

    async def retrieve_policies(self, text: str, classification_result) -> List[Dict[str, Any]]:
        """
        Retrieve relevant policies based on text content and classification
        """
        try:
            # Use the vector store to search for relevant policies
            search_results = self.vector_store.search(
                query=text,
                top_k=5,
                score_threshold=0.3,
                include_content=True
            )
            
            # Format results for policy reasoning
            policies = []
            for result in search_results.get('results', []):
                policy = {
                    'source': result.get('metadata', {}).get('source', 'Unknown'),
                    'content': result.get('content', ''),
                    'similarity_score': result.get('score', 0.0),
                    'metadata': result.get('metadata', {})
                }
                policies.append(policy)
            
            return policies
            
        except Exception as e:
            self.logger.error(f"Error retrieving policies: {str(e)}")
            return []  # Return empty list on error