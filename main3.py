# action_recommender_agent.py

import asyncio
import logging
from app.agents.action_recommender import ActionRecommenderAgent
from app.models.schemas import ModerationAction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestScript")

async def test_action_recommendation(agent, text: str, label: str, severity: str):
    """Test action recommendation for a given text and expected classification"""
    print(f"\n🔍 Testing: {text}")
    print(f"Expected: {label} (Severity: {severity})")
    
    # Create test classification result
    classification = {
        "label": label,
        "severity": severity,
        "confidence": 0.9
    }
    
    # Create test reasoning result
    reasoning = {
        "policy_violations": [f"{label} content detected"],
        "severity_assessment": f"{severity} severity content"
    }
    
    try:
        result = await agent.recommend_action(classification, reasoning)
        print("\n✅ Result:")
        print(f"Action: {result['primary_action']}")
        print(f"Justification: {result['justification']}")
        print(f"Escalation Needed: {result['escalation_needed']}")
        return result
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

async def main():
    print("🚀 Starting Action Recommender Tests")
    print("="*50)
    
    # Initialize agent
    agent = ActionRecommenderAgent()
    
    # Test cases
    test_cases = [
        {
            "text": "I hate everyone from that country!",
            "label": "hate",
            "severity": "high"
        },
        {
            "text": "You're so stupid",
            "label": "toxic",
            "severity": "medium"
        },
        {
            "text": "This movie sucks",
            "label": "offensive",
            "severity": "low"
        },
        {
            "text": "Have a nice day!",
            "label": "neutral",
            "severity": "low"
        },
        {
            "text": "🤔 unclear message ...",
            "label": "ambiguous",
            "severity": "medium"
        }
    ]
    
    # Run tests
    results = []
    for case in test_cases:
        result = await test_action_recommendation(
            agent,
            case["text"],
            case["label"],
            case["severity"]
        )
        results.append(result)
    
    # Summary
    print("\n📊 Test Summary")
    print("="*50)
    print(f"Total tests: {len(test_cases)}")
    print(f"Successful: {len([r for r in results if r is not None])}")
    print(f"Failed: {len([r for r in results if r is None])}")

if __name__ == "__main__":
    asyncio.run(main())
