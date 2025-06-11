# policy_reasoning.py

import asyncio
import logging
from app.agents.policy_reasoning import PolicyReasoningAgent
from app.models.schemas import ClassificationResult, PolicyDocument

# Set up standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PolicyReasoningTest")


async def main():
    agent = PolicyReasoningAgent()

    # Example input data
    test_text = "This group should be banned from our country."

    # Fixed ClassificationResult with all required fields
    classification_result = ClassificationResult(
        label="hate",  # lowercase as required by enum
        confidence=0.95,
        explanation="The text expresses hate towards a group.",
        severity_score=9,
        severity="critical",  # Required field
        reasoning="Explicit hate speech targeting a group.",  # Required field
        metadata={},
    )

    # Update PolicyDocument with more meaningful content
    policy_documents = [
        PolicyDocument(
            content="Hate speech that targets individuals or groups based on protected characteristics such as nationality, ethnicity, or origin is strictly prohibited. This includes calls for exclusion, deportation, or banning of groups from the country.",
            source="Community Guidelines",
            relevance_score=0.92,
            doc_id="policy-001",
            title="Hate Speech Policy",
            policy_type="community_guidelines",
            metadata={},
        )
    ]

    logger.info("Testing PolicyReasoningAgent...")

    # Note: The method name in your policy_reasoning.py is generate_policy_reasoning, not analyze_content_against_policies
    # Converting the data structures to match the expected format
    classification_dict = {
        "classification": classification_result.label,
        "confidence": str(classification_result.confidence),
        "reason": classification_result.reasoning,
        "severity": classification_result.severity
    }

    policy_dicts = [
        {
            "content": doc.content,
            "source": doc.source,
            "similarity_score": doc.relevance_score,
        }
        for doc in policy_documents
    ]

    result = await agent.generate_policy_reasoning(
        original_text=test_text,
        classification_result=classification_dict,
        relevant_policies=policy_dicts,
    )

    logger.info("Policy Reasoning Result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
