import pytest
import pytest_asyncio
from app.agents.policy_reasoning import PolicyReasoningAgent
from app.models.moderation_schema import ClassificationResult
from app.models.policy_schema import PolicyDocument
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPolicyReasoning")

@pytest_asyncio.fixture(scope="function")
async def policy_agent():
    """Fixture to create policy reasoning agent"""
    agent = PolicyReasoningAgent()
    return agent

@pytest_asyncio.fixture
def test_classification_result():
    """Fixture for classification result data"""
    return {
        "classification": "hate",
        "confidence": "0.95",
        "reason": "The text expresses hate towards a group.",
        "severity": "critical"
    }

@pytest_asyncio.fixture
def test_policy_documents():
    """Fixture for policy documents"""
    return [
        {
            "content": "Hate speech that targets individuals or groups based on protected characteristics is prohibited.",
            "source": "Community Guidelines",
            "similarity_score": 0.92
        }
    ]

@pytest.mark.asyncio
async def test_agent_initialization(policy_agent):
    """Test if agent initializes properly"""
    assert policy_agent is not None
    assert policy_agent.model is not None
    logger.info("Agent initialized successfully")

@pytest.mark.asyncio
async def test_policy_reasoning_generation(policy_agent, test_classification_result, test_policy_documents):
    """Test policy reasoning generation for hate speech content"""
    test_text = "This group should be banned from our country."
    
    result = await policy_agent.generate_policy_reasoning(
        original_text=test_text,
        classification_result=test_classification_result,
        relevant_policies=test_policy_documents
    )
    
    assert result is not None
    assert "policy_violations" in result
    assert "detailed_reasoning" in result
    assert "supporting_evidence" in result
    assert "severity_assessment" in result
    assert "additional_context" in result
    assert result["policies_analyzed"] == len(test_policy_documents)
    
    logger.info("Policy reasoning result:")
    logger.info(f"Violations: {result['policy_violations']}")
    logger.info(f"Reasoning: {result['detailed_reasoning']}")
    logger.info(f"Severity: {result['severity_assessment']}")

@pytest.mark.asyncio
async def test_neutral_content_reasoning(policy_agent, test_policy_documents):
    """Test policy reasoning for neutral content"""
    test_text = "The weather is nice today."
    classification = {
        "classification": "neutral",
        "confidence": "0.95",
        "reason": "The text is neutral.",
        "severity": "low"
    }
    
    result = await policy_agent.generate_policy_reasoning(
        original_text=test_text,
        classification_result=classification,
        relevant_policies=test_policy_documents
    )
    
    assert result is not None
    assert len(result["policy_violations"]) <= 1
    # Update assertion to match actual API response
    assert any(["none" in violation.lower() or "no violation" in violation.lower() 
               for violation in result["policy_violations"]])
    assert "low" in result["severity_assessment"].lower()
    logger.info(f"Neutral content result: {result}")

@pytest.mark.asyncio
async def test_response_parsing(policy_agent):
    """Test response parsing functionality"""
    mock_response = """
    ### Policy Violations
    1. Promotes hate against protected group
    2. Calls for discriminatory action
    
    ### Detailed Reasoning
    The content explicitly calls for exclusion.
    
    ### Supporting Evidence
    Relevant policy section violated: "Hate speech targeting groups"
    
    ### Severity Assessment
    High severity due to explicit discrimination
    
    ### Additional Context
    Historical context suggests pattern of similar violations
    """
    
    result = policy_agent._parse_response(mock_response)
    
    assert len(result["policy_violations"]) == 2
    assert "Promotes hate against protected group" in result["policy_violations"]
    assert "content explicitly calls for exclusion" in result["detailed_reasoning"]
    assert "High severity" in result["severity_assessment"]
    logger.info(f"Parsed response: {result}")

@pytest.mark.asyncio
async def test_fallback_reasoning(policy_agent, test_classification_result):
    """Test fallback reasoning mechanism"""
    result = policy_agent._get_fallback_reasoning(test_classification_result)
    
    assert result is not None
    assert "policy_violations" in result
    assert "Unable to determine specific violations" in result["policy_violations"]
    assert result["original_classification"] == test_classification_result
    logger.info(f"Fallback reasoning: {result}")