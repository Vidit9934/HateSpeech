import pytest
import pytest_asyncio
from app.agents.hate_speech_detection import HateSpeechDetectionAgent
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestHateSpeech")

@pytest_asyncio.fixture(scope="function")
async def hate_speech_agent():
    """Fixture to create hate speech detection agent"""
    agent = HateSpeechDetectionAgent()
    return agent

@pytest.mark.asyncio
async def test_agent_initialization(hate_speech_agent):
    """Test if agent initializes properly"""
    assert hate_speech_agent is not None
    assert hate_speech_agent.categories == ["Hate", "Toxic", "Offensive", "Neutral", "Ambiguous"]
    assert hate_speech_agent.model is not None
    logger.info("Agent initialized successfully")

@pytest.mark.asyncio
async def test_hate_speech_detection(hate_speech_agent):
    """Test hate speech detection on toxic content"""
    test_text = "I hate everyone!"
    result = await hate_speech_agent.classify_text(test_text)
    
    assert result is not None
    assert "classification" in result
    assert result["classification"] in hate_speech_agent.categories
    # Fix: Convert confidence to lowercase for comparison
    assert result["confidence"].lower() in ["high", "medium", "low"]
    assert "reason" in result
    
    logger.info(f"Test text: {test_text!r}")
    logger.info(f"Classification: {result['classification']}")
    logger.info(f"Confidence: {result['confidence']}")
    logger.info(f"Reason: {result['reason']}")

@pytest.mark.asyncio
async def test_neutral_content(hate_speech_agent):
    """Test detection on neutral content"""
    test_text = "The weather is nice today."
    result = await hate_speech_agent.classify_text(test_text)
    
    assert result is not None
    assert result["classification"] == "Neutral"
    # Fix: Convert confidence to lowercase for comparison
    assert result["confidence"].lower() in ["high", "medium", "low"]
    assert "reason" in result
    
    logger.info(f"Neutral test result: {result}")

@pytest.mark.asyncio
async def test_edge_cases(hate_speech_agent):
    """Test edge cases like empty string and special characters"""
    test_cases = ["", "   ", "!@#$%", "..."]
    
    for test_text in test_cases:
        result = await hate_speech_agent.classify_text(test_text)
        assert result is not None
        assert result["classification"] in hate_speech_agent.categories
        # Fix: Convert confidence to lowercase for comparison
        assert result["confidence"].lower() in ["high", "medium", "low"]
        logger.info(f"Edge case '{test_text}': {result}")

@pytest.mark.asyncio
async def test_model_response_parsing(hate_speech_agent):
    """Test the response parsing functionality"""
    # Match exact format from successful test outputs
    mock_response = """Classification: Toxic
Confidence: High
Brief Reason: Contains explicit hostile language"""
    
    result = hate_speech_agent._parse_classification_response(mock_response)
    assert result["classification"] == "Toxic"
    assert result["confidence"] == "High"  # Match exact case
    assert result["reason"] == "Contains explicit hostile language"
    logger.info(f"Parsed response: {result}")