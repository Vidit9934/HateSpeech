import pytest
import pytest_asyncio
from app.agents.action_recommender import ActionRecommenderAgent
from app.models.moderation_schema import ModerationAction
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestActionRecommender")

@pytest_asyncio.fixture(scope="function")
async def action_recommender():
    """Fixture to create action recommender agent"""
    agent = ActionRecommenderAgent()
    return agent

@pytest.mark.asyncio
async def test_agent_initialization(action_recommender):
    """Test if agent initializes properly"""
    assert action_recommender is not None
    assert action_recommender.model is not None
    assert action_recommender.action_matrix is not None
    logger.info("Agent initialized successfully")

@pytest.mark.asyncio
async def test_hate_speech_action(action_recommender):
    """Test action recommendation for hate speech"""
    classification = {
        "label": "hate",
        "severity": "high",
        "confidence": 0.9
    }
    reasoning = {
        "policy_violations": ["hate content detected"],
        "severity_assessment": "high severity content"
    }
    
    result = await action_recommender.recommend_action(classification, reasoning)
    
    assert result is not None
    assert result["primary_action"] == ModerationAction.IMMEDIATE_REMOVAL
    assert result["escalation_needed"] is True
    logger.info(f"Hate speech action result: {result}")

@pytest.mark.asyncio
async def test_toxic_content_action(action_recommender):
    """Test action recommendation for toxic content"""
    classification = {
        "label": "toxic",
        "severity": "medium",
        "confidence": 0.9
    }
    reasoning = {
        "policy_violations": ["toxic content detected"],
        "severity_assessment": "medium severity content"
    }
    
    result = await action_recommender.recommend_action(classification, reasoning)
    
    assert result is not None
    assert result["primary_action"] == ModerationAction.WARNING
    logger.info(f"Toxic content action result: {result}")

@pytest.mark.asyncio
async def test_neutral_content_action(action_recommender):
    """Test action recommendation for neutral content"""
    classification = {
        "label": "neutral",
        "severity": "low",
        "confidence": 0.9
    }
    reasoning = {
        "policy_violations": [],
        "severity_assessment": "low severity content"
    }
    
    result = await action_recommender.recommend_action(classification, reasoning)
    
    assert result is not None
    assert result["primary_action"] == ModerationAction.NO_ACTION
    assert result["escalation_needed"] is False
    logger.info(f"Neutral content action result: {result}")

@pytest.mark.asyncio
async def test_ambiguous_content_action(action_recommender):
    """Test action recommendation for ambiguous content"""
    classification = {
        "label": "ambiguous",
        "severity": "medium",
        "confidence": 0.9
    }
    reasoning = {
        "policy_violations": ["unclear content"],
        "severity_assessment": "medium severity"
    }
    
    result = await action_recommender.recommend_action(classification, reasoning)
    
    assert result is not None
    assert result["primary_action"] == ModerationAction.HUMAN_REVIEW
    assert result["escalation_needed"] is True
    logger.info(f"Ambiguous content action result: {result}")

@pytest.mark.asyncio
async def test_error_handling(action_recommender):
    """Test error handling with invalid input"""
    classification = {
        "label": "invalid_label",
        "severity": "invalid_severity"
    }
    reasoning = {
        "policy_violations": [],
        "severity_assessment": ""
    }
    
    result = await action_recommender.recommend_action(classification, reasoning)
    
    assert result is not None
    assert result["primary_action"] == ModerationAction.HUMAN_REVIEW
    assert result["escalation_needed"] is True
    logger.info(f"Error handling result: {result}")