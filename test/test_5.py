import pytest
from app.agents.error_handler import ErrorHandlerAgent
from app.models.base_schema import ErrorType
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestErrorHandler")

@pytest.fixture
def error_handler():
    """Fixture to create error handler agent"""
    return ErrorHandlerAgent()

def test_agent_initialization(error_handler):
    """Test if agent initializes properly"""
    assert error_handler is not None
    assert error_handler.fallback_responses is not None
    # Only check for error types that exist in the agent
    expected_types = ["classification_error", "retrieval_error", "reasoning_error", 
                     "action_error", "system_error"]
    for err_type in expected_types:
        assert err_type in error_handler.fallback_responses
    logger.info("Agent initialized successfully")

def test_classification_error_handling(error_handler):
    """Test handling of classification errors"""
    error = Exception("Classification model failed")
    result = error_handler.handle_error(
        error=error,
        error_type=ErrorType.CLASSIFICATION_ERROR,
        context={"text": "test content"}
    )
    
    assert result is not None
    assert result.get("message") == "System error - please contact support"
    assert result.get("action") == "human_review"
    logger.info(f"Classification error result: {result}")

def test_retrieval_error_handling(error_handler):
    """Test handling of retrieval errors"""
    error = Exception("Vector store unavailable")
    result = error_handler.handle_error(
        error=error,
        error_type=ErrorType.RETRIEVAL_ERROR
    )
    
    assert result is not None
    assert result.get("message") == "System error - please contact support"
    assert result.get("action") == "human_review"
    logger.info(f"Retrieval error result: {result}")

def test_reasoning_error_handling(error_handler):
    """Test handling of reasoning errors"""
    error = Exception("Policy reasoning failed")
    result = error_handler.handle_error(
        error=error,
        error_type=ErrorType.REASONING_ERROR
    )
    
    assert result is not None
    assert result.get("message") == "System error - please contact support"
    assert result.get("action") == "human_review"
    logger.info(f"Reasoning error result: {result}")

def test_action_error_handling(error_handler):
    """Test handling of action recommendation errors"""
    error = Exception("Action recommendation failed")
    result = error_handler.handle_error(
        error=error,
        error_type=ErrorType.ACTION_ERROR
    )
    
    assert result is not None
    assert result.get("message") == "System error - please contact support"
    assert result.get("action") == "human_review"
    logger.info(f"Action error result: {result}")

def test_system_error_handling(error_handler):
    """Test handling of system errors"""
    error = Exception("Critical system error")
    result = error_handler.handle_error(
        error=error,
        error_type=ErrorType.SYSTEM_ERROR
    )
    
    assert result is not None
    assert result.get("message") == "System error - please contact support"
    assert result.get("action") == "human_review"
    logger.info(f"System error result: {result}")

def test_error_handler_failure(error_handler):
    """Test behavior when error handler itself fails"""
    error = Exception("Test error")
    error_type = None  # This will cause the error handler to fail
    
    result = error_handler.handle_error(error, error_type)
    
    assert result is not None
    assert result.get("message") == "System error - please contact support"
    assert result.get("action") == "human_review"
    logger.info(f"Error handler failure result: {result}")