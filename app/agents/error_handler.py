import logging
from typing import Dict, Any
from datetime import datetime
from enum import Enum
from app.models.base_schema import ErrorType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ErrorHandlerAgent")


# class ErrorType(Enum):
#     API_ERROR = "api_error"
#     CLASSIFICATION_ERROR = "classification_error"
#     RETRIEVAL_ERROR = "retrieval_error"
#     REASONING_ERROR = "reasoning_error"
#     ACTION_ERROR = "action_error"
#     SYSTEM_ERROR = "system_error"


class ErrorHandlerAgent:
    """Simple error handler that provides graceful fallbacks and user feedback"""

    def __init__(self):
        # Basic fallback responses for different error types
        self.fallback_responses = {
            ErrorType.CLASSIFICATION_ERROR.value: {
                "classification": "uncertain",
                "confidence": 0.0,
                "reason": "Classification failed - human review needed",
            },
            ErrorType.RETRIEVAL_ERROR.value: {
                "relevant_policies": [],
                "message": "Could not retrieve policies - human review needed",
            },
            ErrorType.REASONING_ERROR.value: {
                "policy_violations": ["Unable to analyze violations"],
                "message": "Reasoning failed - human review needed",
            },
            ErrorType.ACTION_ERROR.value: {
                "action": "human_review",
                "reason": "Action recommendation failed",
            },
            ErrorType.SYSTEM_ERROR.value: {
                "message": "System error occurred",
                "action": "human_review",
            },
        }

    def handle_error(
        self, error: Exception, error_type: ErrorType, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle error and return appropriate fallback response"""

        try:
            # Log the error with context
            error_msg = f"Error in {error_type.value}: {str(error)}"
            if context:
                error_msg += f" | Context: {context}"
            logger.error(error_msg)

            # Get fallback response
            fallback = self.fallback_responses.get(
                error_type, self.fallback_responses[ErrorType.SYSTEM_ERROR]
            )

            # Add error info to response
            response = {
                **fallback,
                "error_info": {
                    "type": error_type.value,
                    "timestamp": datetime.now().isoformat(),
                    "message": str(error),
                },
            }

            return response

        except Exception as e:
            # Emergency fallback if error handling fails
            logger.error(f"Error handler failed: {str(e)}")
            return {
                "message": "System error - please contact support",
                "action": "human_review",
            }