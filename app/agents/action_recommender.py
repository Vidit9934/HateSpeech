from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, Any, Union
from app.utils.config import Config
from app.models.schemas import ActionRecommendation, ModerationAction, ClassificationLabel, Severity
import logging

# Set up standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ActionRecommenderAgent")


class ActionRecommenderAgent:
    """Simple action recommender that maps classifications to moderation actions"""
    
    def __init__(self):
        # Define direct mappings from classification+severity to actions
        self.action_matrix = {
            # Hate Speech Actions
            ("hate", "high"): (ModerationAction.IMMEDIATE_REMOVAL, "Severe hate speech detected - immediate removal required"),
            ("hate", "medium"): (ModerationAction.REMOVE, "Hate speech violation - content removal with warning"),
            ("hate", "low"): (ModerationAction.WARNING, "Borderline hate speech - warning issued"),
            
            # Toxic Content Actions
            ("toxic", "high"): (ModerationAction.REMOVE, "Severe toxic content - removal required"),
            ("toxic", "medium"): (ModerationAction.WARNING, "Toxic content - warning issued"),
            ("toxic", "low"): (ModerationAction.HUMAN_REVIEW, "Mild toxic content - review needed"),
            
            # Offensive Content Actions
            ("offensive", "high"): (ModerationAction.WARNING, "Highly offensive content - warning issued"),
            ("offensive", "medium"): (ModerationAction.HUMAN_REVIEW, "Potentially offensive - review needed"),
            ("offensive", "low"): (ModerationAction.NO_ACTION, "Mildly offensive - monitoring only"),
            
            # Default Actions
            ("neutral", "*"): (ModerationAction.NO_ACTION, "Content appears safe"),
            ("ambiguous", "*"): (ModerationAction.HUMAN_REVIEW, "Unable to determine - human review needed"),
            ("uncertain", "*"): (ModerationAction.HUMAN_REVIEW, "Classification uncertain - human review needed")
        }

        try:
            self.model = AzureChatOpenAI(
                openai_api_version=Config.DIAL_API_VERSION,
                azure_deployment=Config.PRIMARY_MODEL_NAME,
                azure_endpoint=Config.DIAL_API_ENDPOINT,
                api_key=Config.DIAL_API_KEY,
                max_tokens=200,
                temperature=0.1,
            )
            logger.info(
                "Successfully initialized ActionRecommenderAgent with Azure OpenAI"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ActionRecommenderAgent: {str(e)}")
            raise

    async def recommend_action(
        self, classification_result: Dict[str, Any], reasoning_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recommend a moderation action based on classification and severity
        """
        try:
            # Extract classification and severity
            label = str(classification_result.get('label', 'uncertain')).lower()
            severity = str(classification_result.get('severity', 'medium')).lower()
            
            logger.info(f"Determining action for {label} content with {severity} severity")

            # Look up action in matrix, first try exact match
            action_tuple = self.action_matrix.get(
                (label, severity),
                # Fallback to wildcard severity match
                self.action_matrix.get(
                    (label, '*'),
                    # Ultimate fallback
                    (ModerationAction.HUMAN_REVIEW, "Unable to determine action - human review needed")
                )
            )

            # Create recommendation
            recommendation = ActionRecommendation(
                primary_action=action_tuple[0],
                alternative_actions=[],
                justification=action_tuple[1],
                escalation_needed=action_tuple[0] in [ModerationAction.HUMAN_REVIEW, ModerationAction.IMMEDIATE_REMOVAL],
                follow_up_actions=[],
                ban_duration=None
            )

            logger.info(f"Recommended action: {recommendation.primary_action}")
            return recommendation.model_dump()

        except Exception as e:
            logger.error(f"Error recommending action: {str(e)}")
            # Return safe fallback
            return ActionRecommendation(
                primary_action=ModerationAction.HUMAN_REVIEW,
                justification="Error occurred - human review required",
                escalation_needed=True
            ).model_dump()