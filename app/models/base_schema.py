from enum import Enum

class ClassificationLabel(str, Enum):
    """Enum for hate speech classification labels."""
    HATE = "hate"
    TOXIC = "toxic"
    OFFENSIVE = "offensive"
    NEUTRAL = "neutral"
    AMBIGUOUS = "ambiguous"
    UNCERTAIN = "uncertain"

class ModerationAction(str, Enum):
    """Enum for recommended moderation actions."""
    NO_ACTION = "no_action"
    WARNING = "warning" 
    REMOVE = "remove"
    HUMAN_REVIEW = "human_review"
    IMMEDIATE_REMOVAL = "immediate_removal"
    CONTENT_REMOVAL = "content_removal"
    TEMPORARY_BAN = "temporary_ban"
    PERMANENT_BAN = "permanent_ban"
    MANUAL_REVIEW = "manual_review"
    EDUCATIONAL_PROMPT = "educational_prompt"

class Severity(str, Enum):
    """Enum for content severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Platform(str, Enum):
    """Enum for different platforms."""
    REDDIT = "reddit"
    META = "meta"
    GOOGLE = "google"
    INDIAN_PENAL = "indian_penal"
    US_LAWS = "us_laws"
    GENERAL = "general"
    SOCIAL = "social"
    FORUM = "forum"
    CHAT = "chat"

class ErrorType(Enum):
    API_ERROR = "api_error"
    CLASSIFICATION_ERROR = "classification_error"
    RETRIEVAL_ERROR = "retrieval_error"
    REASONING_ERROR = "reasoning_error"
    ACTION_ERROR = "action_error"
    SYSTEM_ERROR = "system_error"