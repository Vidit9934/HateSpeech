"""
Pydantic models for the hate speech detection system.
These models define the data structures used throughout the multi-agent pipeline.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class ClassificationLabel(str, Enum):
    """
    Enum for hate speech classification labels.
    Used by the hate speech detection agent to categorize content.
    """

    HATE = "hate"
    TOXIC = "toxic"
    OFFENSIVE = "offensive"
    NEUTRAL = "neutral"
    AMBIGUOUS = "ambiguous"
    UNCERTAIN = "uncertain"


class ModerationAction(str, Enum):
    """
    Enum for recommended moderation actions.
    Used by the action recommender agent to suggest appropriate responses.
    """
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
    """
    Enum for content severity levels.
    Used to quantify the seriousness of detected issues.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Platform(str, Enum):
    """
    Enum for different platforms with their own policies.
    Used to select appropriate policy documents for context.
    """

    REDDIT = "reddit"
    META = "meta"
    GOOGLE = "google"
    INDIAN_PENAL = "indian_penal"
    US_LAWS = "us_laws"
    GENERAL = "general"
    SOCIAL = "SOCIAL"
    FORUM = "FORUM"
    CHAT = "CHAT"


class ClassificationRequest(BaseModel):
    """
    Model for incoming classification requests.
    This is the main input to the hate speech detection system.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The text content to analyze for hate speech",
    )

    platform: Optional[Platform] = Field(
        default=Platform.GENERAL,
        description="The platform where this content appears (affects policy selection)",
    )

    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context about the content (user history, thread context, etc.)",
    )

    user_id: Optional[str] = Field(
        default=None,
        description="Anonymous user identifier for tracking repeat offenses",
    )

    timestamp: Optional[datetime] = Field(
        default_factory=datetime.utcnow, description="When the content was posted"
    )

    @validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class ClassificationResult(BaseModel):
    """
    Model for hate speech classification results.
    Output from the hate speech detection agent.
    """

    label: ClassificationLabel = Field(
        ..., description="The predicted classification label"
    )

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the classification (0-1)"
    )

    severity: Severity = Field(
        ..., description="Severity level of the detected content"
    )

    reasoning: str = Field(
        ..., description="Explanation of why this classification was chosen"
    )

    detected_categories: List[str] = Field(
        default_factory=list,
        description="Specific categories of hate speech detected (racism, sexism, etc.)",
    )

    keywords_flagged: List[str] = Field(
        default_factory=list,
        description="Specific words or phrases that contributed to the classification",
    )


class PolicyDocument(BaseModel):
    """
    Model for policy documents retrieved by the hybrid retriever agent.
    Represents relevant policy information for context.
    """

    title: str = Field(..., description="Title or identifier of the policy document")

    content: str = Field(..., description="The relevant policy text content")

    source: str = Field(
        ..., description="Source of the policy (reddit, meta, legal, etc.)"
    )

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score for this policy to the current content",
    )

    section: Optional[str] = Field(
        default=None, description="Specific section of the policy document"
    )

    policy_type: str = Field(
        ..., description="Type of policy (community guidelines, legal code, etc.)"
    )


class PolicyReasoning(BaseModel):
    """
    Model for policy-based reasoning results.
    Output from the policy reasoning agent.
    """

    applicable_policies: List[PolicyDocument] = Field(
        default_factory=list, description="List of relevant policy documents"
    )

    policy_violations: List[str] = Field(
        default_factory=list, description="Specific policy violations identified"
    )

    legal_implications: Optional[str] = Field(
        default=None, description="Potential legal implications if applicable"
    )

    precedent_cases: List[str] = Field(
        default_factory=list,
        description="Similar cases or precedents from policy documents",
    )

    reasoning_summary: str = Field(
        ..., description="Summary of the policy-based reasoning process"
    )

    # Fallback reasoning fields
    detailed_reasoning: Optional[str] = Field(
        default=None, description="Detailed reasoning for the fallback classification"
    )

    supporting_evidence: Optional[str] = Field(
        default=None, description="Supporting evidence for the fallback classification"
    )

    severity_assessment: Optional[str] = Field(
        default=None, description="Severity assessment for the fallback classification"
    )

    additional_context: Optional[str] = Field(
        default=None, description="Additional context for the fallback classification"
    )

    original_classification: Optional[Dict[str, str]] = Field(
        default=None, description="Original classification result for reference"
    )

    policies_analyzed: int = Field(
        default=0, description="Number of policies analyzed in the reasoning"
    )


class ActionRecommendation(BaseModel):
    """
    Model for action recommendations.
    Output from the action recommender agent.
    """

    primary_action: ModerationAction = Field(
        ..., description="Primary recommended moderation action"
    )

    alternative_actions: List[ModerationAction] = Field(
        default_factory=list, description="Alternative actions to consider"
    )

    justification: str = Field(
        ..., description="Explanation for why this action is recommended"
    )

    escalation_needed: bool = Field(
        default=False, description="Whether human moderator escalation is needed"
    )

    follow_up_actions: List[str] = Field(
        default_factory=list, description="Additional follow-up actions to consider"
    )

    ban_duration: Optional[int] = Field(
        default=None, description="Duration in hours for temporary bans"
    )


class ErrorInfo(BaseModel):
    """
    Model for error information from the error handler agent.
    """

    error_type: str = Field(..., description="Type of error encountered")

    error_message: str = Field(..., description="Human-readable error message")

    recovery_attempted: bool = Field(
        default=False, description="Whether automatic recovery was attempted"
    )

    fallback_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Fallback result if recovery was successful"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the error occurred"
    )


class ModerationRequest(BaseModel):
    """Request model for content moderation"""

    text: str = Field(..., description="Text content to be moderated")

    platform: Optional[str] = Field(
        default="GENERAL", description="Source platform of the content"
    )

    metadata: Optional[dict] = Field(
        default_factory=dict, description="Additional metadata about the request"
    )


class ModerationResponse(BaseModel):
    """
    Final comprehensive moderation response.
    This is the main output of the entire hate speech detection system.
    """

    # Request information
    request_id: str = Field(..., description="Unique identifier for this request")

    original_text: str = Field(..., description="The original text that was analyzed")

    platform: Platform = Field(..., description="Platform where the content was found")

    # Classification results
    classification: ClassificationResult = Field(
        ..., description="Hate speech classification results"
    )

    # Policy analysis
    policy_analysis: PolicyReasoning = Field(
        ..., description="Policy-based reasoning and analysis"
    )

    # Action recommendations
    recommended_action: ActionRecommendation = Field(
        ..., description="Recommended moderation actions"
    )

    # Processing metadata
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )

    agents_used: List[str] = Field(
        default_factory=list,
        description="List of agents that participated in the analysis",
    )

    confidence_overall: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence in the analysis"
    )

    # Error handling
    errors: List[ErrorInfo] = Field(
        default_factory=list, description="Any errors encountered during processing"
    )

    warnings: List[str] = Field(
        default_factory=list, description="Warnings or notices about the analysis"
    )

    # Timestamps
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the analysis was completed"
    )

    @validator("confidence_overall")
    def validate_overall_confidence(cls, v, values):
        # Overall confidence should be reasonable based on classification confidence
        if "classification" in values:
            class_conf = values["classification"].confidence
            if abs(v - class_conf) > 0.3:  # Allow some deviation but not too much
                return min(v, class_conf + 0.1)  # Cap the overall confidence
        return v


class SystemHealth(BaseModel):
    """
    Model for system health and monitoring.
    Used for API health checks and system monitoring.
    """

    status: str = Field(default="healthy", description="Overall system status")

    agents_status: Dict[str, str] = Field(
        default_factory=dict, description="Status of individual agents"
    )

    vector_store_status: str = Field(
        default="unknown", description="Status of the vector database"
    )

    model_status: str = Field(default="unknown", description="Status of the ML models")

    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the health check was last performed",
    )


class BatchRequest(BaseModel):
    """
    Model for batch processing requests.
    Allows processing multiple texts at once.
    """

    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of texts to analyze (max 100)",
    )

    platform: Optional[Platform] = Field(
        default=Platform.GENERAL, description="Platform for all texts in the batch"
    )

    batch_id: Optional[str] = Field(
        default=None, description="Optional batch identifier"
    )


class BatchResponse(BaseModel):
    """
    Model for batch processing responses.
    """

    batch_id: str = Field(..., description="Unique identifier for this batch")

    results: List[ModerationResponse] = Field(
        ..., description="List of moderation responses for each input text"
    )

    total_processed: int = Field(..., description="Total number of texts processed")

    failed_count: int = Field(
        default=0, description="Number of texts that failed processing"
    )

    processing_time_ms: float = Field(
        ..., description="Total batch processing time in milliseconds"
    )


# Configuration models for the system
class AgentConfig(BaseModel):
    """
    Configuration model for individual agents.
    """

    name: str
    enabled: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3
    model_name: Optional[str] = None
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


class SystemConfig(BaseModel):
    """
    Overall system configuration model.
    """

    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    vector_store_path: str = "./vector_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_model: str = "gpt-3.5-turbo"
    max_policy_docs: int = 5
    confidence_threshold: float = 0.7
    debug_mode: bool = False
