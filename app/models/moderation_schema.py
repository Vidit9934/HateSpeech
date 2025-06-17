from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from .base_schema import ClassificationLabel, ModerationAction, Severity, Platform
from .policy_schema import PolicyReasoning
from .health_schema import ErrorInfo

class ClassificationResult(BaseModel):
    """Model for hate speech classification results."""
    label: ClassificationLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: Severity
    reasoning: str
    detected_categories: List[str] = Field(default_factory=list)
    keywords_flagged: List[str] = Field(default_factory=list)

class ActionRecommendation(BaseModel):
    """Model for action recommendations."""
    primary_action: ModerationAction
    alternative_actions: List[ModerationAction] = Field(default_factory=list)
    justification: str
    escalation_needed: bool = Field(default=False)
    follow_up_actions: List[str] = Field(default_factory=list)
    ban_duration: Optional[int] = Field(default=None)

class ModerationRequest(BaseModel):
    """Request model for content moderation"""
    text: str
    platform: Optional[str] = Field(default="GENERAL")
    metadata: Optional[dict] = Field(default_factory=dict)

class ModerationResponse(BaseModel):
    """Final comprehensive moderation response."""
    request_id: str
    original_text: str
    platform: Platform
    classification: ClassificationResult
    policy_analysis: "PolicyReasoning"  # Forward reference
    recommended_action: ActionRecommendation
    processing_time_ms: float
    agents_used: List[str] = Field(default_factory=list)
    confidence_overall: float = Field(..., ge=0.0, le=1.0)
    errors: List["ErrorInfo"] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)