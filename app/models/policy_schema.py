from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PolicyDocument(BaseModel):
    """Model for policy documents."""
    title: str
    content: str
    source: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    section: Optional[str] = None
    policy_type: str

class PolicyReasoning(BaseModel):
    """Model for policy-based reasoning results."""
    applicable_policies: List[PolicyDocument] = Field(default_factory=list)
    policy_violations: List[str] = Field(default_factory=list)
    legal_implications: Optional[str] = None
    precedent_cases: List[str] = Field(default_factory=list)
    reasoning_summary: str
    detailed_reasoning: Optional[str] = None
    supporting_evidence: Optional[str] = None
    severity_assessment: Optional[str] = None
    additional_context: Optional[str] = None
    original_classification: Optional[Dict[str, str]] = None
    policies_analyzed: int = 0