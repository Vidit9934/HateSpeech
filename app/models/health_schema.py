from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
from datetime import datetime

class SystemHealth(BaseModel):
    """Model for system health and monitoring."""
    status: str = Field(default="healthy")
    agents_status: Dict[str, str] = Field(default_factory=dict)
    vector_store_status: str = Field(default="unknown")
    model_status: str = Field(default="unknown")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class ErrorInfo(BaseModel):
    """Model for error information."""
    error_type: str
    error_message: str
    recovery_attempted: bool = False
    fallback_result: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)