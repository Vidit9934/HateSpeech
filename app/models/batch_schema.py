from pydantic import BaseModel, Field
from typing import List, Optional
from base_schema import Platform
from moderation_schema import ModerationResponse

class BatchRequest(BaseModel):
    """Model for batch processing requests."""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    platform: Optional[Platform] = Field(default=Platform.GENERAL)
    batch_id: Optional[str] = None

class BatchResponse(BaseModel):
    """Model for batch processing responses."""
    batch_id: str
    results: List[ModerationResponse]
    total_processed: int
    failed_count: int = 0
    processing_time_ms: float