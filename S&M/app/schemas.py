# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class IngestEvent(BaseModel):
    timestamp: Optional[datetime] = None
    source: Optional[str] = Field(None, max_length=128)
    event_type: Optional[str] = Field(None, max_length=512)
    level: Optional[str] = Field(None, max_length=16)
    message: Optional[str] = None

    model_config = {
        "extra": "forbid"  # optional: forbid extra fields
    }

