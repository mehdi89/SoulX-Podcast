"""
Pydantic Data Models for API
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    """Generation request model (for JSON body with file upload)"""
    prompt_texts: List[str] = Field(
        ...,
        description="Reference text list, length should match uploaded audio files",
        min_items=1,
        max_items=4
    )
    dialogue_text: str = Field(
        ...,
        description="Dialogue text to generate. Single speaker: direct text. Multi-speaker: use [S1][S2] tags",
        min_length=1
    )
    seed: Optional[int] = Field(
        default=1988,
        description="Random seed for reproducibility"
    )
    temperature: Optional[float] = Field(
        default=0.6,
        ge=0.1,
        le=2.0,
        description="Sampling temperature"
    )
    top_k: Optional[int] = Field(
        default=100,
        ge=1,
        le=500,
        description="Top-K sampling parameter"
    )
    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-P sampling parameter"
    )
    repetition_penalty: Optional[float] = Field(
        default=1.25,
        ge=1.0,
        le=2.0,
        description="Repetition penalty coefficient"
    )

    @validator('dialogue_text')
    def validate_dialogue_text(cls, v):
        """Validate dialogue text format"""
        if not v.strip():
            raise ValueError("dialogue_text cannot be empty")
        return v.strip()


class TaskCreateResponse(BaseModel):
    """Async task creation response"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    created_at: datetime = Field(..., description="Task creation time")
    message: str = Field(default="Task created", description="Info message")


class TaskStatusResponse(BaseModel):
    """Task status query response"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Task status")
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    result_url: Optional[str] = Field(None, description="Result file download URL")
    error: Optional[str] = Field(None, description="Error message")
    created_at: datetime = Field(..., description="Task creation time")
    started_at: Optional[datetime] = Field(None, description="Task start time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "progress": 100,
                "result_url": "/download/123e4567-e89b-12d3-a456-426614174000.wav",
                "error": None,
                "created_at": "2025-11-01T12:00:00Z",
                "started_at": "2025-11-01T12:00:01Z",
                "completed_at": "2025-11-01T12:00:15Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    llm_engine: str = Field(..., description="Current LLM engine (hf/vllm)")
    active_tasks: int = Field(default=0, description="Number of active tasks")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error details")
    task_id: Optional[str] = Field(None, description="Related task ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
