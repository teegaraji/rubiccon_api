from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiMeta(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    executionTimeMs: float = 0.0
    version: str = "1.0.0"


class ApiErrorDetail(BaseModel):
    field: str | None = None
    face: str | None = None
    tileIndex: int | None = None
    reason: str


class ApiErrorPayload(BaseModel):
    code: str
    message: str
    details: list[ApiErrorDetail] | None = None


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: ApiErrorPayload | None = None
    meta: ApiMeta | None = None
