from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ApiMeta(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
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


class ApiResponse[T](BaseModel):
    success: bool
    data: T | None = None
    error: ApiErrorPayload | None = None
    meta: ApiMeta | None = None
