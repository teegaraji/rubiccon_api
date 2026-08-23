import time
from datetime import UTC, datetime

from fastapi import APIRouter

from app.config import settings
from app.schemas.common import ApiMeta, ApiResponse
from app.schemas.validate import ValidateRequest, ValidateResponseData
from app.services.validator import validate_cube_state

router = APIRouter()


@router.post("/validate", response_model=ApiResponse[ValidateResponseData])
async def validate_state(request: ValidateRequest) -> ApiResponse[ValidateResponseData]:
    start_time = time.perf_counter()
    data = validate_cube_state(request.state)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return ApiResponse(
        success=True,
        data=data,
        meta=ApiMeta(
            timestamp=datetime.now(UTC).isoformat(),
            executionTimeMs=elapsed_ms,
            version=settings.VERSION,
        ),
    )
