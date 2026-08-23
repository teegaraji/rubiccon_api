import time
from datetime import UTC, datetime

from fastapi import APIRouter

from app.config import settings
from app.schemas.common import ApiMeta, ApiResponse
from app.schemas.health import HealthResponseData

router = APIRouter()
SERVER_START_TIME = time.time()


@router.get("/health", response_model=ApiResponse[HealthResponseData])
async def get_health() -> ApiResponse[HealthResponseData]:
    start_time = time.perf_counter()
    uptime = round(time.time() - SERVER_START_TIME, 2)

    data = HealthResponseData(
        status="healthy",
        engine="Kociemba Two-Phase Algorithm",
        version=settings.VERSION,
        pruningTablesLoaded=True,
        uptimeSeconds=uptime,
    )

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
