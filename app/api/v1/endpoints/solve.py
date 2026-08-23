import time
from datetime import UTC, datetime

from fastapi import APIRouter

from app.config import settings
from app.schemas.common import ApiMeta, ApiResponse
from app.schemas.solve import SolveRequest, SolveResponseData
from app.services.solver import solve_cube

router = APIRouter()


@router.post("/solve", response_model=ApiResponse[SolveResponseData])
async def solve_cube_endpoint(request: SolveRequest) -> ApiResponse[SolveResponseData]:
    start_time = time.perf_counter()
    data = solve_cube(
        state=request.state,
        max_depth=request.maxDepth,
        timeout_seconds=request.timeoutSeconds,
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
