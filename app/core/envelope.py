from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.core.exceptions import CubeValidationError, SolverExecutionError, SolverTimeoutError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(CubeValidationError)
    async def cube_validation_exception_handler(
        request: Request, exc: CubeValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details if exc.details else None,
                },
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "executionTimeMs": 0.0,
                    "version": settings.VERSION,
                },
            },
        )

    @app.exception_handler(SolverTimeoutError)
    async def solver_timeout_exception_handler(
        request: Request, exc: SolverTimeoutError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": None,
                },
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "executionTimeMs": 0.0,
                    "version": settings.VERSION,
                },
            },
        )

    @app.exception_handler(SolverExecutionError)
    async def solver_execution_exception_handler(
        request: Request, exc: SolverExecutionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details if exc.details else None,
                },
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "executionTimeMs": 0.0,
                    "version": settings.VERSION,
                },
            },
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = []
        for err in exc.errors():
            loc = ".".join(str(item) for item in err.get("loc", []))
            details.append(
                {
                    "field": loc if loc else "body",
                    "reason": err.get("msg", "Invalid field"),
                }
            )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_PAYLOAD_FORMAT",
                    "message": "The request body failed schema validation.",
                    "details": details,
                },
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "executionTimeMs": 0.0,
                    "version": settings.VERSION,
                },
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": str(exc.detail),
                    "details": None,
                },
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "executionTimeMs": 0.0,
                    "version": settings.VERSION,
                },
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": f"An unexpected error occurred: {str(exc)}",
                    "details": None,
                },
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "executionTimeMs": 0.0,
                    "version": settings.VERSION,
                },
            },
        )
