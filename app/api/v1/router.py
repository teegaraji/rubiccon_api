from fastapi import APIRouter

from app.api.v1.endpoints import health, solve, validate

api_v1_router = APIRouter()

api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(validate.router, tags=["Validation"])
api_v1_router.include_router(solve.router, tags=["Solver"])
