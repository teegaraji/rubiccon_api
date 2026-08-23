from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_v1_router
from app.config import settings
from app.core.envelope import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pruning tables or warm-up can be triggered here if needed
    yield
    # Shutdown logic if needed


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Rubiccon Rubik's Cube 3x3 Solver Backend API powered by FastAPI "
        "& Kociemba Two-Phase Algorithm."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
origins = settings.ALLOWED_ORIGINS
if "*" in origins:
    allow_origins = ["*"]
else:
    allow_origins = origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register standard error envelope handlers
register_exception_handlers(app)

# Include API Router
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "api_v1": f"{settings.API_V1_PREFIX}/health",
    }
