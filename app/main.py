from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

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
        "& Kociemba Two-Phase Algorithm.\n\n"
        "### 📡 Protocols:\n"
        "- **RESTful API**: Standard JSON endpoints on `/api/v1`\n"
        "- **WebSocket Stream**: Real-time camera color detection on "
        "`ws://localhost:8000/api/v1/ws/vision`"
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


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Document WebSocket endpoint in OpenAPI schema for Swagger UI visibility
    openapi_schema.setdefault("paths", {})
    openapi_schema["paths"][f"{settings.API_V1_PREFIX}/ws/vision"] = {
        "get": {
            "tags": ["Vision WebSocket"],
            "summary": "Real-Time Camera Frame Color Detection (WebSocket)",
            "description": (
                "**WebSocket Endpoint:** `ws://localhost:8000/api/v1/ws/vision`\n\n"
                "Used for continuous live camera video stream analysis (~10 FPS).\n\n"
                "#### Upstream Client Messages (JSON):\n"
                "1. `init_session`: Initialize CV session (`sessionId`, `initialFace`).\n"
                "2. `process_frame`: Send video frame Base64 image + ROI guide.\n"
                "3. `set_active_face`: Change target face (Up, Right, Front, Down, Left, Back).\n"
                "4. `lock_face`: Confirm 9 tile colors for active face.\n"
                "5. `reset_session`: Reset all mapped faces.\n\n"
                "#### Downstream Server Events (JSON):\n"
                "1. `detection_result`: Returns 9 tile colors, stability score, & isReadyToLock.\n"
                "2. `face_locked`: Confirmation of locked face & nextSuggestedFace.\n"
                "3. `session_completed`: Triggered when 6 faces locked with 54-char string.\n"
                "4. `vision_error`: Emitted on bad payloads or frame decode issues."
            ),
            "responses": {
                "101": {
                    "description": "Switching Protocols to WebSocket (ws://...)",
                }
            },
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "api_v1": f"{settings.API_V1_PREFIX}/health",
        "ws_vision": f"ws://{settings.HOST}:{settings.PORT}{settings.API_V1_PREFIX}/ws/vision",
    }
