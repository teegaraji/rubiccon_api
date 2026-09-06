import os
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from app.api.v1.router import api_v1_router
from app.config import settings
from app.core.envelope import register_exception_handlers

# 1. Define Gradio Dashboard for Hugging Face Space UI
with gr.Blocks(title="Rubiccon Solver & Vision API") as demo:
    gr.Markdown("# 🎲 Rubiccon Solver & Vision Backend API")
    gr.Markdown(
        """
        Backend microservice powered by **FastAPI**, **Kociemba Two-Phase Algorithm**, 
        and **Real-time Computer Vision WebSocket Stream**.

        ### 📡 Available Endpoints:
        - **Interactive API Docs (Swagger UI):** [`/docs`](/docs)
        - **Alternative API Docs (ReDoc):** [`/redoc`](/redoc)
        - **Health Check:** [`/api/v1/health`](/api/v1/health)
        - **Cube Solver:** `POST /api/v1/solve`
        - **Cube State Validator:** `POST /api/v1/validate`
        - **Camera Vision Stream:** `wss://<this-space-domain>/api/v1/ws/vision`

        ---
        *Status: Healthy & Active*
        """
    )

# 2. Attach FastAPI routes & middleware directly to Gradio's underlying FastAPI app
app = demo.app

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

# Register custom exception handlers
register_exception_handlers(app)

# Include API Router (/api/v1)
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


# Swagger Docs & OpenAPI endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{settings.PROJECT_NAME} - Swagger UI",
    )


@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{settings.PROJECT_NAME} - ReDoc",
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Rubiccon Rubik's Cube 3x3 Solver Backend API powered by FastAPI "
            "& Kociemba Two-Phase Algorithm."
        ),
        routes=app.routes,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
