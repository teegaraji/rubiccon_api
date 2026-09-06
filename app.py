import os
import gradio as gr
from app.main import app as fastapi_app

# Gradio Dashboard UI for Hugging Face Space
with gr.Blocks(title="Rubiccon Solver & Vision API") as demo:
    gr.Markdown("#Rubiccon Solver & Vision Backend API")
    gr.Markdown(
        """
        Welcome to the **Rubiccon Solver API** backend running on Hugging Face Spaces!
        
        This service powers the Rubiccon 3D Web Application with real-time Rubik's cube solving 
        (Kociemba Two-Phase Algorithm) and computer vision camera detection via WebSockets.

        ### Service Information & Endpoints
        - **Interactive API Docs (Swagger UI):** [`/docs`](/docs)
        - **Alternative API Docs (ReDoc):** [`/redoc`](/redoc)
        - **Health Check Endpoint:** [`/api/v1/health`](/api/v1/health)
        - **Cube Solver Endpoint:** `POST /api/v1/solve`
        - **Cube Validator Endpoint:** `POST /api/v1/validate`
        - **Camera Vision Stream:** `wss://<this-space-domain>/api/v1/ws/vision`

        ---
        *Service is active and ready to receive requests.*
        """
    )

# Mount FastAPI app onto Gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
