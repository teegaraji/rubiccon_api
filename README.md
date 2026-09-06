---
title: Rubiccon Api
emoji: 🎲
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
---

# Rubiccon Backend API (FastAPI & Kociemba Solver)

> High-performance microservice providing Rubik's Cube 3x3 real-time computer vision frame analysis, mathematical state validation, and solving capabilities powered by **FastAPI** and the **Kociemba Two-Phase Algorithm**.

---

## 🚀 Features

- **Real-Time Vision Streaming via WebSocket (`/api/v1/ws/vision`)**:
  - Continuous camera frame processing (10–15 FPS) with 3x3 tile HSV/RGB color classification.
  - Interactive ROI alignment & stability score calculation (`isReadyToLock`).
  - Face state persistence (`init_session`, `set_active_face`, `lock_face`, `reset_session`).
  - Automatic completion trigger (`session_completed`) returning validated 54-char string.
- **Standard RESTful Endpoints (`/api/v1`)**:
  - `POST /api/v1/solve`: Computes optimal/sub-optimal move sequences with direction, rotation angle, and bilingual guidance (English & Indonesian).
  - `POST /api/v1/validate`: Pre-flight validation (character integrity, tile counts, centers invariant, edge/corner parity).
  - `GET /api/v1/health`: Service health and uptime monitoring.
- **Unified Response Envelope**: Standardized `{ success, data, error, meta }` schema across all endpoints.
- **Robust Error Taxonomy**: Precise validation failure codes (`CUBE_STATE_INVALID`, `COLOR_COUNT_MISMATCH`, `EDGE_FLIP_ERROR`, `CORNER_TWIST_ERROR`, `PERMUTATION_PARITY_ERROR`, `INVALID_PAYLOAD_FORMAT`, etc.).
- **CORS Configured**: Ready for frontend integration (`Next.js / React Three Fiber`).

---

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Data Validation & Settings**: [Pydantic v2](https://docs.pydantic.dev/) & `pydantic-settings`
- **Computer Vision**: [Pillow](https://python-pillow.org/) & [NumPy](https://numpy.org/)
- **Solver Engine**: [Kociemba Two-Phase Algorithm](https://github.com/muodov/kociemba)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/)
- **Testing & Quality**: [pytest](https://pytest.org/), [httpx](https://www.python-httpx.org/), [ruff](https://docs.astral.sh/ruff/)

---

## 📦 Getting Started

### 1. Prerequisites

- Python 3.12+ (or [uv](https://docs.astral.sh/uv/))

### 2. Setup Virtual Environment

Using `uv`:
```bash
# Create venv with Python 3.12
uv venv --python 3.12

# Activate virtualenv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

Using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 4. Running Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 🧪 Testing & Linting

```bash
# Run test suite
pytest -v

# Run linter
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

---

## 📡 API Reference

### 1. Real-Time Vision WebSocket (`ws://localhost:8000/api/v1/ws/vision`)

**Client Upstream Actions**:
- `init_session`: `{ "action": "init_session", "sessionId": "...", "initialFace": "Front" }`
- `process_frame`: `{ "action": "process_frame", "frameId": 1, "activeFace": "Front", "timestamp": 1724426400, "image": "data:image/jpeg;base64,..." }`
- `set_active_face`: `{ "action": "set_active_face", "face": "Right" }`
- `lock_face`: `{ "action": "lock_face", "face": "Front", "overrideTiles": null }`
- `reset_session`: `{ "action": "reset_session" }`

**Server Downstream Events**:
- `detection_result`: Returns 9 tile colors, hex, HSV/RGB values, stability score, and `isReadyToLock`.
- `face_locked`: Confirms face lock, returns completed faces and `nextSuggestedFace`.
- `session_completed`: Emitted when all 6 faces are mapped, returns 54-char string & validity status.
- `vision_error`: Emitted on malformed frames or processing errors.

### 2. REST Endpoints

#### `POST /api/v1/solve`
Computes the move sequence to solve the cube.

**Request:**
```json
{
  "state": "BBURUDBFUFFFRRFUUFLULUFUDLRRDBBDBDBLUDDFLLRRBRLLLBRDDF",
  "maxDepth": 24,
  "timeoutSeconds": 2.0
}
```

#### `POST /api/v1/validate`
Validates a 54-character state string mathematically without calculating moves.

**Request:**
```json
{
  "state": "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
}
```

#### `GET /api/v1/health`
Health check and microservice status.

---

## 🏛️ Daiva Vault Architectural Memory

- **PRD**: `daiva/10 Projects/Rubiccon/prd.md`
- **Architecture**: `daiva/10 Projects/Rubiccon/architecture.md`
- **API Spec**: `daiva/10 Projects/Rubiccon/api_spec.md`
