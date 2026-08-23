# Rubiccon Backend API (FastAPI & Kociemba Solver)

> High-performance microservice providing Rubik's Cube 3x3 mathematical state validation and solving capabilities powered by **FastAPI** and the **Kociemba Two-Phase Algorithm**.

---

## 🚀 Features

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

### `POST /api/v1/solve`
Computes the move sequence to solve the cube.

**Request:**
```json
{
  "state": "BBURUDBFUFFFRRFUUFLULUFUDLRRDBBDBDBLUDDFLLRRBRLLLBRDDF",
  "maxDepth": 24,
  "timeoutSeconds": 2.0
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "isSolved": false,
    "totalMoves": 21,
    "solutionString": "B U' L' D' R' D' L2 D' L F' L' D F2 R2 U R2 B2 U2 L2 F2 D'",
    "moves": [
      {
        "index": 1,
        "notation": "B",
        "face": "B",
        "direction": "CW",
        "angle": 90,
        "instruction": "Rotate Back face clockwise 90°",
        "humanGuidance": "Putar sisi BELAKANG searah jarum jam 90°"
      }
    ],
    "phase1Depth": 8,
    "phase2Depth": 13
  },
  "meta": {
    "timestamp": "2026-08-23T14:20:00.000Z",
    "executionTimeMs": 14.2,
    "version": "1.0.0"
  }
}
```

### `POST /api/v1/validate`
Validates a 54-character state string mathematically without calculating moves.

**Request:**
```json
{
  "state": "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
}
```

---

## 🏛️ Daiva Vault Architectural Memory

- **PRD**: `daiva/10 Projects/Rubiccon/prd.md`
- **Architecture**: `daiva/10 Projects/Rubiccon/architecture.md`
- **API Spec**: `daiva/10 Projects/Rubiccon/api_spec.md`
