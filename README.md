# Traffic Violation Detection System

A highly modular, production-oriented **Traffic Violation Detection System**
built with **FastAPI**, **YOLOv8**, **EasyOCR**, **SQLAlchemy**, **PostgreSQL**,
and **Docker**.

The system performs License Plate Recognition (LPR) and evaluates contextual
metadata against a pluggable rule engine. A core design principle is the
**Human-in-the-Loop constraint**: the AI **never issues a fine directly**.
Every detected violation is persisted with status `pending_human_review`, and
only a human operator can `approve` or `reject` it.

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
   image +      │                 FastAPI API                   │
   metadata ───▶│  /api/v1/analyze                              │
                │      │                                        │
                │      ▼                                        │
                │  AnalysisService (orchestrator)               │
                │   ├─ ImageLoader (base64 / URL)               │
                │   ├─ DetectionService  → YOLOv8 (vehicles)    │
                │   ├─ OCRService        → EasyOCR (plates)     │
                │   └─ RuleEngine        → metadata rules       │
                │      │                                        │
                │      ▼  (violation? -> pending_human_review)  │
                │   CRUD / SQLAlchemy ──────────────▶ PostgreSQL│
                │                                               │
   operator ───▶│  /api/v1/review/{id}  (approve | reject)      │
                └──────────────────────────────────────────────┘
```

Each capability lives in its own module so it can be tested, swapped, or scaled
independently.

---

## Project structure

```
traffic-violation-system/
├── app/
│   ├── main.py                       # FastAPI app factory & lifespan
│   ├── api/
│   │   ├── deps.py                   # Dependency providers
│   │   └── v1/
│   │       ├── router.py             # Aggregates v1 routers
│   │       └── endpoints/
│   │           ├── analyze.py        # POST /api/v1/analyze
│   │           ├── review.py         # GET/POST /api/v1/review/{id}
│   │           └── health.py         # Liveness / DB readiness
│   ├── core/
│   │   ├── config.py                 # Pydantic settings
│   │   ├── enums.py                  # Domain enums (status, types)
│   │   └── logging_config.py         # Logging setup
│   ├── db/
│   │   ├── session.py                # Engine, SessionLocal, Base
│   │   └── init_db.py                # create_all on startup
│   ├── models/                       # SQLAlchemy ORM models
│   │   ├── vehicle.py
│   │   ├── violation.py
│   │   └── fine.py
│   ├── schemas/                      # Pydantic request/response models
│   │   ├── analysis.py
│   │   ├── vehicle.py
│   │   ├── violation.py
│   │   └── fine.py
│   ├── services/                     # Business logic & ML
│   │   ├── detection_service.py      # YOLOv8 wrapper (singleton)
│   │   ├── ocr_service.py            # EasyOCR wrapper (singleton)
│   │   ├── rule_engine.py            # Pluggable rule evaluation
│   │   ├── analysis_service.py       # Pipeline orchestration
│   │   └── crud.py                   # Persistence layer
│   └── utils/
│       ├── image_loader.py           # base64 / URL → numpy array
│       └── plate_utils.py            # Plate text normalisation
├── tests/
│   └── test_rule_engine.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
└── README.md
```

---

## Quick start (Docker)

```bash
# 1. (optional) configure environment
cp .env.example .env

# 2. build & run everything (API + PostgreSQL)
docker-compose up --build
```

The API will be available at **http://localhost:8000** and interactive docs at
**http://localhost:8000/docs**.

> On first run the API container downloads the YOLOv8 (`yolov8n.pt`) and EasyOCR
> weights. These are cached in the `model_cache` Docker volume, so subsequent
> startups are fast.

---

## Data model

| Table        | Purpose                                                                 |
|--------------|-------------------------------------------------------------------------|
| `vehicles`   | One row per recognised license plate.                                   |
| `violations` | Detected violations. Always created as `pending_human_review`.          |
| `fines`      | One-to-one with a violation; created **only** when an operator approves.|

`violations.status` transitions:

```
pending_human_review ──(operator approve)──▶ approved  (optional fine issued)
                     └─(operator reject) ──▶ rejected
```

---

## API reference

### `POST /api/v1/analyze`

Detect vehicles, read the plate, and evaluate the rule engine.

Request (base64 **or** URL, plus metadata):

```json
{
  "image_url": "https://example.com/car.jpg",
  "metadata": { "speed": 80, "speed_limit": 60, "location": "Main St" }
}
```

Response (violation flagged → pending review):

```json
{
  "detected_vehicles": [
    {
      "vehicle_type": "car",
      "detection_confidence": 0.91,
      "bounding_box": [120, 60, 540, 380],
      "license_plate": "ABC1234",
      "plate_confidence": 0.87
    }
  ],
  "primary_license_plate": "ABC1234",
  "violation_detected": true,
  "violation": {
    "id": 1,
    "status": "pending_human_review",
    "violation_type": "over_speeding",
    "detected_speed_kmh": 80,
    "speed_limit_kmh": 60
  },
  "message": "Violation detected and recorded with status 'pending_human_review' ..."
}
```

### `GET /api/v1/review`

List violations (filterable by `?status=pending_human_review`).

### `GET /api/v1/review/{violation_id}`

Fetch a single violation with its vehicle and fine.

### `POST /api/v1/review/{violation_id}`

Human operator decision. **This is the only path that can move a violation out
of `pending_human_review` and the only path that can create a fine.**

```json
{
  "decision": "approved",
  "reviewed_by": "operator_jane",
  "review_notes": "Clear evidence of over-speeding.",
  "fine_amount": 150.0,
  "fine_currency": "USD"
}
```

---

## Example requests (curl)

```bash
# Analyze (will flag a violation because 80 > 60)
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/car.jpg",
       "metadata":{"speed":80,"speed_limit":60,"location":"Main St"}}'

# Review queue
curl "http://localhost:8000/api/v1/review?status=pending_human_review"

# Approve violation #1 and issue a fine
curl -X POST http://localhost:8000/api/v1/review/1 \
  -H "Content-Type: application/json" \
  -d '{"decision":"approved","reviewed_by":"operator_jane","fine_amount":150}'
```

---

## Running tests

```bash
pip install -r requirements.txt pytest
pytest -q
```

The rule-engine tests run without any ML dependencies or a database.

---

## Design notes & extensibility

- **Human-in-the-loop is enforced structurally.** The AI pipeline calls
  `crud.create_pending_violation`, which hard-codes the
  `pending_human_review` status. There is no code path for the AI to set
  `approved`/`rejected` or to create a `Fine`.
- **Adding new rules** is as simple as writing a function
  `(AnalysisMetadata) -> RuleResult | None` and appending it to the `_RULES`
  registry in `rule_engine.py` (e.g. red-light, no-helmet).
- **Models are loaded lazily** as thread-safe singletons, so import time stays
  fast and weights load once per process.
- **Schema migrations**: `init_db` uses `create_all` for simplicity; for
  production, introduce Alembic for versioned migrations.

---

## Mongolian plates: custom weights & OCR

Out of the box the default YOLOv8 (`yolov8n.pt`, COCO) only **finds vehicles**;
it is not trained to read Mongolian plates. Recognition accuracy on Mongolian
(Cyrillic) plates therefore depends on two configurable, swappable stages:

| Stage | Component | Default | For Mongolian plates |
|-------|-----------|---------|----------------------|
| 1. Vehicle detection | `detection_service.py` (`YOLO_MODEL_PATH`) | `yolov8n.pt` | No change needed — cars are cars. |
| 2. Plate localisation | `plate_detection_service.py` (`PLATE_MODEL_PATH`) | *disabled* | **Add a custom YOLOv8 `.pt` trained to find the plate region.** |
| 3. Text recognition (OCR) | `ocr_service.py` (`OCR_LANGUAGES`, `OCR_ALLOWLIST`) | `["en"]` | Set `OCR_LANGUAGES=mn,en` and optionally an allow-list. |

### How to plug in custom weights (no code changes)

1. Put your trained weights in the host `models/` folder, e.g.
   `models/mn_plate_yolov8.pt`. (This folder is mounted read-only into the
   container at `/app/models`.)
2. In `.env` set:
   ```env
   PLATE_MODEL_PATH=/app/models/mn_plate_yolov8.pt
   OCR_LANGUAGES=mn,en
   # optional: constrain recognised characters
   # OCR_ALLOWLIST=ABCEHKMOPTYABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ
   ```
3. `docker-compose up --build`.

When `PLATE_MODEL_PATH` is set, the pipeline first crops the vehicle, then runs
the custom detector to isolate the **tight plate region**, then OCRs only that
region — which is far more accurate than OCR-ing the whole car. When it is empty,
the system gracefully falls back to OCR on the full vehicle crop.

### Notes on the recognition layer

- The plate-text normaliser (`plate_utils.py`) now **preserves Cyrillic**
  characters (`\u0400-\u04FF`), so plates like `1234 УБА` survive cleaning.
- EasyOCR ships a Cyrillic Mongolian model (`mn`). For the highest accuracy on
  Mongolian plate fonts you may eventually train a dedicated recognition model;
  the `OCRService` is isolated so it can be swapped without touching the API.
- This staged design means upgrading to Mongolian-specific weights is purely a
  **configuration change** — the API contract and database schema are unchanged.
