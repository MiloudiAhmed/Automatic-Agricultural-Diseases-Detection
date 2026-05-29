# FastAPI Backend

FastAPI backend for plant leaf disease prediction with TensorFlow/Keras.

## Run

From the project root:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Default Model Files

The backend loads:

- `../models/efficientnet_model.h5`
- `../models/class_names.txt`

Optional environment variables:

- `PLANT_MODEL_PATH`
- `PLANT_CLASS_NAMES_PATH`

## Main Routes

- `GET /api/health`
- `GET /api/classes`
- `POST /api/predict` with a multipart `file` field

## Single-Service Deployment

When `frontend/out` exists, FastAPI also serves the frontend:

- `GET /` returns the Next.js interface
- `GET /_next/*` serves static frontend assets
- `/api/*` stays reserved for backend routes

Build the frontend first:

```powershell
cd ..\frontend
npm run build
```

Then run from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
