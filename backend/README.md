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

