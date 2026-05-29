from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ml_service import get_model_info, predict_leaf_disease


ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_OUT_DIR = ROOT_DIR / "frontend" / "out"
FRONTEND_INDEX = FRONTEND_OUT_DIR / "index.html"

app = FastAPI(
    title="Plant Disease AI API",
    description="Prediction and saliency API for plant leaf disease detection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False, response_model=None)
def root():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)

    return {"message": "Plant Disease AI API", "docs": "/docs"}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/classes")
def classes() -> dict[str, object]:
    return get_model_info()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit etre une image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image vide.")

    try:
        return predict_leaf_disease(image_bytes, filename=file.filename or "leaf-image")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction impossible: {exc}") from exc


if (FRONTEND_OUT_DIR / "_next").exists():
    app.mount(
        "/_next",
        StaticFiles(directory=FRONTEND_OUT_DIR / "_next"),
        name="next-static",
    )


@app.get("/{full_path:path}", include_in_schema=False, response_model=None)
def serve_frontend(full_path: str):
    requested_path = FRONTEND_OUT_DIR / full_path

    if requested_path.is_file():
        return FileResponse(requested_path)

    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)

    raise HTTPException(status_code=404, detail="Frontend build introuvable.")
