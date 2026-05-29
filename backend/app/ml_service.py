from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "efficientnet_model.h5"
DEFAULT_CLASS_NAMES_PATH = ROOT_DIR / "models" / "class_names.txt"


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    class_names: list[str]
    input_size: tuple[int, int]
    model_path: Path


def _resolve_path(env_name: str, default_path: Path) -> Path:
    value = os.getenv(env_name)
    return Path(value).expanduser().resolve() if value else default_path


def _read_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier classes introuvable: {path}")

    class_names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        raise ValueError("Le fichier class_names.txt est vide.")

    return class_names


def _model_input_size(model: Any) -> tuple[int, int]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1] if len(input_shape) > 1 else None
    width = input_shape[2] if len(input_shape) > 2 else None
    if not height or not width:
        return (224, 224)

    return (int(height), int(width))


@lru_cache(maxsize=1)
def load_bundle() -> ModelBundle:
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.models import load_model

    model_path = _resolve_path("PLANT_MODEL_PATH", DEFAULT_MODEL_PATH)
    class_names_path = _resolve_path("PLANT_CLASS_NAMES_PATH", DEFAULT_CLASS_NAMES_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Modele TensorFlow introuvable: {model_path}")

    model = load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
        safe_mode=False,
    )

    return ModelBundle(
        model=model,
        class_names=_read_class_names(class_names_path),
        input_size=_model_input_size(model),
        model_path=model_path,
    )


def get_model_info() -> dict[str, object]:
    bundle = load_bundle()
    return {
        "model": bundle.model_path.name,
        "input_size": {"height": bundle.input_size[0], "width": bundle.input_size[1]},
        "classes": bundle.class_names,
    }


def _load_image(image_bytes: bytes, size: tuple[int, int]) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Image invalide ou format non supporte.") from exc

    return image.resize((size[1], size[0]), Image.Resampling.LANCZOS)


def _image_to_batch(image: Image.Image) -> np.ndarray:
    image_array = np.asarray(image).astype("float32")
    return np.expand_dims(image_array, axis=0)


def _ensure_probabilities(predictions: np.ndarray) -> np.ndarray:
    values = predictions.astype("float64")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-3):
        values = np.exp(values - np.max(values))
        values = values / values.sum()
    return values


def _png_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype("float32")
    values = values - float(values.min())
    max_value = float(values.max())
    if max_value <= 1e-8:
        return np.zeros_like(values, dtype="float32")
    return np.clip(values / max_value, 0.0, 1.0)


def _heatmap_rgb(mask: np.ndarray) -> np.ndarray:
    red = np.clip(mask * 255.0, 0, 255)
    green = np.clip((1.0 - np.abs(mask * 2.0 - 1.0)) * 210.0, 0, 210)
    blue = np.clip((1.0 - mask) * 180.0, 0, 180)
    return np.stack([red, green, blue], axis=-1).astype("uint8")


def _saliency_images(model: Any, batch: np.ndarray, class_index: int, image: Image.Image) -> dict[str, str]:
    import tensorflow as tf

    image_tensor = tf.convert_to_tensor(batch)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor, training=False)
        score = predictions[:, class_index]

    gradients = tape.gradient(score, image_tensor)
    if gradients is None:
        raise ValueError("Impossible de calculer la saliency map pour ce modele.")

    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0].numpy()
    saliency = _normalize_map(saliency)

    original = np.asarray(image).astype("float32")
    heatmap = _heatmap_rgb(saliency).astype("float32")
    alpha = (0.18 + 0.52 * saliency)[..., np.newaxis]
    overlay = np.clip(original * (1.0 - alpha) + heatmap * alpha, 0, 255).astype("uint8")

    return {
        "heatmap": _png_data_url(Image.fromarray(heatmap.astype("uint8"))),
        "overlay": _png_data_url(Image.fromarray(overlay)),
    }


def _display_label(class_name: str) -> str:
    label = class_name.replace("___", " - ").replace("_", " ")
    return " ".join(part.capitalize() if part.islower() else part for part in label.split())


def _recommendation(class_name: str, confidence: float) -> str:
    if "healthy" in class_name.lower():
        return "Feuille saine detectee. Continuez la surveillance et gardez des conditions de culture stables."
    if confidence < 0.65:
        return "Prediction incertaine. Reprenez une photo nette, bien eclairee, avec une seule feuille au centre."
    return "Maladie probable detectee. Isolez les feuilles atteintes et confirmez le diagnostic avant traitement."


def predict_leaf_disease(image_bytes: bytes, filename: str) -> dict[str, Any]:
    bundle = load_bundle()
    image = _load_image(image_bytes, bundle.input_size)
    batch = _image_to_batch(image)

    raw_predictions = bundle.model(batch, training=False).numpy()[0]
    probabilities = _ensure_probabilities(raw_predictions)

    if len(probabilities) != len(bundle.class_names):
        raise ValueError(
            f"Nombre de sorties modele ({len(probabilities)}) different du nombre de classes ({len(bundle.class_names)})."
        )

    top_indices = np.argsort(probabilities)[::-1]
    class_index = int(top_indices[0])
    class_name = bundle.class_names[class_index]
    confidence = float(probabilities[class_index])
    saliency = _saliency_images(bundle.model, batch, class_index, image)

    return {
        "filename": filename,
        "model": bundle.model_path.name,
        "class_name": class_name,
        "label": _display_label(class_name),
        "confidence": confidence,
        "is_healthy": "healthy" in class_name.lower(),
        "recommendation": _recommendation(class_name, confidence),
        "preview": _png_data_url(image),
        "saliency": saliency,
        "top_predictions": [
            {
                "class_name": bundle.class_names[int(index)],
                "label": _display_label(bundle.class_names[int(index)]),
                "confidence": float(probabilities[int(index)]),
            }
            for index in top_indices[: min(5, len(top_indices))]
        ],
    }
