from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = (224, 224)


def parse_class_name(class_name: str) -> dict[str, str]:
    parts = class_name.split("__")
    if len(parts) != 3:
        return {
            "manufacturer": parts[0].replace("-", " ") if parts else class_name,
            "model": "Unknown",
            "year_range": "Unknown",
        }

    manufacturer, model, year_range = parts
    return {
        "manufacturer": manufacturer.replace("-", " "),
        "model": model.replace("-", " "),
        "year_range": year_range.replace("-", "–", 1),
    }


def load_class_names(path: str | Path) -> list[str]:
    class_path = Path(path)
    if not class_path.exists():
        raise FileNotFoundError(
            f"Class-name file not found: {class_path}. Train the model first."
        )
    with class_path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("class_names.json must contain a list of strings.")
    return data


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def predict_top_k(
    model: tf.keras.Model,
    image: Image.Image,
    class_names: list[str],
    top_k: int = 3,
) -> list[dict[str, object]]:
    probabilities = model.predict(prepare_image(image), verbose=0)[0]
    top_k = min(top_k, len(class_names))
    indices = np.argsort(probabilities)[::-1][:top_k]

    results: list[dict[str, object]] = []
    for index in indices:
        details = parse_class_name(class_names[int(index)])
        results.append(
            {
                **details,
                "class_name": class_names[int(index)],
                "confidence": float(probabilities[int(index)]),
            }
        )
    return results
