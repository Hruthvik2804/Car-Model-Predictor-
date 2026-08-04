from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class ZeroShotCarRecognizer:
    """Recognize cars by comparing an image with catalog text descriptions."""

    def __init__(
        self,
        catalog_path: str | Path = "vehicle_catalog.json",
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.catalog = self._load_catalog(catalog_path)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.prompts = [
            f"a clear photograph of a {item['manufacturer']} {item['model']} car"
            for item in self.catalog
        ]

    @staticmethod
    def _load_catalog(path: str | Path) -> list[dict[str, str]]:
        catalog_path = Path(path)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Vehicle catalog not found: {catalog_path}")
        with catalog_path.open("r", encoding="utf-8") as file:
            catalog: Any = json.load(file)
        if not isinstance(catalog, list) or not catalog:
            raise ValueError("Vehicle catalog must contain a non-empty list.")
        return catalog

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict[str, object]]:
        top_k = max(1, min(top_k, len(self.catalog)))
        inputs = self.processor(
            text=self.prompts,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)[0]
        scores, indices = torch.topk(probabilities, k=top_k)

        results: list[dict[str, object]] = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            vehicle = self.catalog[index]
            results.append(
                {
                    "manufacturer": vehicle["manufacturer"],
                    "model": vehicle["model"],
                    "year_range": vehicle["year_range"],
                    "confidence": float(score),
                    "class_name": f"{vehicle['manufacturer']}__{vehicle['model']}__{vehicle['year_range']}",
                }
            )
        return results
