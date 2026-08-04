from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import tensorflow as tf

from utils import load_class_names, predict_top_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize a vehicle from an image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/car_model_recognition.keras")
    parser.add_argument("--classes", default="models/class_names.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names(args.classes)

    with Image.open(image_path) as image:
        results = predict_top_k(model, image, class_names, top_k=3)

    best = results[0]
    print("\nPrediction")
    print("-" * 40)
    print(f"Manufacturer: {best['manufacturer']}")
    print(f"Model:        {best['model']}")
    print(f"Year range:   {best['year_range']}")
    print(f"Confidence:   {best['confidence'] * 100:.2f}%")

    print("\nTop matches")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['manufacturer']} {result['model']} "
            f"({result['year_range']}) — {result['confidence'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
