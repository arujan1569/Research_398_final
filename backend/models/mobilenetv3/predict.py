"""Reference Keras prediction. Usage: python predict.py /path/to/photo.jpg"""
import json
import os
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


def prepare_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            image = Image.alpha_composite(
                Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba)
        image = image.convert("RGB")
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (224, 224), (0, 0, 0))
        canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
        return np.asarray(canvas, dtype=np.float32)[None]


def predict(path, model_directory=None):
    folder = Path(model_directory or Path(__file__).parent)
    labels = json.loads((folder / "class_names.json").read_text())
    model = tf.keras.models.load_model(folder / "rice_disease.keras", compile=False)
    scores = model.predict(prepare_image(path), verbose=0)[0]
    order = np.argsort(scores)[::-1]
    return {
        "predicted_class": labels[int(order[0])],
        "model_score": float(scores[order[0]]),
        "top_three": [{"class": labels[int(i)], "model_score": float(scores[i])}
                      for i in order[:3]],
        "note": "Uncalibrated closed-set scores; not a confirmed diagnosis or unknown-image detector.",
    }


if __name__ == "__main__":
    print(json.dumps(predict(sys.argv[1]), indent=2))
