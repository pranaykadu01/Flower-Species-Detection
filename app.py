
from flask import Flask, render_template, request
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "flower_species_model.keras"

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load model only once when application starts
print("Loading flower detection model...")
load_start = time.perf_counter()

model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model loaded in {time.perf_counter() - load_start:.2f} seconds")

# Flower classes - preserve training order
class_names = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

if len(class_names) != model.output_shape[-1]:
    raise ValueError("Model output and class count mismatch!")

# Flower information
flower_details = {
    "daisy": {
        "common_name": "Daisy",
        "botanical_name": "Bellis perennis"
    },
    "dandelion": {
        "common_name": "Dandelion",
        "botanical_name": "Taraxacum officinale"
    },
    "rose": {
        "common_name": "Rose",
        "botanical_name": "Rosa spp."
    },
    "sunflower": {
        "common_name": "Sunflower",
        "botanical_name": "Helianthus annuus"
    },
    "tulip": {
        "common_name": "Tulip",
        "botanical_name": "Tulipa spp."
    }
}

print("Flower model loaded successfully!")
print("Available classes:", class_names)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    botanical_name = None
    confidence = None
    error = None

    if request.method == "POST":

        request_start = time.perf_counter()
        app.logger.info("========== NEW PREDICTION REQUEST ==========")

        try:
            app.logger.info(
                "Request received. Content length: %s",
                request.content_length
            )

            # Read uploaded file
            app.logger.info("Checking uploaded file...")

            if "file" not in request.files:
                raise ValueError("No file field found")

            file = request.files["file"]

            if file.filename == "":
                raise ValueError("No file selected")

            # Image processing
            image_start = time.perf_counter()

            app.logger.info("Reading image bytes...")

            image_bytes = file.read()

            if not image_bytes:
                raise ValueError("Empty image uploaded")

            app.logger.info("Image size: %s bytes", len(image_bytes))

            app.logger.info("Decoding image...")

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            app.logger.info("Resizing image...")

            image = image.resize((224, 224))

            image_array = np.asarray(
                image, dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array, axis=0
            )

            image_time = time.perf_counter() - image_start

            app.logger.info(
                "Image processing completed in %.3f seconds",
                image_time
            )

            # CNN prediction
            app.logger.info("STARTING CNN INFERENCE...")

            prediction_start = time.perf_counter()

            predictions = model(
                image_array,
                training=False
            ).numpy()

            prediction_time = time.perf_counter() - prediction_start

            app.logger.info(
                "CNN inference completed in %.3f seconds",
                prediction_time
            )

            # Classification
            predicted_index = int(
                np.argmax(predictions[0])
            )

            predicted_class = class_names[predicted_index]

            details = flower_details[predicted_class]

            prediction = details["common_name"]
            botanical_name = details["botanical_name"]

            confidence = round(
                float(predictions[0][predicted_index]) * 100,
                2
            )

            total_time = time.perf_counter() - request_start

            app.logger.info("Predicted flower: %s", prediction)
            app.logger.info("Confidence: %.2f%%", confidence)
            app.logger.info(
                "TOTAL REQUEST TIME: %.3f seconds",
                total_time
            )

            app.logger.info("========== REQUEST COMPLETED ==========")

        except Exception:
            app.logger.exception("PREDICTION FAILED")

            error = (
                "Unable to process this image. "
                "Please upload a valid flower image."
            )


    return render_template(
        "index.html",
        prediction=prediction,
        botanical_name=botanical_name,
        confidence=confidence,
        error=error
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        prediction=None,
        botanical_name=None,
        confidence=None,
        error="Image is too large. Maximum size is 10 MB."
    ), 413


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5001)),
        debug=False
    )
