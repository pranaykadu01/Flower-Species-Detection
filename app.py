
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
        app.logger.info("Flower prediction request received")

        if "file" not in request.files:
            error = "Please select a flower image."

        else:
            file = request.files["file"]

            if file.filename == "":
                error = "Please select a flower image."

            else:
                try:
                    # Read and decode image
                    image_start = time.perf_counter()

                    image_bytes = file.read()

                    if not image_bytes:
                        raise ValueError("Empty image uploaded")

                    image = Image.open(
                        io.BytesIO(image_bytes)
                    ).convert("RGB")

                    # Resize to model input dimensions
                    image = image.resize((224, 224))

                    # Convert to NumPy array
                    image_array = np.asarray(
                        image, dtype=np.float32
                    )

                    # Add batch dimension
                    image_array = np.expand_dims(
                        image_array, axis=0
                    )

                    image_time = time.perf_counter() - image_start

                    # Direct CNN inference
                    prediction_start = time.perf_counter()

                    predictions = model(
                        image_array,
                        training=False
                    ).numpy()

                    prediction_time = (
                        time.perf_counter() - prediction_start
                    )

                    # Find predicted flower
                    predicted_index = int(
                        np.argmax(predictions[0])
                    )

                    predicted_class = class_names[predicted_index]

                    # Get flower information
                    details = flower_details[predicted_class]

                    prediction = details["common_name"]
                    botanical_name = details["botanical_name"]

                    confidence = round(
                        float(predictions[0][predicted_index]) * 100,
                        2
                    )

                    total_time = time.perf_counter() - request_start

                    # Performance logs
                    app.logger.info(
                        f"Image processing: {image_time:.3f}s"
                    )

                    app.logger.info(
                        f"CNN inference: {prediction_time:.3f}s"
                    )

                    app.logger.info(
                        f"Total prediction time: {total_time:.3f}s"
                    )

                    app.logger.info(
                        f"Predicted flower: {prediction}, "
                        f"Confidence: {confidence}%"
                    )

                except Exception as e:

                    app.logger.exception("Prediction failed")

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
