
from flask import Flask, render_template, request
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import time
import logging

# ==========================================
# FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "flower_species_model.keras"

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

app.logger.setLevel(logging.INFO)


# ==========================================
# LOAD TRAINED CNN MODEL
# ==========================================

print("==========================================")
print("Loading Flower Detection Model...")
print("==========================================")

load_start = time.perf_counter()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

load_time = time.perf_counter() - load_start

print(f"Model loaded successfully in {load_time:.2f} seconds")

# ==========================================
# FLOWER CLASSES
# ==========================================

# IMPORTANT:
# Preserve the exact class order used during training.

class_names = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        f"Class mismatch: {len(class_names)} classes, "
        f"but model has {model.output_shape[-1]} outputs."
    )


# ==========================================
# FLOWER INFORMATION
# ==========================================

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

print("Available flower classes:", class_names)
print("Flower model initialization completed!")


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_bytes):

    app.logger.info("Starting image preprocessing")

    # Decode uploaded image
    image = Image.open(
        io.BytesIO(image_bytes)
    )

    # Check image format
    if image.format not in ("JPEG", "PNG"):
        raise ValueError("Unsupported image format")

    # Convert to RGB
    image = image.convert("RGB")

    app.logger.info(
        "Original image dimensions: %s",
        image.size
    )

    # Resize image to CNN input size
    image = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    app.logger.info(
        "Processed image shape: %s",
        image_array.shape
    )

    app.logger.info(
        "Image preprocessing completed"
    )

    return image_array


# ==========================================
# FLOWER PREDICTION FUNCTION
# ==========================================

def predict_flower(image_array):

    app.logger.info("STARTING CNN INFERENCE")

    prediction_start = time.perf_counter()

    # Run trained CNN model
    predictions = model(
        image_array,
        training=False
    )

    # Convert TensorFlow output to NumPy
    predictions = np.asarray(
        predictions.numpy()
    )

    prediction_time = (
        time.perf_counter() - prediction_start
    )

    app.logger.info(
        "CNN inference completed in %.3f seconds",
        prediction_time
    )

    # Validate model output
    if predictions.ndim != 2:
        raise ValueError(
            f"Unexpected prediction dimensions: {predictions.shape}"
        )

    if predictions.shape[0] != 1:
        raise ValueError(
            "Unexpected prediction batch size"
        )

    if predictions.shape[1] != len(class_names):
        raise ValueError(
            "Model prediction count does not match flower classes"
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            "Model returned invalid prediction values"
        )

    app.logger.info(
        "Raw prediction values: %s",
        predictions[0].tolist()
    )

    # Find highest prediction
    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = class_names[predicted_index]

    # Get flower details
    details = flower_details[predicted_class]

    common_name = details["common_name"]

    botanical_name = details["botanical_name"]

    confidence = round(
        float(predictions[0][predicted_index]) * 100,
        2
    )

    app.logger.info(
        "PREDICTED FLOWER: %s",
        common_name
    )

    app.logger.info(
        "BOTANICAL NAME: %s",
        botanical_name
    )

    app.logger.info(
        "CONFIDENCE: %.2f%%",
        confidence
    )

    return common_name, botanical_name, confidence


# ==========================================
# HOME AND PREDICTION ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    botanical_name = None
    confidence = None
    error = None

    # ======================================
    # GET REQUEST
    # ======================================

    if request.method == "GET":

        app.logger.info(
            "Homepage accessed successfully"
        )

        return render_template(
            "index.html",
            prediction=None,
            botanical_name=None,
            confidence=None,
            error=None
        )

    # ======================================
    # POST REQUEST
    # ======================================

    request_start = time.perf_counter()

    app.logger.info(
        "=========================================="
    )

    app.logger.info(
        "NEW FLOWER PREDICTION REQUEST RECEIVED"
    )

    app.logger.info(
        "Request content length: %s bytes",
        request.content_length
    )

    try:

        # ----------------------------------
        # STEP 1: CHECK UPLOADED FILE
        # ----------------------------------

        app.logger.info(
            "STEP 1: Checking uploaded file"
        )

        if "file" not in request.files:

            raise ValueError(
                "No file field found in request"
            )

        file = request.files["file"]

        if not file or file.filename == "":

            raise ValueError(
                "No image selected"
            )

        app.logger.info(
            "Uploaded filename: %s",
            file.filename
        )

        # ----------------------------------
        # STEP 2: READ IMAGE
        # ----------------------------------

        app.logger.info(
            "STEP 2: Reading image bytes"
        )

        image_bytes = file.read()

        if not image_bytes:

            raise ValueError(
                "Uploaded image is empty"
            )

        app.logger.info(
            "Image received: %s bytes",
            len(image_bytes)
        )

        # ----------------------------------
        # STEP 3: PREPROCESS IMAGE
        # ----------------------------------

        app.logger.info(
            "STEP 3: Image preprocessing started"
        )

        image_start = time.perf_counter()

        image_array = preprocess_image(
            image_bytes
        )

        image_time = (
            time.perf_counter() - image_start
        )

        app.logger.info(
            "Image processing time: %.3f seconds",
            image_time
        )

        # ----------------------------------
        # STEP 4: CNN PREDICTION
        # ----------------------------------

        app.logger.info(
            "STEP 4: Sending image to CNN model"
        )

        prediction, botanical_name, confidence = predict_flower(
            image_array
        )

        # ----------------------------------
        # STEP 5: COMPLETED
        # ----------------------------------

        total_time = (
            time.perf_counter() - request_start
        )

        app.logger.info(
            "=========================================="
        )

        app.logger.info(
            "PREDICTION COMPLETED SUCCESSFULLY"
        )

        app.logger.info(
            "Flower: %s",
            prediction
        )

        app.logger.info(
            "Confidence: %.2f%%",
            confidence
        )

        app.logger.info(
            "Total request time: %.3f seconds",
            total_time
        )

        app.logger.info(
            "=========================================="
        )

    # ======================================
    # ERROR HANDLING
    # ======================================

    except UnidentifiedImageError:

        app.logger.exception(
            "INVALID IMAGE FILE"
        )

        error = (
            "Invalid image file. "
            "Please upload a valid JPG or PNG image."
        )

    except Exception as e:

        app.logger.exception(
            "FLOWER PREDICTION FAILED: %s",
            str(e)
        )

        error = (
            "Unable to identify the flower. "
            "Please try another clear flower image."
        )

    # ======================================
    # RETURN RESULT
    # ======================================

    app.logger.info(
        "Returning response to browser"
    )

    return render_template(
        "index.html",
        prediction=prediction,
        botanical_name=botanical_name,
        confidence=confidence,
        error=error
    )


# ==========================================
# FILE SIZE ERROR HANDLER
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    app.logger.warning(
        "Uploaded image exceeded maximum file size"
    )

    return render_template(
        "index.html",
        prediction=None,
        botanical_name=None,
        confidence=None,
        error="Image is too large. Maximum size is 10 MB."
    ), 413


# ==========================================
# APPLICATION START
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5001)
    )

    app.logger.info(
        "Starting Flower Species Detection application"
    )

    app.logger.info(
        "Application port: %s",
        port
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
