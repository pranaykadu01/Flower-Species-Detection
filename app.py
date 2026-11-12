
from flask import Flask, render_template, request
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "flower_species_model.keras"

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load trained flower model
model = tf.keras.models.load_model(MODEL_PATH)

# Class names in alphabetical training order
class_names = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

# Validate model output
if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        f"Class mismatch: {len(class_names)} classes found, "
        f"but model has {model.output_shape[-1]} outputs."
    )

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

        if "file" not in request.files:
            error = "Please select a flower image."

        else:
            file = request.files["file"]

            if file.filename == "":
                error = "Please select a flower image."

            else:
                try:
                    # Read uploaded image
                    image = Image.open(
                        io.BytesIO(file.read())
                    ).convert("RGB")

                    # Resize image
                    image = image.resize((224, 224))

                    # Convert to NumPy array
                    image_array = np.array(
                        image, dtype=np.float32
                    )

                    # Add batch dimension
                    image_array = np.expand_dims(
                        image_array, axis=0
                    )

                    # Model prediction
                    predictions = model.predict(
                        image_array, verbose=0
                    )

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

                except Exception as e:
                    print("Prediction error:", e)
                    error = (
                        "Unable to process this image. "
                        "Please upload a valid image."
                    )

    return render_template(
        "index.html",
        prediction=prediction,
        botanical_name=botanical_name,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
