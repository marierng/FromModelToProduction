"""
Flask service for Fashion-MNIST inference.
Endpoints
  POST /predict   -> JSON probs for one image
  GET  /health    -> {"status":"up"}
"""

import io
import logging
import os
from typing import List

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow import keras

# ── Logging ───────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
file_base = os.path.splitext(os.path.basename(__file__))[0]          # "api"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{file_base}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Model ─────────────────────────────────────────────────────────────
MODEL_PATH = "fashion_model.h5"
CLASS_NAMES: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

logger.info("Loading model from %s", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH)
logger.info("Model loaded successfully")

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)

def preprocess(img: Image.Image) -> np.ndarray:
    """28×28 grayscale, scale 0-1, add batch & channel dims."""
    img = img.convert("L").resize((28, 28))
    x = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(x, axis=(0, -1))               # (1, 28, 28, 1)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        logger.warning("POST /predict without file part")
        return jsonify({"error": "no file"}), 400

    try:
        img = Image.open(io.BytesIO(request.files["file"].read()))
        probs = model.predict(preprocess(img))[0].tolist()
        logger.info("Prediction ok")
        return jsonify({"probabilities": probs}), 200
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return jsonify({"error": str(exc)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "up"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


#-------------------------


#from flask import Flask, request, jsonify
#from tensorflow.keras.models import load_model
#import numpy as np
#import base64
#from PIL import Image
#import io


#import logging, os
#os.makedirs("logs", exist_ok=True)

#logging.basicConfig(
#    level=logging.INFO,
#    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#    handlers=[
#        logging.FileHandler("logs/api.log" if __name__ == "api" else "logs/batch.log"),
#        logging.StreamHandler()
#    ]
#)
#logger = logging.getLogger(__name__)


#app = Flask(__name__)
#model = load_model('fashion_model.keras')

# Define categories if needed
#categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#def preprocess_image(image_bytes):
#    image = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
#    image = np.array(image).reshape(1, 28, 28, 1) / 255.0
#    return image

#@app.route('/predict', methods=['POST'])
#def predict():
#    data = request.get_json(force=True)
#    image_b64 = data['image']
#    image_bytes = base64.b64decode(image_b64)
#    img = preprocess_image(image_bytes)
#    predictions = model.predict(img)
#    probs = predictions[0].tolist()
#    return jsonify(dict(zip(categories, probs)))


#@app.route('/health', methods=['GET'])
#def health():
#    return jsonify({"status": "up"}), 200


#if __name__ == '__main__':
#    app.run(debug=True)


