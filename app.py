import io
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow import keras
from PIL import Image

app = Flask(__name__)

# ============================================================
# LOAD MODEL & CONFIG
# ============================================================

MODEL_PATH     = "fake_equipment_detector.keras"
THRESHOLD_PATH = "threshold.json"

print("⏳ Loading model...")
model = keras.models.load_model(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    config = json.load(f)

THRESHOLD = float(config["threshold"])
IMG_SIZE  = int(config["img_size"])

print(f"✅ Autoencoder model loaded")
print(f"   Threshold  : {THRESHOLD}")
print(f"   Image size : {IMG_SIZE}x{IMG_SIZE}")
print(f"🚀 Visit : http://localhost:5000")

# ============================================================
# PREDICTION FUNCTION — Autoencoder (reconstruction error)
# ============================================================

def predict_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, IMG_SIZE, IMG_SIZE, 3)

    output = model.predict(arr, verbose=0)  # (1, IMG_SIZE, IMG_SIZE, 3)

    print(f"   Model output shape : {output.shape}")
    print(f"   Model output       : {output}")

    recon_error = float(np.mean((arr - output) ** 2))
    label       = "FAKE" if recon_error > THRESHOLD else "REAL"

    ratio = recon_error / THRESHOLD if THRESHOLD > 0 else 1.0
    confidence = round(min(ratio, 2.0) / 2.0 * 100, 1) if label == "FAKE" \
                 else round(max(0.0, 1.0 - ratio) * 100, 1)

    return label, round(recon_error, 6), confidence

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"err": "No image uploaded"}), 400

    file      = request.files["image"]
    img_bytes = file.read()

    try:
        label, recon_error, confidence = predict_image(img_bytes)
        return jsonify({
            "result"     : label,
            "recon_error": recon_error,
            "threshold"  : THRESHOLD,
            "confidence" : confidence,
            "filename"   : file.filename
        })
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"err": str(e)}), 500

@app.route("/calibrate", methods=["POST"])
def calibrate():
    """POST a known-REAL image to measure its MSE — helps set the right threshold."""
    if "image" not in request.files:
        return jsonify({"err": "No image uploaded"}), 400

    file      = request.files["image"]
    img_bytes = file.read()

    try:
        img = Image.open(io.BytesIO(img_bytes)).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        output      = model.predict(arr, verbose=0)
        recon_error = float(np.mean((arr - output) ** 2))

        return jsonify({
            "filename"         : file.filename,
            "mse"              : round(recon_error, 6),
            "current_threshold": THRESHOLD,
            "suggestion"       : "Set threshold slightly above this MSE for REAL images"
        })
    except Exception as e:
        return jsonify({"err": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)