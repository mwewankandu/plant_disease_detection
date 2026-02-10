# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pickle
import os

# -----------------------------------------------------------------------------
# Flask app setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# Paths (ROBUST & SAFE)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "..", "ml_model", "plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "..", "ml_model", "class_names.pkl"
)

# -----------------------------------------------------------------------------
# API Class
# -----------------------------------------------------------------------------
class DiseaseDetectionAPI:
    def __init__(self):
        # Validate files
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(f"Class names not found at {CLASS_NAMES_PATH}")

        # Load model
        self.model = tf.keras.models.load_model(MODEL_PATH)

        # Load class names
        with open(CLASS_NAMES_PATH, "rb") as f:
            self.class_names = pickle.load(f)

        # Disease information database
        self.disease_info = {
            "Tomato_Early_Blight": {
                "name": "Tomato Early Blight",
                "scientific_name": "Alternaria solani",
                "description": "Fungal disease causing concentric rings on leaves",
                "treatment": {
                    "chemical": [
                        "Chlorothalonil",
                        "Mancozeb",
                        "Copper fungicides"
                    ],
                    "organic": [
                        "Neem oil",
                        "Baking soda spray",
                        "Proper spacing"
                    ],
                    "dosage": "Apply every 7–14 days as preventive"
                },
                "symptoms": [
                    "Brown spots with concentric rings",
                    "Yellowing leaves",
                    "Leaf drop"
                ]
            },
            "Potato_Late_Blight": {
                "name": "Potato Late Blight",
                "scientific_name": "Phytophthora infestans",
                "description": "Destructive disease that caused Irish Potato Famine",
                "treatment": {
                    "chemical": [
                        "Chlorothalonil",
                        "Metalaxyl",
                        "Famoxadone"
                    ],
                    "organic": [
                        "Copper sprays",
                        "Biological fungicides"
                    ],
                    "dosage": "Apply before rainy periods"
                },
                "symptoms": [
                    "Water-soaked lesions",
                    "White mold growth",
                    "Rapid plant death"
                ]
            },
            "Tomato_Healthy": {
                "name": "Healthy Tomato Plant",
                "description": "Plant shows no signs of disease",
                "maintenance": [
                    "Regular watering",
                    "Balanced fertilization",
                    "Proper pruning"
                ]
            }
        }

        print("API initialized successfully")
        print(f"Loaded model with {len(self.class_names)} classes")
        print("Classes:", self.class_names)

    # -------------------------------------------------------------------------
    # IMAGE PREPROCESSING (MATCHES TRAINING)
    # -------------------------------------------------------------------------
    def preprocess_image(self, image_bytes):
        """
        IMPORTANT:
        Training used MobileNetV2 preprocessing:
        (x / 127.5) - 1
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0

        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------
    def predict(self, image_bytes):
        img_array = self.preprocess_image(image_bytes)

        predictions = self.model.predict(img_array, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx] * 100)
        disease_name = self.class_names[predicted_idx]

        result = {
            "disease": disease_name,
            "confidence": confidence,
            "disease_info": self.disease_info.get(disease_name, {}),
            "all_predictions": []
        }

        for i, prob in enumerate(predictions):
            result["all_predictions"].append({
                "class": self.class_names[i],
                "confidence": float(prob * 100)
            })

        result["all_predictions"].sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return result


# -----------------------------------------------------------------------------
# Initialize API (ONCE)
# -----------------------------------------------------------------------------
api = DiseaseDetectionAPI()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "Plant Disease Detection API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Upload image",
            "/diseases": "GET - List diseases",
            "/info/<disease_name>": "GET - Disease info"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_bytes = request.files["image"].read()
        result = api.predict(image_bytes)

        return jsonify({
            "success": True,
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/diseases", methods=["GET"])
def get_diseases():
    diseases = []

    for key in api.class_names:
        info = api.disease_info.get(key, {"name": key})
        diseases.append({
            "id": key,
            "name": info.get("name", key),
            "description": info.get("description", "")
        })

    return jsonify({
        "success": True,
        "count": len(diseases),
        "diseases": diseases
    })


@app.route("/info/<disease_name>", methods=["GET"])
def get_disease_info(disease_name):
    info = api.disease_info.get(disease_name)

    if not info:
        return jsonify({"error": "Disease not found"}), 404

    return jsonify({
        "success": True,
        "disease_info": info
    })


# -----------------------------------------------------------------------------
# Run server
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Plant Disease Detection API")
    print("Server running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
