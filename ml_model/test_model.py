import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os
import warnings

# Optional: silence TF Lite deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DiseasePredictor:
    def __init__(
        self,
        model_path="plant_disease_model.keras",
        tflite_path="model.tflite",
        class_names_path="class_names.pkl",
        img_size=224
    ):
        self.img_size = img_size

        # Load full Keras model
        self.model = tf.keras.models.load_model(model_path)

        # Load class names
        with open(class_names_path, "rb") as f:
            self.class_names = pickle.load(f)

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"Model loaded with {len(self.class_names)} classes: {self.class_names}")

    def preprocess_image(self, image_path):
        """
        Preprocess image EXACTLY like training:
        (x / 127.5) - 1
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size))

        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0  # IMPORTANT FIX

        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path, use_tflite=False):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_array = self.preprocess_image(image_path)

        if use_tflite:
            self.interpreter.set_tensor(
                self.input_details[0]["index"], img_array
            )
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(
                self.output_details[0]["index"]
            )[0]
        else:
            predictions = self.model.predict(img_array, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx] * 100)

        results = [
            {
                "class": self.class_names[i],
                "confidence": float(pred * 100)
            }
            for i, pred in enumerate(predictions)
        ]

        results.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "predicted_class": self.class_names[predicted_idx],
            "confidence": confidence,
            "all_predictions": results
        }

    def get_treatment_info(self, disease_name):
        treatments = {
            "Tomato_Early_Blight": {
                "description": "Early blight is a fungal disease affecting tomato plants.",
                "chemical_treatment": [
                    "Chlorothalonil (every 7–10 days)",
                    "Copper-based fungicides",
                    "Mancozeb at first sign"
                ],
                "organic_treatment": [
                    "Neem oil spray",
                    "Baking soda solution",
                    "Proper pruning for airflow"
                ],
                "prevention": "Crop rotation, remove infected leaves, avoid overhead watering"
            },
            "Potato_Late_Blight": {
                "description": "Late blight caused by Phytophthora infestans.",
                "chemical_treatment": [
                    "Chlorothalonil",
                    "Metalaxyl-based fungicides",
                    "Famoxadone + Cymoxanil"
                ],
                "organic_treatment": [
                    "Copper sulfate sprays",
                    "Bacillus subtilis",
                    "Garlic & chili extract sprays"
                ],
                "prevention": "Disease-free seed potatoes, good drainage"
            },
            "Tomato_Healthy": {
                "description": "Plant is healthy.",
                "maintenance": [
                    "Proper watering",
                    "Balanced fertilizer",
                    "Regular inspection",
                    "Pruning and support"
                ],
                "prevention": "Mulching, spacing, crop rotation"
            }
        }

        return treatments.get(
            disease_name,
            {
                "description": "No information available.",
                "chemical_treatment": ["Consult agricultural officer"],
                "organic_treatment": ["Practice good sanitation"]
            }
        )


def test_with_sample():
    predictor = DiseasePredictor()

    # 🔴 CHANGE THIS LINE ONLY 🔴
    test_image = r"C:\Users\User\Music\plant_disease_detection\ml_model\test_sample.jpg"

    try:
        result = predictor.predict(test_image, use_tflite=False)

        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        print(f"Predicted Disease: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")

        print("\nAll Predictions:")
        for pred in result["all_predictions"]:
            print(f"  - {pred['class']}: {pred['confidence']:.2f}%")

        treatment = predictor.get_treatment_info(result["predicted_class"])

        print("\nTREATMENT INFORMATION")
        print("-" * 50)
        print("Description:", treatment.get("description", "N/A"))

        if "chemical_treatment" in treatment:
            print("\nChemical Treatments:")
            for item in treatment["chemical_treatment"]:
                print(f"  • {item}")

        if "organic_treatment" in treatment:
            print("\nOrganic Treatments:")
            for item in treatment["organic_treatment"]:
                print(f"  • {item}")

        print("\nPrevention:", treatment.get("prevention", "N/A"))

    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    test_with_sample()
