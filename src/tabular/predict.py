"""
Tabular Crop Suitability Predictor.
Loads trained RandomForest/GradientBoosting model, StandardScaler, and LabelEncoder
to return top-k recommended crops with probabilities and agronomic details.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd

import config

CROP_TRANSLATIONS = {
    "rice": {"kn": "ಭತ್ತ (Rice)", "season": "Kharif / Rabi"},
    "maize": {"kn": "ಮೆಕ್ಕೆಜೋಳ (Maize)", "season": "Kharif / Rabi"},
    "chickpea": {"kn": "ಕಡಲೆ (Bengal Gram)", "season": "Rabi"},
    "kidneybeans": {"kn": "ರಾಜ್ಮಾ (Kidney Beans)", "season": "Kharif / Rabi"},
    "pigeonpeas": {"kn": "ತೊಗರಿ (Tur / Arhar)", "season": "Kharif"},
    "mothbeans": {"kn": "ಮಡಿಕೆ (Moth Bean)", "season": "Kharif"},
    "mungbean": {"kn": "ಹೆಸರುಕಾಳು (Green Gram)", "season": "Kharif / Summer"},
    "blackgram": {"kn": "ಉದ್ದಿನಕಾಳು (Black Gram)", "season": "Kharif / Rabi"},
    "lentil": {"kn": "ಮಸೂರ (Lentil)", "season": "Rabi"},
    "pomegranate": {"kn": "ದಾಳಿಂಬೆ (Pomegranate)", "season": "Perennial / Annual"},
    "banana": {"kn": "ಬಾಳೆ (Banana)", "season": "All Seasons"},
    "mango": {"kn": "ಮಾವಿನ ಹಣ್ಣು (Mango)", "season": "Summer Harvest"},
    "grapes": {"kn": "ದ್ರಾಕ್ಷಿ (Grapes)", "season": "Winter / Spring"},
    "watermelon": {"kn": "ಕಲ್ಲಂಗಡಿ (Watermelon)", "season": "Zaid / Summer"},
    "muskmelon": {"kn": "ಕರಬೂಜ (Muskmelon)", "season": "Zaid / Summer"},
    "apple": {"kn": "ಸೇಬು (Apple)", "season": "Temperate Hill Regions"},
    "orange": {"kn": "ಕಿತ್ತಳೆ (Orange)", "season": "Subtropical / Winter"},
    "papaya": {"kn": "ಪಪ್ಪಾಯಿ (Papaya)", "season": "Tropical / Perennial"},
    "coconut": {"kn": "ತೆಂಗು (Coconut)", "season": "Coastal / Tropical"},
    "cotton": {"kn": "ಹತ್ತಿ (Cotton)", "season": "Kharif"},
    "jute": {"kn": "ಸೆಣಬು (Jute)", "season": "Kharif"},
    "coffee": {"kn": "ಕಾಫಿ (Coffee)", "season": "Plantation / Hilly"},
}


class CropPredictor:
    """Inference engine for crop recommendation using trained tabular ML model."""

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        scaler_path: Path = config.SCALER_PATH,
        encoder_path: Path = config.ENCODER_PATH,
    ):
        if not (model_path.exists() and scaler_path.exists() and encoder_path.exists()):
            raise FileNotFoundError(
                f"Missing tabular model artifacts. Ensure training has run: {model_path}"
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        self.classes = list(self.encoder.classes_)

    def predict(
        self,
        n: float,
        p: float,
        k: float,
        temp: float,
        humidity: float,
        ph: float,
        rainfall: float,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Predict crop suitability for given soil and environmental conditions.

        Args:
            n: Nitrogen content (kg/ha or index)
            p: Phosphorus content
            k: Potassium content
            temp: Temperature in Celsius
            humidity: Relative humidity in %
            ph: Soil pH (e.g. 5.5 - 8.5)
            rainfall: Expected rainfall in mm
            top_k: Number of ranked crops to return

        Returns:
            Dict containing top crops, probabilities, and agronomic insights.
        """
        features_df = pd.DataFrame(
            [[n, p, k, temp, humidity, ph, rainfall]],
            columns=config.TABULAR_FEATURES,
        )
        features_scaled = self.scaler.transform(features_df)

        # Get prediction probabilities
        probs = self.model.predict_proba(features_scaled)[0]

        # Sort indices by descending probability
        top_indices = np.argsort(probs)[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            crop_name = self.classes[idx]
            prob = float(probs[idx])
            trans = CROP_TRANSLATIONS.get(
                crop_name.lower(),
                {"kn": crop_name, "season": "General"}
            )
            recommendations.append({
                "crop": crop_name,
                "kannada_name": trans["kn"],
                "probability": round(prob, 4),
                "confidence_percent": round(prob * 100, 2),
                "recommended_season": trans["season"],
            })

        return {
            "top_crop": recommendations[0]["crop"],
            "top_crop_kannada": recommendations[0]["kannada_name"],
            "top_confidence": recommendations[0]["confidence_percent"],
            "recommendations": recommendations,
            "input_features": {
                "N": n,
                "P": p,
                "K": k,
                "temperature_c": temp,
                "humidity_pct": humidity,
                "ph": ph,
                "rainfall_mm": rainfall,
            },
        }


if __name__ == "__main__":
    predictor = CropPredictor()
    # Test sample typical for Rice / paddy
    res = predictor.predict(n=90, p=42, k=43, temp=21.0, humidity=82.0, ph=6.5, rainfall=200.0)
    print("Test Prediction:", res)
