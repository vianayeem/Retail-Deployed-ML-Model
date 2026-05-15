# ml_service/model_loader.py

import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")


def load_model(filename: str):
    path = os.path.join(MODEL_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {filename}")

    try:
        model = joblib.load(path)
        print(f"✅ Loaded model: {filename}")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model {filename}: {e}")


# Core models
kmeans_model = load_model("kmeans_model.pkl")
clv_model = load_model("clv_model.pkl")
nba_model = load_model("nba_model.pkl")

# Supporting transformers
rfm_scaler = load_model("rfm_scaler.pkl")
pca_model = load_model("pca_model.pkl")

# Encoders
product_encoder = load_model("product_encoder.pkl")
target_encoder = load_model("target_encoder.pkl")
