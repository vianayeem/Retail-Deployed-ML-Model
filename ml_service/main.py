# ml_service/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from model_loader import (
    kmeans_model,
    clv_model,
    nba_model,
    product_encoder,
    target_encoder,
    rfm_scaler,
    pca_model
)
from ml_service.app.schemas import SegmentRequest, CLVRequest, NBARequest


app = FastAPI(
    title="Retail Tech – ML Service",
    version="1.0.0",
    description="Machine Learning inference APIs for Retail Tech Dashboard"
)

# -----------------------
# CORS (Backend access)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict later to backend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Health Check
# -----------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Retail Tech ML Service"
    }


@app.get("/")
def root():
    return {
        "message": "Retail Tech ML API is running",
        "docs": "/docs"
    }


# -----------------------
# Customer Segmentation
# -----------------------
@app.post("/customer-segment")
def customer_segment(data: SegmentRequest):
    try:
        X = np.array([[data.recency, data.frequency, data.monetary]])
        X_scaled = rfm_scaler.transform(X)
        segment = int(kmeans_model.predict(X_scaled)[0])

        return {"segment": segment}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# CLV Prediction
# -----------------------
@app.post("/predict-clv")
def predict_clv(data: CLVRequest):
    try:
        X_rfm = np.array([[data.recency, data.frequency, data.monetary]])
        X_scaled = rfm_scaler.transform(X_rfm)
        X_pca = pca_model.transform(X_scaled)

        X_final = np.hstack([
            X_scaled,
            [[data.segment]],
            X_pca
        ])

        clv_value = float(clv_model.predict(X_final)[0])

        return {"predicted_clv": round(clv_value, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# Next Best Product
# -----------------------
@app.post("/next-product")
def next_best_product(data: NBARequest):
    try:
        X = np.array([[data.recency, data.frequency, data.monetary, data.last_product]])
        encoded_prediction = nba_model.predict(X)[0]

        product_code = target_encoder.inverse_transform([encoded_prediction])[0]
        product_name = product_encoder.inverse_transform([product_code])[0]

        return {"recommended_product": product_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
