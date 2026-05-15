# ml_service/schemas.py

from pydantic import BaseModel, Field


class SegmentRequest(BaseModel):
    recency: float = Field(..., example=30)
    frequency: float = Field(..., example=5)
    monetary: float = Field(..., example=500.0)


class CLVRequest(BaseModel):
    recency: float = Field(..., example=30)
    frequency: float = Field(..., example=5)
    monetary: float = Field(..., example=500.0)
    segment: int = Field(..., example=2)


class NBARequest(BaseModel):
    recency: float = Field(..., example=30)
    frequency: float = Field(..., example=5)
    monetary: float = Field(..., example=500.0)
    last_product: int = Field(..., example=101)

