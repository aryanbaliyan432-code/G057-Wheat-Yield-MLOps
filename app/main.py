"""FastAPI Production Inference Service for Wheat Yield Prediction.

Exposes REST endpoints for model health, service discovery, and yield predictions
with Pydantic validation, structured JSON logging, and error handling.
"""

from contextlib import asynccontextmanager
import os
import time
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pydantic import BaseModel, Field

from src.wheat_yield.logger import get_logger

logger = get_logger("wheat_yield.api")

# Global reference to loaded champion model
model_artifact = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")


def load_model(path: str = MODEL_PATH):
    """Load trained scikit-learn model pipeline from disk."""
    global model_artifact
    if not os.path.exists(path):
        fallback = "models/best_model.pkl"
        if os.path.exists(fallback):
            path = fallback
        else:
            logger.warning(
                f"Model file not found at {path} or {fallback}",
                extra={"event_type": "model_loading_warning"}
            )
            return None

    try:
        start_time = time.time()
        model_artifact = joblib.load(path)
        load_time_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"Model successfully loaded from {path} in {load_time_ms} ms",
            extra={"event_type": "model_loading", "details": {"model_path": path, "load_time_ms": load_time_ms}}
        )
        return model_artifact
    except Exception as e:
        logger.error(
            f"Failed to load model from {path}: {str(e)}",
            exc_info=True,
            extra={"event_type": "model_loading_error"}
        )
        raise


# Preload model at import time if present
try:
    load_model()
except Exception:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Initializing Wheat Yield Prediction API Service...", extra={"event_type": "startup"})
    load_model()
    yield
    logger.info("Shutting down Wheat Yield Prediction API Service...", extra={"event_type": "shutdown"})


app = FastAPI(
    title="Wheat Yield Prediction API",
    description="Jaipur Kisan Producer Company - Wheat Yield per Acre MLOps Service",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Input payload for wheat yield prediction with client business validation rules."""

    rainfall_mm: float = Field(..., description="Rainfall in mm during the season")
    fertilizer_kg: float = Field(
        ...,
        le=300.0,
        description="Fertilizer applied in kg/acre (must not exceed 300)"
    )
    sowing_delay_days: float = Field(..., description="Days sown after the ideal sowing date")
    irrigation_count: float = Field(..., description="Number of irrigations performed")
    soil_ph: float = Field(..., description="Soil pH value")
    farm_acres: float = Field(
        ...,
        ge=0.0,
        description="Total farm size in acres (must be non-negative)"
    )
    seed_variety: str = Field(..., description="Seed variety (e.g. HD-2967, PBW-343, DBW-187)")
    tehsil: str = Field(..., description="Sub-district / Tehsil (e.g. Tehsil North, Tehsil South, Tehsil West)")

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "rainfall_mm": 650.0,
                "fertilizer_kg": 110.0,
                "sowing_delay_days": 10.0,
                "irrigation_count": 4.0,
                "soil_ph": 6.8,
                "farm_acres": 5.5,
                "seed_variety": "HD-2967",
                "tehsil": "Tehsil South"
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output payload for wheat yield prediction."""

    predicted_yield_qtl_per_acre: float = Field(
        ...,
        description="Predicted wheat harvested in quintals per acre"
    )
    unit: str = "quintals_per_acre"
    model_version: str = "champion"
    status: str = "success"


# -------------------------------------------------------------
# Exception Handlers
# -------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle 422 schema validation errors with structured JSON logging."""
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {exc.errors()}",
        extra={
            "event_type": "validation_error",
            "details": {"errors": exc.errors()}
        }
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Input validation failed. Please check field constraints."}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unhandled 500 errors gracefully with logging."""
    logger.error(
        f"Unhandled error on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
        extra={"event_type": "internal_server_error"}
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred while processing the request."}
    )


# -------------------------------------------------------------
# Middleware for Request/Response Logging
# -------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log every HTTP request and response latency."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)

    logger.info(
        f"{request.method} {request.url.path} responded {response.status_code} in {duration_ms} ms",
        extra={
            "event_type": "http_request",
            "latency_ms": duration_ms,
            "status_code": response.status_code,
            "details": {
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown"
            }
        }
    )
    return response


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/", tags=["Discovery"])
def root() -> Dict[str, Any]:
    """Root endpoint providing service information and API discovery."""
    return {
        "service": "Wheat Yield Prediction API",
        "description": "Production MLOps Service for Jaipur Kisan Producer Company",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/health", tags=["Health"])
def health() -> Dict[str, Any]:
    """Health check endpoint required by Kubernetes liveness and readiness probes."""
    is_ready = model_artifact is not None
    return {
        "status": "ok",
        "model_loaded": is_ready,
        "service": "wheat-yield-api"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict wheat yield per acre based on season conditions and agronomic inputs."""
    global model_artifact
    if model_artifact is None:
        # Attempt to load model on-demand
        model_artifact = load_model()
        if model_artifact is None:
            logger.error("Model unavailable for inference", extra={"event_type": "model_unavailable"})
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifact is not loaded or unavailable."
            )

    start_time = time.time()
    try:
        # Prepare feature DataFrame matching training columns
        input_dict = payload.model_dump()
        df_input = pd.DataFrame([input_dict])

        # Execute prediction
        raw_prediction = model_artifact.predict(df_input)
        predicted_yield = float(raw_prediction[0])

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        # Log prediction event (never logging any PII)
        logger.info(
            f"Prediction computed: {predicted_yield:.2f} qtl/acre in {inference_time_ms} ms",
            extra={
                "event_type": "prediction_request",
                "latency_ms": inference_time_ms,
                "details": {
                    "tehsil": payload.tehsil,
                    "seed_variety": payload.seed_variety,
                    "farm_acres": payload.farm_acres,
                    "predicted_yield_qtl_per_acre": round(predicted_yield, 2)
                }
            }
        )

        return PredictionResponse(
            predicted_yield_qtl_per_acre=round(predicted_yield, 2),
            unit="quintals_per_acre",
            model_version="champion",
            status="success"
        )

    except Exception as e:
        logger.error(
            f"Prediction calculation failed: {str(e)}",
            exc_info=True,
            extra={"event_type": "prediction_error"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
