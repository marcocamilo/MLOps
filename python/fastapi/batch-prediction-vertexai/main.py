import logging
import os
from typing import Any, Dict, List, Optional, Protocol, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modules.utils import (process_batch_predictions,
                           process_prediction_probabilities)

# ──────────────────────────────────────────────────────────────────────
#   ENVIRONMENT VARIABLES
# ──────────────────────────────────────────────────────────────────────
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./configs/service-account.json"


# ──────────────────────────────────────────────────────────────────────
#   FASTAPI CONFIG
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Batch Prediction API",
    description="API for performing batch predictions",
    version="0.1.0",
)
logger = logging.getLogger("uvicorn.error")


# ──────────────────────────────────────────────────────────────────────
#   REQUEST MODELS AND TYPE HINTS
# ──────────────────────────────────────────────────────────────────────
class RequestBody(BaseModel):
    """
    Standard request body for batch prediction
    """

    instances: List[Union[str, List[str]]]
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional parameters for prediction"
    )


class PredictionConfig(BaseModel):
    """
    Configuration for prediction process
    """

    top_n: int = Field(default=3, ge=1, le=10, description="Number of top predictions")
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to include a prediction",
    )


class PredictProbaModel(Protocol):
    """
    Dummy class for type hinting model class with predict_proba method
    """

    def predict_proba(self, X: Any) -> Any: ...


# ──────────────────────────────────────────────────────────────────────
#   DEPENDENCIES
# ──────────────────────────────────────────────────────────────────────
def load_model() -> Optional[PredictProbaModel]:
    """
    Load the prediction model

    Returns:
    --------
    Optional model with predict_proba method
    """
    try:
        import joblib

        model_path = "./models/your_model.pkl"
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


MODEL = load_model()


# ──────────────────────────────────────────────────────────────────────
#   FUNCTIONS
# ──────────────────────────────────────────────────────────────────────
def process_predictions(instances: List[str], model=None):
    if model is None:
        raise ValueError("❗ No prediction model provided")

    # Preprocessing logic
    # ...

    # Prediction
    logger.info("🔄 Getting topic predictions...")
    pred_probs = model.predict_proba(instances)
    top_n_predictions = process_prediction_probabilities(pred_probs.get())

    logger.info("✅ Topics predicted!")
    return top_n_predictions


# ──────────────────────────────────────────────────────────────────────
#   API ROUTES
# ──────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    model_status = "Loaded" if MODEL is not None else "Not Loaded"
    logger.info("✅ Health check passed!")
    return {"status": "Healthy", "model_status": model_status}


@app.post("/predict")
def predict(request: RequestBody):
    try:
        logger.info(
            f"👍 Received prediction request with {len(request.instances)} instances"
        )

        if request.instances:
            logger.info(
                f"ℹ️ First instance type: {type(request.instances[0])}, value: {request.instances[0]}"
            )

        predictions = process_batch_predictions(
            instances=request.instances,
            prediction_func=process_predictions,
            id_extractor=None,
            default_prediction=None,
        )

        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"❌ Error in /predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
