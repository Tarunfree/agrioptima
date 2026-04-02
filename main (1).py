"""
AgriOptima — Resource-Constrained Crop Recommendation API
FastAPI application exposing Swagger docs at /docs
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ──────────────────────────────────────────────
# Bootstrap — load artefacts once at startup
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

try:
    full_model    = joblib.load(MODELS_DIR / "full_model.pkl")
    single_model  = joblib.load(MODELS_DIR / "single_model.pkl")
    scaler        = joblib.load(MODELS_DIR / "scaler.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    with open(MODELS_DIR / "metadata.json") as f:
        META = json.load(f)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model artefacts not found. Run `python scripts/train.py` first.\n{e}"
    )

FEATURES         = META["features"]          # ["N", "P", "K", "pH"]
BEST_FEATURE     = META["best_feature"]      # e.g. "K"
BEST_FEATURE_IDX = META["best_feature_idx"]
CROPS            = META["crops"]

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="AgriOptima — Crop Recommendation API",
    description="""
## 🌾 AgriOptima: Resource-Constrained Crop Recommendation

AgriOptima helps small-scale farmers choose the optimal crop using **minimal soil testing**.

### Key idea
Comprehensive soil tests (N, P, K, pH) cost farmers significantly. Using **univariate feature
selection**, we identify the *single most predictive soil attribute* and train a lightweight
classifier on it — reducing testing costs by **75%** while maintaining acceptable accuracy.

### Endpoints

| Endpoint | Description |
|---|---|
| `POST /predict/single` | Recommend a crop from **one** soil value (cheapest option) |
| `POST /predict/full` | Recommend a crop from **all 4** soil values (highest accuracy) |
| `GET /feature-importance` | Ranked list of soil attributes by predictive power |
| `GET /model-comparison` | Accuracy trade-off: single-feature vs full-feature model |
| `GET /crops` | List all supported crop classes |
| `GET /health` | Service health check |

### Tech Stack
- **Backend**: FastAPI + Python 3.x
- **ML**: Scikit-learn (Logistic Regression, Gaussian Naïve Bayes)
- **Feature Selection**: Univariate ANOVA F-test
- **Data**: N, P, K, pH soil attributes → 22 crop classes
    """,
    version="1.0.0",
    contact={"name": "AgriOptima Team", "email": "team@agrioptima.dev"},
    license_info={"name": "MIT"},
    tags_metadata=[
        {"name": "Prediction",        "description": "Crop recommendation endpoints"},
        {"name": "Analysis",          "description": "Feature importance and model comparison"},
        {"name": "Utility",           "description": "Health check and metadata"},
    ],
)

# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class SingleFeatureInput(BaseModel):
    """Input for the resource-constrained single-feature prediction."""

    value: float = Field(
        ...,
        description=f"Value for the most predictive soil attribute: **{BEST_FEATURE}** (Potassium, mg/kg)",
        example=30.0,
        ge=0,
        le=300,
    )

    class Config:
        schema_extra = {
            "example": {"value": 30.0}
        }


class FullFeatureInput(BaseModel):
    """Input for the full 4-attribute crop prediction."""

    N:  float = Field(..., description="Nitrogen content (mg/kg)", example=90.0,  ge=0, le=300)
    P:  float = Field(..., description="Phosphorous content (mg/kg)", example=42.0, ge=0, le=300)
    K:  float = Field(..., description="Potassium content (mg/kg)", example=43.0,  ge=0, le=300)
    pH: float = Field(..., description="Soil pH level", example=6.5, ge=0.0, le=14.0)

    class Config:
        schema_extra = {
            "example": {"N": 90.0, "P": 42.0, "K": 43.0, "pH": 6.5}
        }


class PredictionResponse(BaseModel):
    recommended_crop: str  = Field(..., description="Predicted crop name")
    confidence:       float = Field(..., description="Model confidence (0–1)", ge=0, le=1)
    model_used:       str  = Field(..., description="Classifier name")
    mode:             str  = Field(..., description="'single_feature' or 'full_feature'")
    key_nutrient:     Optional[str] = Field(None, description="Soil attribute used for prediction")
    top_3_crops:      list  = Field(..., description="Top 3 crop candidates with probabilities")


class FeatureImportanceItem(BaseModel):
    feature:     str   = Field(..., description="Soil attribute name")
    rank:        int   = Field(..., description="Rank (1 = most predictive)")
    f_score:     float = Field(..., description="ANOVA F-score")
    p_value:     float = Field(..., description="Statistical p-value")
    description: str   = Field(..., description="Human-readable interpretation")


class ModelComparisonResponse(BaseModel):
    full_feature_model:   dict = Field(..., description="Metrics for the 4-attribute model")
    single_feature_model: dict = Field(..., description="Metrics for the 1-attribute model")
    best_feature:         str  = Field(..., description="The single feature selected")
    accuracy_drop_pct:    float = Field(..., description="Accuracy reduction using 1 feature (%)")
    cost_reduction_pct:   float = Field(..., description="Estimated soil-testing cost saved (%)")
    verdict:              str  = Field(..., description="Cost–accuracy trade-off summary")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "N":  "Nitrogen — drives leaf growth and chlorophyll production",
    "P":  "Phosphorous — supports root development and energy transfer",
    "K":  "Potassium — regulates water uptake and disease resistance",
    "pH": "Soil pH — determines nutrient availability and microbial activity",
}


def _scale_and_predict(model, X_raw: np.ndarray, full: bool) -> dict:
    """Scale input, run model, return structured result."""
    # Always scale full 4-feature vector, then slice if needed
    if full:
        X_sc = scaler.transform(X_raw)
    else:
        # reconstruct neutral 4-feature vector with only best feature populated
        neutral = np.zeros((1, 4))
        neutral[0, BEST_FEATURE_IDX] = X_raw[0, 0]
        # scale individually by replicating the scaler's params for that one feature
        mean_  = scaler.mean_[BEST_FEATURE_IDX]
        scale_ = scaler.scale_[BEST_FEATURE_IDX]
        scaled_val = (X_raw[0, 0] - mean_) / scale_
        X_sc = scaled_val.reshape(1, 1)

    probas = model.predict_proba(X_sc)[0]
    top3_idx = np.argsort(probas)[::-1][:3]

    return {
        "recommended_crop": label_encoder.inverse_transform([np.argmax(probas)])[0],
        "confidence":       round(float(probas.max()), 4),
        "top_3_crops": [
            {
                "crop":        label_encoder.inverse_transform([i])[0],
                "probability": round(float(probas[i]), 4),
            }
            for i in top3_idx
        ],
    }


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health", tags=["Utility"], summary="Service health check")
def health():
    """Returns service status and loaded model information."""
    return {
        "status":       "healthy",
        "best_feature": BEST_FEATURE,
        "crops_count":  len(CROPS),
        "full_model":   META["full_model_name"],
        "single_model": META["single_model_name"],
    }


@app.get("/crops", tags=["Utility"], summary="List all supported crop classes")
def list_crops():
    """Returns all 22 crop types the model can recommend."""
    return {"crops": sorted(CROPS), "count": len(CROPS)}


@app.post(
    "/predict/single",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Recommend crop using ONE soil attribute (resource-constrained mode)",
)
def predict_single(body: SingleFeatureInput):
    """
    **Lightweight prediction** — uses only the single most informative soil attribute
    identified through univariate feature selection (ANOVA F-test).

    This is the **core contribution** of AgriOptima: provide a useful crop recommendation
    by testing *just one* soil parameter, cutting farmer testing costs by ~75%.

    - Requires: one soil value
    - Model: best single-feature classifier (Logistic Regression or Gaussian NB)
    - Accuracy: lower than full model, but practically useful
    """
    X_raw = np.array([[body.value]])
    result = _scale_and_predict(single_model, X_raw, full=False)

    return PredictionResponse(
        recommended_crop=result["recommended_crop"],
        confidence=result["confidence"],
        model_used=META["single_model_name"],
        mode="single_feature",
        key_nutrient=BEST_FEATURE,
        top_3_crops=result["top_3_crops"],
    )


@app.post(
    "/predict/full",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Recommend crop using ALL 4 soil attributes (full-accuracy mode)",
)
def predict_full(body: FullFeatureInput):
    """
    **High-accuracy prediction** — uses all four soil attributes (N, P, K, pH).

    Use this when the farmer has access to a full soil test. Compare its output
    against `/predict/single` to see the accuracy gain from additional testing.

    - Requires: N, P, K, pH values
    - Model: best full-feature classifier
    - Accuracy: highest possible with this feature set
    """
    X_raw = np.array([[body.N, body.P, body.K, body.pH]])
    X_sc  = scaler.transform(X_raw)
    probas = full_model.predict_proba(X_sc)[0]
    top3_idx = np.argsort(probas)[::-1][:3]

    return PredictionResponse(
        recommended_crop=label_encoder.inverse_transform([np.argmax(probas)])[0],
        confidence=round(float(probas.max()), 4),
        model_used=META["full_model_name"],
        mode="full_feature",
        key_nutrient=None,
        top_3_crops=[
            {
                "crop":        label_encoder.inverse_transform([i])[0],
                "probability": round(float(probas[i]), 4),
            }
            for i in top3_idx
        ],
    )


@app.get(
    "/feature-importance",
    response_model=list[FeatureImportanceItem],
    tags=["Analysis"],
    summary="Ranked soil attributes by predictive power (univariate ANOVA F-test)",
)
def feature_importance():
    """
    Returns all four soil attributes ranked by their **ANOVA F-score** — the metric
    used to select the single best feature for the resource-constrained model.

    Higher F-score → stronger relationship between that nutrient and crop type.

    This underpins the project's central claim: that one attribute (typically **K**)
    captures enough discriminative signal to make a useful crop recommendation.
    """
    fi = META["feature_importance"]
    items = []
    for feat in sorted(fi, key=lambda f: fi[f]["rank"]):
        items.append(
            FeatureImportanceItem(
                feature=feat,
                rank=fi[feat]["rank"],
                f_score=round(fi[feat]["f_score"], 2),
                p_value=fi[feat]["p_value"],
                description=FEATURE_DESCRIPTIONS.get(feat, ""),
            )
        )
    return items


@app.get(
    "/model-comparison",
    response_model=ModelComparisonResponse,
    tags=["Analysis"],
    summary="Cost vs accuracy trade-off: single-feature vs full-feature model",
)
def model_comparison():
    """
    Compares the **full 4-attribute model** against the **single-attribute model**
    across accuracy, precision, recall, and F1-score.

    This is the **quantitative validation** of AgriOptima's thesis:
    > "A 75% reduction in testing cost at an acceptable accuracy trade-off."

    Metrics used: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score.
    """
    m = META["metrics"]
    drop = META["accuracy_drop_pct"]

    if drop <= 10:
        verdict = (
            f"✅ Excellent trade-off: only {drop}% accuracy loss for 75% cost reduction. "
            f"Single-feature model is strongly recommended for resource-constrained settings."
        )
    elif drop <= 25:
        verdict = (
            f"⚠️  Acceptable trade-off: {drop}% accuracy loss for 75% cost reduction. "
            f"Single-feature model is viable but farmers near crop boundaries should prefer full testing."
        )
    else:
        verdict = (
            f"⚡ Significant trade-off: {drop}% accuracy loss. "
            f"Single-feature model provides a useful first estimate; full testing recommended where possible."
        )

    return ModelComparisonResponse(
        full_feature_model={
            "model":    META["full_model_name"],
            "features": FEATURES,
            **m["full_feature"],
        },
        single_feature_model={
            "model":   META["single_model_name"],
            "feature": BEST_FEATURE,
            **m["single_feature"],
        },
        best_feature=BEST_FEATURE,
        accuracy_drop_pct=drop,
        cost_reduction_pct=META["cost_reduction_pct"],
        verdict=verdict,
    )
