# 🌾 AgriOptima — Resource-Constrained Crop Recommendation

A lightweight ML system that recommends crops to small-scale farmers using **just one soil attribute**,
reducing soil testing costs by ~75% while maintaining practical utility.

---

## Project Summary

| Property | Detail |
|---|---|
| Problem | Comprehensive soil testing (N, P, K, pH) is cost-prohibitive for small farmers |
| Solution | Univariate feature selection identifies the single best predictor; a lightweight classifier uses it |
| Models | Logistic Regression + Gaussian Naïve Bayes (best selected automatically) |
| Crops | 22 classes (rice, maize, coffee, mango, cotton, …) |
| Key finding | **Potassium (K)** is the most discriminative single soil attribute (F=2828) |

---

## Project Structure

```
agrioptima/
├── api/
│   └── main.py            ← FastAPI application
├── data/
│   └── crop_data.csv      ← Generated training dataset
├── models/
│   ├── full_model.pkl     ← Best 4-feature classifier
│   ├── single_model.pkl   ← Best 1-feature classifier
│   ├── scaler.pkl         ← StandardScaler
│   ├── label_encoder.pkl  ← LabelEncoder (crop names)
│   └── metadata.json      ← Feature importance + metrics
├── scripts/
│   └── train.py           ← End-to-end training pipeline
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (generates data/ and models/)
python scripts/train.py

# 3. Start API server
uvicorn api.main:app --reload --port 8000
```

---

## API Endpoints

Once running, open **http://localhost:8000/docs** for interactive Swagger UI.

### `POST /predict/single`
Resource-constrained prediction using **one** soil value (the best feature — K by default).

```json
// Request
{ "value": 30.0 }

// Response
{
  "recommended_crop": "mango",
  "confidence": 0.1784,
  "model_used": "GaussianNaiveBayes",
  "mode": "single_feature",
  "key_nutrient": "K",
  "top_3_crops": [
    {"crop": "mango",   "probability": 0.1784},
    {"crop": "coconut", "probability": 0.1150},
    {"crop": "coffee",  "probability": 0.0991}
  ]
}
```

### `POST /predict/full`
High-accuracy prediction using all 4 soil attributes.

```json
// Request
{ "N": 90.0, "P": 42.0, "K": 43.0, "pH": 6.5 }

// Response
{
  "recommended_crop": "rice",
  "confidence": 0.5149,
  "model_used": "GaussianNaiveBayes",
  "mode": "full_feature",
  "top_3_crops": [...]
}
```

### `GET /feature-importance`
Returns soil attributes ranked by ANOVA F-score.

```json
[
  {"feature": "K",  "rank": 1, "f_score": 2828.92, "p_value": 0.0},
  {"feature": "N",  "rank": 2, "f_score": 1163.77, "p_value": 0.0},
  {"feature": "P",  "rank": 3, "f_score": 815.40,  "p_value": 0.0},
  {"feature": "pH", "rank": 4, "f_score": 87.71,   "p_value": 0.0}
]
```

### `GET /model-comparison`
Cost vs accuracy trade-off analysis.

```json
{
  "full_feature_model":   {"model": "GaussianNaiveBayes", "accuracy": 0.7159, ...},
  "single_feature_model": {"model": "GaussianNaiveBayes", "accuracy": 0.2318, ...},
  "best_feature": "K",
  "accuracy_drop_pct": 67.62,
  "cost_reduction_pct": 75.0,
  "verdict": "..."
}
```

### `GET /crops` — List all 22 supported crop classes
### `GET /health` — Service health check

---

## Team

Shaurya Pethe, Tarun Yadav, Devraj Charan, Navjot Singh, Vishal Malik, Akhil Sharma
