"""
AgriOptima - Model Training Script
Trains Logistic Regression and Gaussian Naive Bayes on crop recommendation data.
Performs univariate feature selection to identify the single best soil attribute.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import f_classif

# ──────────────────────────────────────────────
# 1.  GENERATE REALISTIC SYNTHETIC DATASET
#     (mirrors the Kaggle Crop Recommendation
#      dataset distribution for N, P, K, pH)
# ──────────────────────────────────────────────

CROPS = {
    "rice":       {"N": (60, 100), "P": (30, 60),  "K": (30, 50),  "pH": (5.5, 7.0)},
    "maize":      {"N": (60, 100), "P": (50, 80),  "K": (15, 30),  "pH": (5.5, 7.0)},
    "chickpea":   {"N": (20, 60),  "P": (50, 90),  "K": (60, 100), "pH": (6.0, 8.5)},
    "kidneybeans":{"N": (10, 40),  "P": (55, 90),  "K": (15, 30),  "pH": (5.5, 7.0)},
    "pigeonpeas": {"N": (10, 40),  "P": (55, 90),  "K": (15, 30),  "pH": (5.5, 7.0)},
    "mothbeans":  {"N": (15, 45),  "P": (30, 60),  "K": (15, 30),  "pH": (3.5, 7.0)},
    "mungbean":   {"N": (10, 40),  "P": (30, 60),  "K": (15, 30),  "pH": (6.0, 7.5)},
    "blackgram":  {"N": (10, 40),  "P": (55, 90),  "K": (15, 30),  "pH": (5.5, 7.5)},
    "lentil":     {"N": (10, 30),  "P": (55, 90),  "K": (15, 30),  "pH": (6.5, 8.5)},
    "pomegranate":{"N": (15, 45),  "P": (55, 90),  "K": (35, 55),  "pH": (5.5, 7.5)},
    "banana":     {"N": (80, 120), "P": (55, 90),  "K": (40, 60),  "pH": (5.5, 7.0)},
    "mango":      {"N": (15, 45),  "P": (10, 30),  "K": (20, 40),  "pH": (4.5, 7.0)},
    "grapes":     {"N": (10, 30),  "P": (100,160), "K": (150,210), "pH": (5.5, 7.0)},
    "watermelon": {"N": (80, 120), "P": (10, 30),  "K": (40, 60),  "pH": (6.0, 7.5)},
    "muskmelon":  {"N": (80, 120), "P": (10, 30),  "K": (40, 60),  "pH": (6.0, 7.5)},
    "apple":      {"N": (0,  20),  "P": (100,160), "K": (150,210), "pH": (5.5, 7.0)},
    "orange":     {"N": (0,  20),  "P": (10, 30),  "K": (5,  20),  "pH": (6.0, 7.5)},
    "papaya":     {"N": (40, 60),  "P": (55, 80),  "K": (40, 60),  "pH": (6.5, 8.5)},
    "coconut":    {"N": (15, 35),  "P": (5,  30),  "K": (25, 45),  "pH": (5.0, 8.0)},
    "cotton":     {"N": (100,140), "P": (15, 40),  "K": (15, 30),  "pH": (6.0, 7.5)},
    "jute":       {"N": (60, 100), "P": (30, 60),  "K": (30, 50),  "pH": (6.0, 8.0)},
    "coffee":     {"N": (80, 120), "P": (30, 60),  "K": (25, 45),  "pH": (3.5, 6.5)},
}

SAMPLES_PER_CROP = 100
np.random.seed(42)

rows = []
for crop, ranges in CROPS.items():
    n_samples = SAMPLES_PER_CROP
    N   = np.random.uniform(*ranges["N"],   n_samples)
    P   = np.random.uniform(*ranges["P"],   n_samples)
    K   = np.random.uniform(*ranges["K"],   n_samples)
    pH  = np.random.uniform(*ranges["pH"],  n_samples)
    # add small Gaussian noise
    N  += np.random.normal(0, 2, n_samples)
    P  += np.random.normal(0, 2, n_samples)
    K  += np.random.normal(0, 2, n_samples)
    pH += np.random.normal(0, 0.1, n_samples)
    pH  = np.clip(pH, 3.0, 9.5)
    for i in range(n_samples):
        rows.append({"N": N[i], "P": P[i], "K": K[i], "pH": pH[i], "label": crop})

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/crop_data.csv", index=False)
print(f"[DATA] Generated {len(df)} samples across {len(CROPS)} crops.")

# ──────────────────────────────────────────────
# 2.  PREPROCESSING
# ──────────────────────────────────────────────

FEATURES = ["N", "P", "K", "pH"]

le = LabelEncoder()
df["crop_encoded"] = le.fit_transform(df["label"])

X = df[FEATURES].values
y = df["crop_encoded"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

# ──────────────────────────────────────────────
# 3.  FEATURE SELECTION — UNIVARIATE (F-score)
# ──────────────────────────────────────────────

f_scores, p_values = f_classif(X_train, y_train)

feature_importance = {
    feat: {"f_score": float(f_scores[i]), "p_value": float(p_values[i])}
    for i, feat in enumerate(FEATURES)
}

# rank descending by F-score
ranked_features = sorted(FEATURES, key=lambda f: feature_importance[f]["f_score"], reverse=True)
best_feature     = ranked_features[0]
best_feature_idx = FEATURES.index(best_feature)

print(f"\n[FEATURE SELECTION] Univariate F-scores:")
for feat in ranked_features:
    print(f"  {feat:4s}  F={feature_importance[feat]['f_score']:.2f}  p={feature_importance[feat]['p_value']:.4f}")
print(f"  → Best single feature: {best_feature}")

# ──────────────────────────────────────────────
# 4.  TRAIN FULL-FEATURE MODELS
# ──────────────────────────────────────────────

def evaluate(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return {
        "accuracy":  round(float(accuracy_score(y_te, preds)), 4),
        "precision": round(float(precision_score(y_te, preds, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_te, preds, average="weighted", zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_te, preds, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_te, preds).tolist(),
    }

lr_full  = LogisticRegression(max_iter=1000, random_state=42)
gnb_full = GaussianNB()

metrics_lr_full  = evaluate(lr_full,  X_train_sc, y_train, X_test_sc, y_test)
metrics_gnb_full = evaluate(gnb_full, X_train_sc, y_train, X_test_sc, y_test)

print(f"\n[FULL FEATURES]  LR  accuracy={metrics_lr_full['accuracy']}  F1={metrics_lr_full['f1_score']}")
print(f"[FULL FEATURES]  GNB accuracy={metrics_gnb_full['accuracy']}  F1={metrics_gnb_full['f1_score']}")

# pick best full-feature model
best_full_model  = lr_full if metrics_lr_full["f1_score"] >= metrics_gnb_full["f1_score"] else gnb_full
best_full_name   = "LogisticRegression" if metrics_lr_full["f1_score"] >= metrics_gnb_full["f1_score"] else "GaussianNaiveBayes"
metrics_full_best = metrics_lr_full if best_full_name == "LogisticRegression" else metrics_gnb_full

# ──────────────────────────────────────────────
# 5.  TRAIN SINGLE-FEATURE MODELS
# ──────────────────────────────────────────────

X_train_single = X_train_sc[:, best_feature_idx].reshape(-1, 1)
X_test_single  = X_test_sc[:,  best_feature_idx].reshape(-1, 1)

lr_single  = LogisticRegression(max_iter=1000, random_state=42)
gnb_single = GaussianNB()

metrics_lr_single  = evaluate(lr_single,  X_train_single, y_train, X_test_single, y_test)
metrics_gnb_single = evaluate(gnb_single, X_train_single, y_train, X_test_single, y_test)

print(f"\n[SINGLE FEATURE] LR  accuracy={metrics_lr_single['accuracy']}  F1={metrics_lr_single['f1_score']}")
print(f"[SINGLE FEATURE] GNB accuracy={metrics_gnb_single['accuracy']}  F1={metrics_gnb_single['f1_score']}")

best_single_model  = lr_single if metrics_lr_single["f1_score"] >= metrics_gnb_single["f1_score"] else gnb_single
best_single_name   = "LogisticRegression" if metrics_lr_single["f1_score"] >= metrics_gnb_single["f1_score"] else "GaussianNaiveBayes"
metrics_single_best = metrics_lr_single if best_single_name == "LogisticRegression" else metrics_gnb_single

# ──────────────────────────────────────────────
# 6.  SAVE ARTEFACTS
# ──────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

joblib.dump(best_full_model,   "models/full_model.pkl")
joblib.dump(best_single_model, "models/single_model.pkl")
joblib.dump(scaler,            "models/scaler.pkl")
joblib.dump(le,                "models/label_encoder.pkl")

metadata = {
    "features":          FEATURES,
    "best_feature":      best_feature,
    "best_feature_idx":  best_feature_idx,
    "crops":             le.classes_.tolist(),
    "full_model_name":   best_full_name,
    "single_model_name": best_single_name,
    "metrics": {
        "full_feature":   {k: v for k, v in metrics_full_best.items() if k != "confusion_matrix"},
        "single_feature": {k: v for k, v in metrics_single_best.items() if k != "confusion_matrix"},
    },
    "feature_importance": {
        feat: {
            "rank":    ranked_features.index(feat) + 1,
            "f_score": feature_importance[feat]["f_score"],
            "p_value": feature_importance[feat]["p_value"],
        }
        for feat in FEATURES
    },
    "accuracy_drop_pct": round(
        (metrics_full_best["accuracy"] - metrics_single_best["accuracy"])
        / metrics_full_best["accuracy"] * 100, 2
    ),
    "cost_reduction_pct": 75.0,   # project claim: 75% cost reduction
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n[SAVE] Models and metadata written to models/")
print(f"\n=== SUMMARY ===")
print(f"Best feature        : {best_feature}")
print(f"Full model          : {best_full_name}  acc={metrics_full_best['accuracy']}")
print(f"Single-feature model: {best_single_name}  acc={metrics_single_best['accuracy']}")
print(f"Accuracy drop       : {metadata['accuracy_drop_pct']}%")
print(f"Cost reduction      : {metadata['cost_reduction_pct']}%")
