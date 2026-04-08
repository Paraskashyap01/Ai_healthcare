"""
Diabetes - Model Training
Member 1's responsibility

Models trained:
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Voting Ensemble (combines all 3)

Run preprocess_diabetes.py first, then this file.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import joblib
import os
import sys

# Add src to path so we can import evaluate.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.evaluate import evaluate_model
from data.preprocess_diabetes import load_data, preprocess, split_and_smote, scale_features

MODELS_DIR = "models/saved_models/"
os.makedirs(MODELS_DIR, exist_ok=True)

DISEASE = "diabetes"

def train_all_models(X_train, y_train, X_test, y_test, feature_names):
    """Train all models, evaluate them, and save the best one."""

    all_metrics = []

    # ── 1. Logistic Regression ───────────────────────────────────────────────
    print("\n🔵 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    metrics = evaluate_model(lr, X_test, y_test, "Logistic Regression", DISEASE)
    all_metrics.append(metrics)

    # ── 2. Random Forest ─────────────────────────────────────────────────────
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    metrics = evaluate_model(rf, X_test, y_test, "Random Forest", DISEASE)
    all_metrics.append(metrics)

    # ── 3. XGBoost ───────────────────────────────────────────────────────────
    print("\n⚡ Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    metrics = evaluate_model(xgb, X_test, y_test, "XGBoost", DISEASE)
    all_metrics.append(metrics)

    # ── 4. Voting Ensemble ───────────────────────────────────────────────────
    print("\n🗳️  Training Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
        voting='soft'  # soft = uses probabilities, usually better
    )
    ensemble.fit(X_train, y_train)
    metrics = evaluate_model(ensemble, X_test, y_test, "Voting Ensemble", DISEASE)
    all_metrics.append(metrics)

    # ── Save best model (by AUC-ROC) ─────────────────────────────────────────
    best = max(all_metrics, key=lambda x: x['AUC-ROC'])
    print(f"\n🏆 Best Model: {best['Model']} with AUC-ROC = {best['AUC-ROC']}")

    # Save all models individually for SHAP/LIME later
    joblib.dump(rf, f"{MODELS_DIR}rf_{DISEASE}.joblib")
    joblib.dump(xgb, f"{MODELS_DIR}xgb_{DISEASE}.joblib")
    joblib.dump(ensemble, f"{MODELS_DIR}ensemble_{DISEASE}.joblib")

    # Save training data for LIME explainability in dashboard
    np.save(f"{MODELS_DIR}X_train_{DISEASE}.npy", X_train)
    print("All models and training data saved.")

    # Save metrics to CSV
    pd.DataFrame(all_metrics).to_csv(f"results/metrics/metrics_{DISEASE}.csv", index=False)
    print(f"Metrics saved to results/metrics/metrics_{DISEASE}.csv")

    return rf, xgb, ensemble, feature_names


if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test, feature_names = split_and_smote(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    train_all_models(X_train_scaled, y_train, X_test_scaled, y_test, feature_names)
    print(f"\n✅ {DISEASE.capitalize()} model training complete.")
