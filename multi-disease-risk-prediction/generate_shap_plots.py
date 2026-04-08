"""
Generate SHAP summary and bar plots for all 3 diseases.
These are needed by the dashboard's Model Comparison page.

Run from project root:
    python generate_shap_plots.py
"""

import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

MODELS_DIR = "models/saved_models/"
FIGURES_DIR = "results/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

DISEASES = {
    "diabetes": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ],
    "heart": [
        "Age", "Sex", "RestBP", "Chol", "Fbs", "RestECG", "MaxHR",
        "ExAng", "Oldpeak", "Slope", "Ca",
        "ChestPain_nonanginal", "ChestPain_nontypical", "ChestPain_typical",
        "Thal_normal", "Thal_reversable"
    ],
    "kidney": [
        "age", "bp", "sg", "al", "su",
        "rbc", "pc", "pcc", "ba",
        "bgr", "bu", "sc", "sod", "pot",
        "hemo", "pcv", "wc", "rc",
        "htn", "dm", "cad", "appet", "pe", "ane"
    ],
}


def generate_shap_plots(disease, feature_names):
    """Generate SHAP summary and bar plots for a disease's RF model."""
    print(f"\n{'='*50}")
    print(f"  Generating SHAP plots for: {disease.upper()}")
    print(f"{'='*50}")

    # Load model and training data
    rf_model = joblib.load(f"{MODELS_DIR}rf_{disease}.joblib")
    X_train = np.load(f"{MODELS_DIR}X_train_{disease}.npy")

    print(f"  Model loaded: rf_{disease}.joblib")
    print(f"  Training data shape: {X_train.shape}")

    # Use a subsample for speed (max 200 samples)
    if X_train.shape[0] > 200:
        idx = np.random.RandomState(42).choice(X_train.shape[0], 200, replace=False)
        X_sample = X_train[idx]
    else:
        X_sample = X_train

    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values is a list [class_0, class_1]
    # We want class 1 (disease positive)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # ── SHAP Summary Plot (beeswarm) ──────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary — {disease.capitalize()} (Random Forest)", fontsize=14)
    plt.tight_layout()
    summary_path = f"{FIGURES_DIR}shap_summary_{disease}_rf.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {summary_path}")

    # ── SHAP Bar Plot (mean |SHAP|) ───────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {disease.capitalize()} (Random Forest)", fontsize=14)
    plt.tight_layout()
    bar_path = f"{FIGURES_DIR}shap_bar_{disease}_rf.png"
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {bar_path}")


if __name__ == "__main__":
    for disease, features in DISEASES.items():
        generate_shap_plots(disease, features)

    print(f"\n🎉 All SHAP plots generated successfully!")
    print(f"   Check: {FIGURES_DIR}")
