"""
SHAP Explainability
Member 4's responsibility

This script generates:
1. Global SHAP summary plot — which features matter most overall
2. Per-patient waterfall plot — why this specific prediction was made

Run AFTER all models are trained and saved.
"""

import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

MODELS_DIR = "models/saved_models/"
FIGURES_DIR = "results/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)


def get_shap_explainer(model, X_train, model_type="tree"):
    """
    Create the right SHAP explainer based on model type.

    - tree: for Random Forest and XGBoost (fast, exact)
    - linear: for Logistic Regression
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # Kernel is slowest but works for any model
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    return explainer


def plot_global_summary(explainer, X_test, feature_names, disease_name, model_name):
    """
    Summary plot showing global feature importance.
    Each dot is one patient — color shows feature value (red=high, blue=low).
    X-axis shows impact on model output.
    """
    shap_values = explainer.shap_values(X_test)

    # Handle different SHAP return formats:
    # - Old API: list of arrays [class0_array, class1_array]
    # - New API: ndarray with shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use class 1 (disease present)
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # Select class 1

    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False
    )
    path = f"{FIGURES_DIR}shap_summary_{disease_name}_{model_name}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP summary plot saved: {path}")
    return shap_values


def plot_bar_summary(explainer, X_test, feature_names, disease_name, model_name):
    """
    Bar plot showing mean absolute SHAP values per feature.
    Simpler view of global feature importance.
    """
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    path = f"{FIGURES_DIR}shap_bar_{disease_name}_{model_name}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP bar plot saved: {path}")


def plot_waterfall_single_patient(explainer, X_test, shap_values, patient_index, feature_names, disease_name):
    """
    Waterfall plot for a single patient showing which features pushed
    the prediction up (red) or down (blue).
    """
    expected = explainer.expected_value
    if isinstance(expected, list):
        expected = expected[1]
    elif hasattr(expected, '__len__') and len(expected) > 1:
        expected = expected[1]

    # Ensure we have 1D shap values for a single patient
    patient_shap = shap_values[patient_index]
    if hasattr(patient_shap, 'ndim') and patient_shap.ndim == 2:
        patient_shap = patient_shap[:, 1]  # Select class 1

    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=patient_shap,
            base_values=float(expected),
            data=X_test[patient_index],
            feature_names=feature_names
        ),
        show=False
    )
    path = f"{FIGURES_DIR}shap_waterfall_{disease_name}_patient{patient_index}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP waterfall plot saved: {path}")


def explain_disease(disease_name, X_test, feature_names):
    """
    Full SHAP explanation pipeline for one disease.
    Loads the saved Random Forest model and generates all plots.
    """
    print(f"\n{'='*50}")
    print(f"  SHAP Explanation — {disease_name.upper()}")
    print(f"{'='*50}")

    model_path = f"{MODELS_DIR}rf_{disease_name}.joblib"
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}. Run train_{disease_name}.py first.")
        return None, None

    model = joblib.load(model_path)
    explainer = get_shap_explainer(model, X_test, model_type="tree")

    # Global summary (dot plot)
    shap_values = plot_global_summary(explainer, X_test, feature_names, disease_name, "rf")

    # Global summary (bar plot)
    plot_bar_summary(explainer, X_test, feature_names, disease_name, "rf")

    # Explain first 3 patients as examples
    for i in range(min(3, len(X_test))):
        plot_waterfall_single_patient(explainer, X_test, shap_values, i, feature_names, disease_name)

    return explainer, shap_values


def get_shap_for_patient(model, patient_data, feature_names):
    """
    Used by the dashboard — returns SHAP values for a single patient input.

    Args:
        model: trained model
        patient_data: numpy array of shape (1, n_features)
        feature_names: list of feature names

    Returns:
        shap.Explanation object for use in shap.waterfall_plot()
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    expected = explainer.expected_value
    if isinstance(expected, list):
        expected = expected[1]
    elif hasattr(expected, '__len__') and len(expected) > 1:
        expected = expected[1]

    return shap.Explanation(
        values=shap_values[0],
        base_values=float(expected),
        data=patient_data[0],
        feature_names=feature_names
    )


# ── Run directly to generate all SHAP plots ──────────────────────────────────
if __name__ == "__main__":

    # ── Diabetes ──────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  DIABETES — SHAP ANALYSIS")
    print("═"*60)
    try:
        from data.preprocess_diabetes import load_data as load_diabetes, preprocess as prep_diabetes
        from data.preprocess_diabetes import split_and_smote as split_diabetes, scale_features as scale_diabetes

        df = load_diabetes()
        df = prep_diabetes(df)
        X_train, X_test, y_train, y_test, features = split_diabetes(df)
        X_train_s, X_test_s = scale_diabetes(X_train, X_test)
        explain_disease("diabetes", X_test_s, features)
    except Exception as e:
        print(f"⚠️  Diabetes SHAP failed: {e}")

    # ── Heart Disease ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  HEART DISEASE — SHAP ANALYSIS")
    print("═"*60)
    try:
        from data.preprocess_heart import load_data as load_heart, preprocess as prep_heart
        from data.preprocess_heart import split_and_smote as split_heart, scale_features as scale_heart

        df = load_heart()
        df = prep_heart(df)
        X_train, X_test, y_train, y_test, features = split_heart(df)
        X_train_s, X_test_s = scale_heart(X_train, X_test)
        explain_disease("heart", X_test_s, features)
    except Exception as e:
        print(f"⚠️  Heart SHAP failed: {e}")

    # ── Kidney Disease ────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  KIDNEY DISEASE — SHAP ANALYSIS")
    print("═"*60)
    try:
        from data.preprocess_kidney import load_data as load_kidney, preprocess as prep_kidney
        from data.preprocess_kidney import split_and_smote as split_kidney, scale_features as scale_kidney

        df = load_kidney()
        df = prep_kidney(df)
        X_train, X_test, y_train, y_test, features = split_kidney(df)
        X_train_s, X_test_s = scale_kidney(X_train, X_test)
        explain_disease("kidney", X_test_s, features)
    except Exception as e:
        print(f"⚠️  Kidney SHAP failed: {e}")

    print("\n✅ SHAP analysis complete for all diseases!")



















