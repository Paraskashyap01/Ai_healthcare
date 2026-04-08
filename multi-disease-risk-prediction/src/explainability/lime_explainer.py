"""
LIME Explainability
Member 4's responsibility

LIME (Local Interpretable Model-agnostic Explanations)
Explains individual predictions by fitting a simple linear model
locally around that data point.

Compare LIME output vs SHAP output — do they agree on which features matter?
"""

import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

MODELS_DIR = "models/saved_models/"
FIGURES_DIR = "results/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)


def create_lime_explainer(X_train, feature_names, class_names=["No Disease", "Disease"]):
    """
    Create a LIME TabularExplainer.

    Args:
        X_train: Training data (numpy array) — LIME uses this to understand feature ranges
        feature_names: List of feature names
        class_names: Labels for the two classes

    Returns:
        LimeTabularExplainer object
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )
    return explainer


def explain_patient_lime(lime_explainer, model, patient_data, feature_names, disease_name, patient_index=0):
    """
    Generate a LIME explanation for a single patient.

    Args:
        lime_explainer: LimeTabularExplainer object
        model: Trained sklearn-compatible model
        patient_data: 1D numpy array of one patient's features
        feature_names: List of feature names
        disease_name: e.g. 'diabetes'
        patient_index: for naming the saved file

    Returns:
        lime explanation object
    """
    explanation = lime_explainer.explain_instance(
        data_row=patient_data,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )

    # Save plot
    fig = explanation.as_pyplot_figure()
    fig.suptitle(f"LIME Explanation — {disease_name} (Patient {patient_index})", fontsize=12)
    path = f"{FIGURES_DIR}lime_{disease_name}_patient{patient_index}.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ LIME explanation saved: {path}")

    # Print top features
    print(f"\nTop LIME features for Patient {patient_index}:")
    for feature, weight in explanation.as_list():
        direction = "↑ risk" if weight > 0 else "↓ risk"
        print(f"  {feature}: {weight:.4f} ({direction})")

    return explanation


def get_lime_for_patient(lime_explainer, model, patient_data, feature_names):
    """
    Used by the dashboard — returns LIME explanation for a single patient.

    Args:
        lime_explainer: LimeTabularExplainer
        model: trained model
        patient_data: 1D numpy array
        feature_names: list of feature names

    Returns:
        List of (feature_condition, weight) tuples for display
    """
    explanation = lime_explainer.explain_instance(
        data_row=patient_data,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    return explanation.as_list(), explanation.as_pyplot_figure()


def explain_disease_lime(disease_name, model, X_train, X_test, feature_names, num_patients=3):
    """
    Full LIME explanation pipeline for one disease.

    Args:
        disease_name: e.g. 'diabetes'
        model: trained model
        X_train: training data (for LIME background)
        X_test: test data (patients to explain)
        feature_names: list of feature names
        num_patients: how many patients to explain (default 3)
    """
    print(f"\n{'='*50}")
    print(f"  LIME Explanation — {disease_name.upper()}")
    print(f"{'='*50}")

    # Create LIME explainer using training data
    lime_explainer = create_lime_explainer(X_train, feature_names)

    # Explain first N patients
    for i in range(min(num_patients, len(X_test))):
        explain_patient_lime(
            lime_explainer, model, X_test[i],
            feature_names, disease_name, patient_index=i
        )

    return lime_explainer


# ── Run directly to generate LIME plots ──────────────────────────────────────
if __name__ == "__main__":

    # ── Diabetes ──────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  DIABETES — LIME ANALYSIS")
    print("═"*60)
    try:
        from data.preprocess_diabetes import load_data as load_diabetes, preprocess as prep_diabetes
        from data.preprocess_diabetes import split_and_smote as split_diabetes, scale_features as scale_diabetes

        df = load_diabetes()
        df = prep_diabetes(df)
        X_train, X_test, y_train, y_test, features = split_diabetes(df)
        X_train_s, X_test_s = scale_diabetes(X_train, X_test)

        model = joblib.load(f"{MODELS_DIR}rf_diabetes.joblib")
        explain_disease_lime("diabetes", model, X_train_s, X_test_s, features)
    except Exception as e:
        print(f"⚠️  Diabetes LIME failed: {e}")

    # ── Heart Disease ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  HEART DISEASE — LIME ANALYSIS")
    print("═"*60)
    try:
        from data.preprocess_heart import load_data as load_heart, preprocess as prep_heart
        from data.preprocess_heart import split_and_smote as split_heart, scale_features as scale_heart

        df = load_heart()
        df = prep_heart(df)
        X_train, X_test, y_train, y_test, features = split_heart(df)
        X_train_s, X_test_s = scale_heart(X_train, X_test)

        model = joblib.load(f"{MODELS_DIR}rf_heart.joblib")
        explain_disease_lime("heart", model, X_train_s, X_test_s, features)
    except Exception as e:
        print(f"⚠️  Heart LIME failed: {e}")

    # ── Kidney Disease ────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  KIDNEY DISEASE — LIME ANALYSIS")
    print("═"*60)
    try:
        from data.preprocess_kidney import load_data as load_kidney, preprocess as prep_kidney
        from data.preprocess_kidney import split_and_smote as split_kidney, scale_features as scale_kidney

        df = load_kidney()
        df = prep_kidney(df)
        X_train, X_test, y_train, y_test, features = split_kidney(df)
        X_train_s, X_test_s = scale_kidney(X_train, X_test)

        model = joblib.load(f"{MODELS_DIR}rf_kidney.joblib")
        explain_disease_lime("kidney", model, X_train_s, X_test_s, features)
    except Exception as e:
        print(f"⚠️  Kidney LIME failed: {e}")

    print("\n✅ LIME analysis complete for all diseases!")
