"""
Diabetes Page — Streamlit Dashboard
Member 5's responsibility

Shows:
- Patient input sliders
- Risk prediction + probability
- SHAP waterfall plot
- LIME explanation
- What-If analysis slider
"""

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

MODELS_DIR = "models/saved_models/"


@st.cache_resource
def load_model_and_scaler():
    """Load saved model and scaler. Cached so it only loads once."""
    try:
        model = joblib.load(f"{MODELS_DIR}ensemble_diabetes.joblib")
        scaler = joblib.load(f"{MODELS_DIR}scaler_diabetes.joblib")
        return model, scaler
    except FileNotFoundError:
        return None, None


@st.cache_resource
def load_training_data():
    """Load saved training data for LIME explainability."""
    try:
        X_train = np.load(f"{MODELS_DIR}X_train_diabetes.npy")
        return X_train
    except FileNotFoundError:
        return None


def show():
    st.title("🩸 Diabetes Risk Prediction")
    st.markdown("Enter patient details below to predict diabetes risk.")
    st.markdown("---")

    model, scaler = load_model_and_scaler()
    if model is None:
        st.error("⚠️ Model not found. Please run `python src/models/train_diabetes.py` first.")
        return

    # ── Patient Input ──────────────────────────────────────────────────────────
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 3, key="d_preg")
        glucose = st.slider("Glucose (mg/dL)", 0, 200, 120, key="d_gluc")
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70, key="d_bp")
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20, key="d_skin")

    with col2:
        insulin = st.slider("Insulin (mu U/ml)", 0, 846, 80, key="d_ins")
        bmi = st.slider("BMI", 0.0, 67.0, 25.0, step=0.1, key="d_bmi")
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01, key="d_dpf")
        age = st.slider("Age", 21, 81, 30, key="d_age")

    patient_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, dpf, age]])

    # ── Prediction ────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔍 Predict Diabetes Risk", type="primary", key="d_predict"):
        patient_scaled = scaler.transform(patient_input)
        probability = model.predict_proba(patient_scaled)[0][1]
        prediction = model.predict(patient_scaled)[0]

        # Store results in session state so they persist across re-renders
        st.session_state["d_probability"] = probability
        st.session_state["d_prediction"] = prediction
        st.session_state["d_patient_scaled"] = patient_scaled
        st.session_state["d_patient_input"] = patient_input

    # ── Display results (from session state) ──────────────────────────────────
    if "d_probability" in st.session_state:
        probability = st.session_state["d_probability"]
        prediction = st.session_state["d_prediction"]
        patient_scaled = st.session_state["d_patient_scaled"]

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error(f"### ⚠️ High Risk of Diabetes\nProbability: **{probability*100:.1f}%**")
            else:
                st.success(f"### ✅ Low Risk of Diabetes\nProbability: **{probability*100:.1f}%**")

        with col2:
            st.metric("Risk Probability", f"{probability*100:.1f}%")
            st.progress(float(probability))

        # ── SHAP Explanation ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Why this prediction? (SHAP)")
        st.markdown("Red bars increase risk, blue bars decrease risk.")

        try:
            from src.explainability.shap_explainer import get_shap_for_patient
            import shap

            rf_model = joblib.load(f"{MODELS_DIR}rf_diabetes.joblib")
            explanation = get_shap_for_patient(rf_model, patient_scaled, FEATURE_NAMES)

            fig, ax = plt.subplots()
            shap.waterfall_plot(explanation, show=False)
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        # ── LIME Explanation ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Why this prediction? (LIME)")
        st.markdown("LIME fits a simple model locally to explain this specific prediction.")

        try:
            from src.explainability.lime_explainer import create_lime_explainer, get_lime_for_patient

            rf_model = joblib.load(f"{MODELS_DIR}rf_diabetes.joblib")

            # Use real training data for LIME (not random dummy data)
            X_train = load_training_data()
            if X_train is None:
                st.warning("Training data not found. Please re-run `python src/models/train_diabetes.py` to save training data for LIME.")
            else:
                lime_explainer = create_lime_explainer(X_train, FEATURE_NAMES)
                lime_list, lime_fig = get_lime_for_patient(lime_explainer, rf_model, patient_scaled[0], FEATURE_NAMES)

                st.pyplot(lime_fig)
                plt.close()

                st.markdown("**Feature Contributions:**")
                for feature, weight in lime_list[:8]:
                    direction = "🔴 ↑ risk" if weight > 0 else "🔵 ↓ risk"
                    st.markdown(f"- {feature}: `{weight:.4f}` {direction}")
        except Exception as e:
            st.warning(f"LIME explanation unavailable: {e}")

        # ── What-If Analysis ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔄 What-If Analysis")
        st.markdown("Change one value and see how the risk changes.")

        whatif_feature = st.selectbox("Select feature to change", FEATURE_NAMES, key="d_whatif_feat")
        feat_idx = FEATURE_NAMES.index(whatif_feature)
        current_val = patient_input[0][feat_idx]

        whatif_val = st.slider(
            f"What if {whatif_feature} was...",
            float(current_val * 0.5), float(current_val * 1.5 + 1),
            float(current_val), key="d_whatif_val"
        )

        whatif_input = patient_input.copy()
        whatif_input[0][feat_idx] = whatif_val
        whatif_scaled = scaler.transform(whatif_input)
        whatif_prob = model.predict_proba(whatif_scaled)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Risk", f"{probability*100:.1f}%")
        with col2:
            delta = whatif_prob - probability
            st.metric("New Risk", f"{whatif_prob*100:.1f}%",
                      delta=f"{delta*100:+.1f}%",
                      delta_color="inverse")
