"""
Heart Disease Page — Streamlit Dashboard
Member 5's responsibility

Shows:
- Patient input sliders/selectboxes
- Risk prediction + probability
- SHAP waterfall plot
- LIME explanation
- What-If analysis
"""

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Feature names AFTER one-hot encoding (matches what models were trained on)
FEATURE_NAMES = [
    "Age", "Sex", "RestBP", "Chol", "Fbs", "RestECG", "MaxHR",
    "ExAng", "Oldpeak", "Slope", "Ca",
    "ChestPain_nonanginal", "ChestPain_nontypical", "ChestPain_typical",
    "Thal_normal", "Thal_reversable"
]

MODELS_DIR = "models/saved_models/"


@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(f"{MODELS_DIR}ensemble_heart.joblib")
        scaler = joblib.load(f"{MODELS_DIR}scaler_heart.joblib")
        return model, scaler
    except FileNotFoundError:
        return None, None


@st.cache_resource
def load_training_data():
    """Load saved training data for LIME explainability."""
    try:
        X_train = np.load(f"{MODELS_DIR}X_train_heart.npy")
        return X_train
    except FileNotFoundError:
        return None


def show():
    st.title("❤️ Heart Disease Risk Prediction")
    st.markdown("Enter patient details below to predict heart disease risk.")
    st.markdown("---")

    model, scaler = load_model_and_scaler()
    if model is None:
        st.error("⚠️ Model not found. Please run `python src/models/train_heart.py` first.")
        return

    # ── Patient Input ──────────────────────────────────────────────────────────
    st.subheader("Patient Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 80, 55, key="h_age")
        sex = st.selectbox("Sex", ["Male", "Female"], key="h_sex")
        chest_pain = st.selectbox("Chest Pain Type",
                                  ["asymptomatic", "nonanginal", "nontypical", "typical"],
                                  key="h_cp")
        rest_bp = st.slider("Resting Blood Pressure", 80, 200, 130, key="h_bp")

    with col2:
        chol = st.slider("Cholesterol (mg/dL)", 100, 600, 240, key="h_chol")
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"], key="h_fbs")
        rest_ecg = st.selectbox("Resting ECG", [0, 1, 2], key="h_ecg",
                                help="0=Normal, 1=ST-T wave abnormality, 2=LV hypertrophy")
        max_hr = st.slider("Max Heart Rate", 60, 210, 150, key="h_maxhr")

    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], key="h_exang")
        oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 7.0, 1.0, step=0.1, key="h_old")
        slope = st.selectbox("Slope of Peak Exercise ST", [1, 2, 3], key="h_slope")
        ca = st.slider("Number of Major Vessels (Ca)", 0, 3, 0, key="h_ca")
        thal = st.selectbox("Thalassemia", ["fixed", "normal", "reversable"], key="h_thal")

    # Encode categorical features
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0

    # One-hot encode ChestPain (drop_first=True, dropped "asymptomatic")
    cp_nonanginal = 1 if chest_pain == "nonanginal" else 0
    cp_nontypical = 1 if chest_pain == "nontypical" else 0
    cp_typical = 1 if chest_pain == "typical" else 0

    # One-hot encode Thal (drop_first=True, dropped "fixed")
    thal_normal = 1 if thal == "normal" else 0
    thal_reversable = 1 if thal == "reversable" else 0

    patient_input = np.array([[
        age, sex_val, rest_bp, chol, fbs_val, rest_ecg, max_hr,
        exang_val, oldpeak, slope, ca,
        cp_nonanginal, cp_nontypical, cp_typical,
        thal_normal, thal_reversable
    ]])

    # ── Prediction ────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔍 Predict Heart Disease Risk", type="primary", key="h_predict"):
        patient_scaled = scaler.transform(patient_input)
        probability = model.predict_proba(patient_scaled)[0][1]
        prediction = model.predict(patient_scaled)[0]

        # Store results in session state so they persist across re-renders
        st.session_state["h_probability"] = probability
        st.session_state["h_prediction"] = prediction
        st.session_state["h_patient_scaled"] = patient_scaled
        st.session_state["h_patient_input"] = patient_input

    # ── Display results (from session state) ──────────────────────────────────
    if "h_probability" in st.session_state:
        probability = st.session_state["h_probability"]
        prediction = st.session_state["h_prediction"]
        patient_scaled = st.session_state["h_patient_scaled"]

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error(f"### ⚠️ High Risk of Heart Disease\nProbability: **{probability*100:.1f}%**")
            else:
                st.success(f"### ✅ Low Risk of Heart Disease\nProbability: **{probability*100:.1f}%**")

        with col2:
            st.metric("Risk Probability", f"{probability*100:.1f}%")
            st.progress(float(probability))

        # ── SHAP Explanation ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Why this prediction? (SHAP)")

        try:
            from src.explainability.shap_explainer import get_shap_for_patient
            import shap

            rf_model = joblib.load(f"{MODELS_DIR}rf_heart.joblib")
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

        try:
            from src.explainability.lime_explainer import create_lime_explainer, get_lime_for_patient

            rf_model = joblib.load(f"{MODELS_DIR}rf_heart.joblib")

            # Use real training data for LIME (not random dummy data)
            X_train = load_training_data()
            if X_train is None:
                st.warning("Training data not found. Please re-run `python src/models/train_heart.py` to save training data for LIME.")
            else:
                lime_explainer = create_lime_explainer(X_train, FEATURE_NAMES)
                lime_list, lime_fig = get_lime_for_patient(lime_explainer, rf_model, patient_scaled[0], FEATURE_NAMES)

                st.pyplot(lime_fig)
                plt.close()

                st.markdown("**Top Feature Contributions:**")
                for feature, weight in lime_list[:8]:
                    direction = "🔴 ↑ risk" if weight > 0 else "🔵 ↓ risk"
                    st.markdown(f"- {feature}: `{weight:.4f}` {direction}")
        except Exception as e:
            st.warning(f"LIME explanation unavailable: {e}")

        # ── What-If Analysis ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔄 What-If Analysis")

        whatif_col = st.selectbox("Change which feature?",
                                  ["Cholesterol", "Max Heart Rate", "Oldpeak", "Age"],
                                  key="h_whatif_feat")

        idx_map = {"Cholesterol": 3, "Max Heart Rate": 6, "Oldpeak": 8, "Age": 0}
        range_map = {
            "Cholesterol": (100, 600, int(chol)),
            "Max Heart Rate": (60, 210, int(max_hr)),
            "Oldpeak": (0.0, 7.0, float(oldpeak)),
            "Age": (20, 80, int(age))
        }

        feat_idx = idx_map[whatif_col]
        rng = range_map[whatif_col]
        whatif_val = st.slider(f"What if {whatif_col} was...", rng[0], rng[1], rng[2], key="h_whatif_val")

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
