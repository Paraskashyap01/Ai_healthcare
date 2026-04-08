"""
Kidney Disease Page — Streamlit Dashboard
Member 5's responsibility

Shows:
- Patient input (mix of sliders and checkboxes for binary features)
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

# Feature names after preprocessing (24 features, no 'classification')
FEATURE_NAMES = [
    "age", "bp", "sg", "al", "su",
    "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane"
]

MODELS_DIR = "models/saved_models/"


@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(f"{MODELS_DIR}ensemble_kidney.joblib")
        scaler = joblib.load(f"{MODELS_DIR}scaler_kidney.joblib")
        return model, scaler
    except FileNotFoundError:
        return None, None


@st.cache_resource
def load_training_data():
    """Load saved training data for LIME explainability."""
    try:
        X_train = np.load(f"{MODELS_DIR}X_train_kidney.npy")
        return X_train
    except FileNotFoundError:
        return None


def show():
    st.title("🫘 Kidney Disease Risk Prediction")
    st.markdown("Enter patient details below to predict chronic kidney disease (CKD) risk.")
    st.markdown("---")

    model, scaler = load_model_and_scaler()
    if model is None:
        st.error("⚠️ Model not found. Please run `python src/models/train_kidney.py` first.")
        return

    # ── Patient Input ──────────────────────────────────────────────────────────
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Vitals**")
        age = st.slider("Age", 2, 90, 50, key="k_age")
        bp = st.slider("Blood Pressure (mm Hg)", 50, 180, 80, key="k_bp")
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=2, key="k_sg")
        al = st.slider("Albumin (0-5)", 0, 5, 0, key="k_al")
        su = st.slider("Sugar (0-5)", 0, 5, 0, key="k_su")
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"], key="k_rbc")
        pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"], key="k_pc")
        pcc = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"], key="k_pcc")

    with col2:
        st.markdown("**Blood Tests**")
        ba = st.selectbox("Bacteria", ["Not Present", "Present"], key="k_ba")
        bgr = st.slider("Blood Glucose Random (mg/dL)", 22, 490, 120, key="k_bgr")
        bu = st.slider("Blood Urea (mg/dL)", 1, 400, 40, key="k_bu")
        sc = st.slider("Serum Creatinine (mg/dL)", 0.4, 15.0, 1.2, step=0.1, key="k_sc")
        sod = st.slider("Sodium (mEq/L)", 100, 163, 140, key="k_sod")
        pot = st.slider("Potassium (mEq/L)", 2.5, 47.0, 4.5, step=0.1, key="k_pot")
        hemo = st.slider("Hemoglobin (g/dL)", 3.0, 18.0, 13.0, step=0.1, key="k_hemo")
        pcv = st.slider("Packed Cell Volume", 9, 54, 40, key="k_pcv")

    with col3:
        st.markdown("**Additional Tests & History**")
        wc = st.slider("White Blood Cell Count", 2200, 26400, 8000, step=100, key="k_wc")
        rc = st.slider("Red Blood Cell Count (M/cmm)", 2.1, 8.0, 5.0, step=0.1, key="k_rc")
        htn = st.selectbox("Hypertension", ["No", "Yes"], key="k_htn")
        dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"], key="k_dm")
        cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"], key="k_cad")
        appet = st.selectbox("Appetite", ["Good", "Poor"], key="k_appet")
        pe = st.selectbox("Pedal Edema", ["No", "Yes"], key="k_pe")
        ane = st.selectbox("Anemia", ["No", "Yes"], key="k_ane")

    # Encode inputs
    rbc_val = 1 if rbc == "Abnormal" else 0
    pc_val = 1 if pc == "Abnormal" else 0
    pcc_val = 1 if pcc == "Present" else 0
    ba_val = 1 if ba == "Present" else 0
    htn_val = 1 if htn == "Yes" else 0
    dm_val = 1 if dm == "Yes" else 0
    cad_val = 1 if cad == "Yes" else 0
    appet_val = 1 if appet == "Good" else 0
    pe_val = 1 if pe == "Yes" else 0
    ane_val = 1 if ane == "Yes" else 0

    patient_input = np.array([[
        age, bp, sg, al, su,
        rbc_val, pc_val, pcc_val, ba_val,
        bgr, bu, sc, sod, pot,
        hemo, pcv, wc, rc,
        htn_val, dm_val, cad_val, appet_val, pe_val, ane_val
    ]])

    # ── Prediction ────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔍 Predict Kidney Disease Risk", type="primary", key="k_predict"):
        patient_scaled = scaler.transform(patient_input)
        probability = model.predict_proba(patient_scaled)[0][1]
        prediction = model.predict(patient_scaled)[0]

        # Store results in session state so they persist across re-renders
        st.session_state["k_probability"] = probability
        st.session_state["k_prediction"] = prediction
        st.session_state["k_patient_scaled"] = patient_scaled
        st.session_state["k_patient_input"] = patient_input

    # ── Display results (from session state) ──────────────────────────────────
    if "k_probability" in st.session_state:
        probability = st.session_state["k_probability"]
        prediction = st.session_state["k_prediction"]
        patient_scaled = st.session_state["k_patient_scaled"]

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error(f"### ⚠️ High Risk of Chronic Kidney Disease\nProbability: **{probability*100:.1f}%**")
            else:
                st.success(f"### ✅ Low Risk of Chronic Kidney Disease\nProbability: **{probability*100:.1f}%**")

        with col2:
            st.metric("Risk Probability", f"{probability*100:.1f}%")
            st.progress(float(probability))

        # ── SHAP Explanation ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Why this prediction? (SHAP)")

        try:
            from src.explainability.shap_explainer import get_shap_for_patient
            import shap

            rf_model = joblib.load(f"{MODELS_DIR}rf_kidney.joblib")
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

            rf_model = joblib.load(f"{MODELS_DIR}rf_kidney.joblib")

            # Use real training data for LIME (not random dummy data)
            X_train = load_training_data()
            if X_train is None:
                st.warning("Training data not found. Please re-run `python src/models/train_kidney.py` to save training data for LIME.")
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
                                  ["Serum Creatinine (sc)", "Hemoglobin (hemo)", "Blood Glucose (bgr)", "Blood Urea (bu)"],
                                  key="k_whatif_feat")

        idx_map = {
            "Serum Creatinine (sc)": 11,
            "Hemoglobin (hemo)": 14,
            "Blood Glucose (bgr)": 9,
            "Blood Urea (bu)": 10
        }
        range_map = {
            "Serum Creatinine (sc)": (0.4, 15.0, float(sc)),
            "Hemoglobin (hemo)": (3.0, 18.0, float(hemo)),
            "Blood Glucose (bgr)": (22.0, 490.0, float(bgr)),
            "Blood Urea (bu)": (1.0, 400.0, float(bu))
        }

        feat_idx = idx_map[whatif_col]
        rng = range_map[whatif_col]
        whatif_val = st.slider(f"What if {whatif_col} was...", rng[0], rng[1], rng[2], key="k_whatif_val")

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
