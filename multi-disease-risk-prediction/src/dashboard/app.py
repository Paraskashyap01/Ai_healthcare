"""
Streamlit Dashboard — Main Entry Point
Member 5's responsibility

Run with: streamlit run src/dashboard/app.py
(from the multi-disease-risk-prediction directory)
"""

import streamlit as st
import sys
import os

# Add project root to path so imports work
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(project_root))

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Disease Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.title("🏥 Disease Risk Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Disease",
    ["🏠 Home", "🩸 Diabetes", "❤️ Heart Disease", "🫘 Kidney Disease", "📊 Model Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack:**")
st.sidebar.markdown("- Scikit-learn, XGBoost")
st.sidebar.markdown("- SHAP + LIME (XAI)")
st.sidebar.markdown("- Streamlit")
st.sidebar.markdown("---")
st.sidebar.markdown("*AIML Subject Project*")

# ── Page Routing ──────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏥 Explainable Multi-Disease Risk Prediction System")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 🩸 Diabetes\nPredicts diabetes risk using glucose, BMI, age and other health markers.")
    with col2:
        st.warning("### ❤️ Heart Disease\nPredicts heart disease risk using cholesterol, blood pressure, and ECG data.")
    with col3:
        st.error("### 🫘 Kidney Disease\nPredicts chronic kidney disease using blood and urine test results.")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. **Select a disease** from the sidebar
    2. **Enter patient health data** using the input fields
    3. **Get a risk prediction** with probability score
    4. **Understand the prediction** through SHAP and LIME explanations
    5. **Try What-If analysis** to see how changing one value affects risk
    """)

    st.markdown("---")
    st.markdown("### Models Used")
    st.markdown("""
    Each disease uses an ensemble of:
    - **Logistic Regression** — baseline
    - **Random Forest** — primary model
    - **XGBoost** — boosted model
    - **Voting Ensemble** — combines all 3 (best performer)
    """)

    st.markdown("---")
    st.markdown("### Explainability")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)**
        - Shows global and per-patient feature importance
        - Based on game-theoretic Shapley values
        - Red = increases risk, Blue = decreases risk
        """)
    with col2:
        st.markdown("""
        **LIME (Local Interpretable Model-agnostic Explanations)**
        - Explains individual predictions locally
        - Fits a simple model around each data point
        - Shows which features drive that specific prediction
        """)

elif page == "🩸 Diabetes":
    from src.dashboard.views.diabetes_page import show
    show()

elif page == "❤️ Heart Disease":
    from src.dashboard.views.heart_page import show
    show()

elif page == "🫘 Kidney Disease":
    from src.dashboard.views.kidney_page import show
    show()

elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison Across All Diseases")
    st.markdown("---")

    import pandas as pd

    # Load saved metrics CSVs
    all_metrics = []
    for disease in ["diabetes", "heart", "kidney"]:
        path = f"results/metrics/metrics_{disease}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_metrics.append(df)

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)

        # Overall table
        st.subheader("All Model Results")
        st.dataframe(
            combined.style.highlight_max(
                subset=["AUC-ROC", "F1 Score", "Accuracy"],
                color="lightgreen"
            ),
            use_container_width=True
        )

        # Best per disease
        st.markdown("---")
        st.subheader("🏆 Best Model Per Disease (by AUC-ROC)")
        for disease in ["diabetes", "heart", "kidney"]:
            disease_df = combined[combined["Disease"] == disease]
            if not disease_df.empty:
                best = disease_df.loc[disease_df["AUC-ROC"].idxmax()]
                st.markdown(f"**{disease.capitalize()}:** {best['Model']} — AUC-ROC: {best['AUC-ROC']:.4f}, F1: {best['F1 Score']:.4f}")

        # Comparison charts
        st.markdown("---")
        st.subheader("📈 AUC-ROC Comparison")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        diseases = combined["Disease"].unique()
        models = combined["Model"].unique()
        x = range(len(models))
        width = 0.25

        for i, disease in enumerate(diseases):
            disease_data = combined[combined["Disease"] == disease]
            values = [disease_data[disease_data["Model"] == m]["AUC-ROC"].values[0]
                      if m in disease_data["Model"].values else 0
                      for m in models]
            ax.bar([xi + i * width for xi in x], values, width, label=disease.capitalize())

        ax.set_xlabel("Model")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("AUC-ROC Score by Model and Disease")
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels(models, rotation=15)
        ax.legend()
        ax.set_ylim(0.7, 1.05)
        plt.tight_layout()
        st.pyplot(fig)

        # Show saved evaluation plots
        st.markdown("---")
        st.subheader("📊 Evaluation Plots")
        plot_type = st.selectbox("Select Plot Type", ["Confusion Matrix", "ROC Curve", "SHAP Summary", "SHAP Bar"])
        disease_select = st.selectbox("Select Disease", ["diabetes", "heart", "kidney"])

        prefix_map = {
            "Confusion Matrix": "cm",
            "ROC Curve": "roc",
            "SHAP Summary": "shap_summary",
            "SHAP Bar": "shap_bar"
        }
        prefix = prefix_map[plot_type]

        if plot_type in ["SHAP Summary", "SHAP Bar"]:
            img_path = f"results/figures/{prefix}_{disease_select}_rf.png"
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{plot_type} — {disease_select.capitalize()}")
            else:
                st.warning(f"Plot not found: {img_path}")
        else:
            cols = st.columns(2)
            models_to_show = ["logistic_regression", "random_forest", "xgboost", "voting_ensemble"]
            for i, model_key in enumerate(models_to_show):
                img_path = f"results/figures/{prefix}_{disease_select}_{model_key}.png"
                if os.path.exists(img_path):
                    with cols[i % 2]:
                        st.image(img_path, caption=f"{model_key.replace('_', ' ').title()}")
    else:
        st.warning("No metrics found. Train models first by running the training scripts.")
