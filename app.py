"""
Heart Disease Risk Checker
Production-grade Streamlit app with ML evaluation
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ================== CONFIG ==================
NUM_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CAT_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Heart Disease Risk Checker",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== CSS ==================
st.markdown(
    """
<style>
/* Headings */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    color: #ef4444;
}
.sub-header {
    text-align: center;
    color: #9ca3af;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Landing card */
.hero-box {
    background: #0f172a;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 1.8rem;
    color: #e5e7eb;
}
.hero-box h4 {
    color: #f9fafb;
}
.hero-box ul li {
    color: #d1d5db;
}
.hero-box p {
    color: #d1d5db;
}

/* Disclaimer */
.disclaimer {
    background-color: #020617;
    color: #e5e7eb;
    padding: 1.2rem;
    border-left: 5px solid #f59e0b;
    margin: 1.5rem 0;
}

/* Risk cards */
.risk-box {
    padding: 2rem;
    border-radius: 14px;
    text-align: center;
    margin-top: 1.5rem;
}
.risk-low {
    background-color: #052e16;
    border: 2px solid #22c55e;
    color: #dcfce7;
}
.risk-medium {
    background-color: #451a03;
    border: 2px solid #f59e0b;
    color: #fffbeb;
}
.risk-high {
    background-color: #450a0a;
    border: 2px solid #ef4444;
    color: #fee2e2;
}
.risk-box h1, .risk-box h2, .risk-box p {
    color: inherit !important;
}

/* Section titles */
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 2rem;
    color: #f9fafb;
}
</style>
""",
    unsafe_allow_html=True,
)

# ================== LOAD PIPELINE ==================
@st.cache_resource
def load_pipeline():
    path = BASE_DIR / "heart_disease_pipeline.pkl"
    if not path.exists():
        st.error(
            "❌ `heart_disease_pipeline.pkl` not found.\n\n"
            "Make sure the file is in the same folder as `app.py` "
            "and was created by running `train_pipeline.py`."
        )
        st.stop()
    return joblib.load(path)


# ================== UTIL ==================
def risk_bucket(p: float):
    if p < 0.3:
        return "Low", "🟢", "risk-low"
    elif p < 0.6:
        return "Medium", "🟡", "risk-medium"
    else:
        return "High", "🔴", "risk-high"


def safe_name(name: str) -> str:
    return name.replace(" ", "_")


# ================== PAGES ==================
def landing_page():
    st.markdown(
        '<p class="main-header">❤️ Heart Disease Risk Checker</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Estimate your heart disease risk using AI in under 2 minutes</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
        <div class="hero-box">
            <h4>✅ What this tool does</h4>
            <ul>
                <li>Uses machine learning trained on real clinical data</li>
                <li>Evaluates multiple ML models (LR, RF, SVM, GB, KNN, XGBoost)</li>
                <li>Provides technical evaluation with ROC curves & confusion matrices</li>
            </ul>

            <h4>⚠️ Important</h4>
            <p>
            This tool is for <b>educational and research purposes only</b>.
            It does NOT replace professional medical advice.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Start Risk Assessment", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()


def assessment_page():
    st.markdown(
        '<p class="main-header">🩺 Health Assessment</p>', unsafe_allow_html=True
    )

    st.markdown(
        """
    <div class="disclaimer">
    <strong>Medical Disclaimer:</strong><br>
    This AI model provides statistical risk estimates only and is not a medical diagnosis.
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.form("health_form"):
        age = st.slider("Age", 20, 100, 50)
        sex = st.radio("Sex", ["Male", "Female"])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
        thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 0.0)
        cp = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"],
        )
        exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
        slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"]
        )
        fbs = st.radio("Fasting Blood Sugar > 120", ["No", "Yes"])
        restecg = st.selectbox(
            "Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
        )

        submitted = st.form_submit_button("🔍 Calculate Risk")

    if submitted:
        st.session_state.input = {
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "trestbps": trestbps,
            "chol": chol,
            "thalach": thalach,
            "oldpeak": oldpeak,
            "cp": [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal",
                "Asymptomatic",
            ].index(cp),
            "exang": 1 if exang == "Yes" else 0,
            "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
            "ca": ca,
            "thal": ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1,
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(
                restecg
            ),
        }
        st.session_state.page = "result"
        st.rerun()


def result_page():
    if "input" not in st.session_state:
        st.warning("Please complete an assessment first.")
        st.session_state.page = "form"
        st.rerun()

    pipeline = load_pipeline()
    df = pd.DataFrame([st.session_state.input])

    try:
        prob = pipeline.predict_proba(df)[0, 1]
    except Exception as e:
        st.error("Prediction failed. Check that the pipeline was trained with the "
                 "same feature set as the form.")
        st.exception(e)
        return

    risk, icon, css = risk_bucket(prob)

    st.markdown(
        f"""
    <div class="risk-box {css}">
        <h1>{icon} {risk} Risk</h1>
        <h2>{prob*100:.1f}% likelihood</h2>
        <p>Based on your health profile</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 New Assessment", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()
    with col2:
        if st.button("📊 ML Model Evaluation", use_container_width=True):
            st.session_state.page = "ml"
            st.rerun()


def ml_page():
    st.markdown(
        '<p class="main-header">📊 Model Evaluation (ML Scientist View)</p>',
        unsafe_allow_html=True,
    )

    comparison_path = BASE_DIR / "data" / "model_comparison.csv"
    roc_path = BASE_DIR / "data" / "roc_curves.joblib"

    if not comparison_path.exists() or not roc_path.exists():
        st.error(
            "Model evaluation artifacts not found.\n\n"
            "Run `evaluate_models.py` locally and commit the generated files inside "
            "`data/` (model_comparison.csv, roc_curves.joblib, cm_*.joblib)."
        )
        return

    df = pd.read_csv(comparison_path)
    st.dataframe(df.style.highlight_max(axis=0))

    best = df.loc[df["ROC_AUC"].idxmax()]
    st.success(
        f"🏆 Best Model: {best['Model']} (ROC-AUC = {best['ROC_AUC']:.3f})"
    )

    st.markdown(
        '<p class="section-title">ROC Curve Comparison</p>', unsafe_allow_html=True
    )
    roc_data = joblib.load(roc_path)

    fig, ax = plt.subplots()
    for model, (fpr, tpr, auc_val) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{model} (AUC={auc_val:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        '<p class="section-title">Confusion Matrices</p>', unsafe_allow_html=True
    )
    cols = st.columns(len(df))
    for i, name in enumerate(df["Model"]):
        with cols[i]:
            cm_path = BASE_DIR / "data" / f"cm_{safe_name(name)}.joblib"
            if not cm_path.exists():
                st.warning(f"No confusion matrix found for {name}")
                continue
            cm = joblib.load(cm_path)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(name, fontsize=10)
            st.pyplot(fig)


# ================== ROUTER ==================
def main():
    if "page" not in st.session_state:
        st.session_state.page = "landing"

    pages = {
        "landing": landing_page,
        "form": assessment_page,
        "result": result_page,
        "ml": ml_page,
    }

    pages[st.session_state.page]()


if __name__ == "__main__":
    main()
