
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler

# ===============================
# Load Models
# ===============================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

models = {}
if os.path.exists(MODELS_DIR):
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".joblib"):
            model_name = filename.replace(".joblib", "")
            models[model_name] = joblib.load(os.path.join(MODELS_DIR, filename))
else:
    st.error("⚠️ No models folder found at ../models. Please check your paths.")

# ===============================
# Preprocessing Function
# ===============================
def preprocess_input(df):
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # Convert categorical columns to string
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Scale numerical features
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

    return df_encoded

# ===============================
# Streamlit UI
# ===============================
st.title("❤️ Heart Disease Prediction App")

st.sidebar.header("Enter Patient Details")

# Collect user inputs
age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-3) colored by fluoroscopy", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thal", [0, 1, 2, 3])

# Convert to DataFrame
features = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}])

# Preprocess input
features_processed = preprocess_input(features)

# ===============================
# Prediction Section
# ===============================
if st.button("Predict"):
    if not models:
        st.error("No models available!")
    else:
        for name, model in models.items():
            # Align feature columns (fill missing ones with 0)
            X_input = features_processed.reindex(columns=model.feature_names_in_, fill_value=0)

            # Predict
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0, 1] if hasattr(model, "predict_proba") else None

            st.subheader(f"📌 Model: {name}")
            st.write("✅ Prediction:", "Heart Disease" if pred == 1 else "No Heart Disease")
            if proba is not None:
                st.write(f"🔢 Probability of Heart Disease: {proba:.2f}")

# ===============================
# Model Evaluation Section
# ===============================
st.header("📊 Model Evaluation & Comparison")

uploaded_file = st.file_uploader("Upload test dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)

    # Split features/target
    X_test_raw = df_test.drop("target", axis=1)
    y_test = df_test["target"]

    # Preprocess test data
    X_test_processed = preprocess_input(X_test_raw)

    # Store results for comparison
    results = []

    for name, model in models.items():
        st.subheader(f"Model: {name}")

        X_eval = X_test_processed.reindex(columns=model.feature_names_in_, fill_value=0)
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": roc_auc if roc_auc else 0
        })

        st.write(f"Accuracy: {acc:.2f}")
        st.write(f"Precision: {prec:.2f}")
        st.write(f"Recall: {rec:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
        if roc_auc:
            st.write(f"ROC AUC: {roc_auc:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           title=f"Confusion Matrix - {name}")
        st.plotly_chart(cm_fig)

        # ROC Curve
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_fig = px.area(
                x=fpr, y=tpr,
                title=f"ROC Curve - {name}",
                labels=dict(x="False Positive Rate", y="True Positive Rate"),
                width=500, height=400
            )
            roc_fig.add_shape(
                type="line", line=dict(dash="dash"),
                x0=0, x1=1, y0=0, y1=1
            )
            st.plotly_chart(roc_fig)

    # ===============================
    # Model Comparison Charts
    # ===============================
    results_df = pd.DataFrame(results)
    st.subheader("📊 Model Comparison Overview")

    # Bar chart for Accuracy
    acc_fig = px.bar(results_df, x="Model", y="Accuracy", color="Model",
                     title="Accuracy Comparison of Models", text_auto=True)
    st.plotly_chart(acc_fig)

    # Multi-metric chart
    metrics_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    metrics_fig = px.bar(metrics_df, x="Model", y="Score", color="Metric",
                         barmode="group", title="Comparison of All Metrics")
    st.plotly_chart(metrics_fig)

    # Best model suggestion
    best_model = results_df.loc[results_df["Accuracy"].idxmax()]
    st.success(f"🏆 Best Model Based on Accuracy: **{best_model['Model']}** "
               f"(Accuracy: {best_model['Accuracy']:.2f})")

