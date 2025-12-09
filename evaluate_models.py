import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

# ================= LOAD DATA =================
df = pd.read_csv("data/dataset.csv")

num = ['age','trestbps','chol','thalach','oldpeak']
cat = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

X = df[num + cat]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= PREPROCESSOR =================
prep = ColumnTransformer([
    ('num', StandardScaler(), num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
])

# ================= SKLEARN MODELS =================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1),
    "SVM (Linear)": SVC(kernel="linear", probability=True),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

results = []
roc_curves = {}

# ========== Evaluate sklearn models ==========
for name, model in models.items():
    print(f"Training: {name}")

    pipe = Pipeline([
        ('prep', prep),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    roc_curves[name] = (fpr, tpr, roc_auc)

    cm = confusion_matrix(y_test, y_pred)
    joblib.dump(cm, f"data/cm_{name.replace(' ', '_')}.joblib")

    results.append([name, acc, prec, rec, roc_auc])

# ========== XGBOOST (OUTSIDE PIPELINE) ==========
print("Training: XGBoost")

# preprocess manually
X_train_xgb = prep.fit_transform(X_train)
X_test_xgb = prep.transform(X_test)

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train_xgb, y_train)

y_pred = xgb_model.predict(X_test_xgb)
y_prob = xgb_model.predict_proba(X_test_xgb)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

roc_curves["XGBoost"] = (fpr, tpr, roc_auc)

cm = confusion_matrix(y_test, y_pred)
joblib.dump(cm, "data/cm_XGBoost.joblib")

results.append(["XGBoost", acc, prec, rec, roc_auc])

# ================= SAVE RESULTS =================
pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "ROC_AUC"]
).to_csv("data/model_comparison.csv", index=False)

joblib.dump(roc_curves, "data/roc_curves.joblib")

print("✅ Model evaluation completed (XGBoost fixed & included)")
