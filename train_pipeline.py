# train_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# ======================
# 1. Load dataset
# ======================
df = pd.read_csv("data/dataset.csv")  # ✅ path matches your structure

# ======================
# 2. Feature lists
# ======================
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

target = 'target'   # change ONLY if your column name is different

X = df[num_features + cat_features]
y = df[target]

# ======================
# 3. Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 4. Preprocessing
# ======================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

# ======================
# 5. Model
# ======================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ('preprocessing', preprocessor),
        ('model', model)
    ]
)

# ======================
# 6. Train
# ======================
pipeline.fit(X_train, y_train)

# ======================
# 7. Evaluate (sanity check)
# ======================
proba = pipeline.predict_proba(X_test)[:, 1]
pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC :", roc_auc_score(y_test, proba))

# ======================
# 8. Save pipeline
# ======================
joblib.dump(pipeline, "heart_disease_pipeline.pkl")

print("✅ Saved: heart_disease_pipeline.pkl")
