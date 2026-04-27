"""
Train a Logistic Regression model on the Pima Indians Diabetes dataset
and export both the model and the scaler as pickle files.

Usage:
    python train_model.py

This will create:
    - backend/model.pkl  (the trained classifier + scaler bundled together)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ──────────────────────────────────────────────
# 1. Load the dataset
# ──────────────────────────────────────────────
df = pd.read_csv("diabetes.csv")
print(f"Dataset loaded — shape: {df.shape}")

# ──────────────────────────────────────────────
# 2. Separate features and target
# ──────────────────────────────────────────────
X = df.drop(columns="Outcome")
y = df["Outcome"]

# ──────────────────────────────────────────────
# 3. Standardize the features
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ──────────────────────────────────────────────
# 4. Train / Test split
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=2
)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ──────────────────────────────────────────────
# 5. Train a Logistic Regression model
#    (supports predict_proba for probability output)
# ──────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 6. Evaluate
# ──────────────────────────────────────────────
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"\nTraining Accuracy : {train_acc:.4f}")
print(f"Test Accuracy     : {test_acc:.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, model.predict(X_test)))

# ──────────────────────────────────────────────
# 7. Export the model and scaler as a single pkl
# ──────────────────────────────────────────────
os.makedirs("backend", exist_ok=True)

bundle = {"model": model, "scaler": scaler}
pkl_path = os.path.join("backend", "model.pkl")

with open(pkl_path, "wb") as f:
    pickle.dump(bundle, f)

print(f"\n[OK] Model + scaler saved to  {pkl_path}")
