# Q4 - Naive Bayes

import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load Dataset
cancer = load_breast_cancer(as_frame=True)
df = cancer.frame

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy :", round(accuracy_score(y_test, y_pred),4))
print("Precision:", round(precision_score(y_test, y_pred),4))
print("Recall   :", round(recall_score(y_test, y_pred),4))
print("F1 Score :", round(f1_score(y_test, y_pred),4))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save Model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/naive_bayes_model.pkl")
joblib.dump(scaler, "models/naive_scaler.pkl")

print("\nModel Saved Successfully!")