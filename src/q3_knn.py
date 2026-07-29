# Q3 - KNN

import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

best_accuracy = 0
best_k = 0
best_model = None

print("KNN Accuracy Comparison")
print("-" * 30)

for k in [3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"k = {k} --> Accuracy = {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
        best_model = model

print("\nBest K =", best_k)
print("Best Accuracy =", round(best_accuracy,4))

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/knn_model.pkl")
joblib.dump(scaler, "models/knn_scaler.pkl")

print("\nModel Saved Successfully!")