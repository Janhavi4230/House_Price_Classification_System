# Import required libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load California Housing Dataset
housing = fetch_california_housing(as_frame=True)

# Convert to DataFrame
df = housing.frame

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset information
print("\nDataset Info:")
print(df.info())

# Display column names
print("\nColumns:")
print(df.columns)
# -----------------------------
# Data Preprocessing
# -----------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Features (X) and Target (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)
# -----------------------------
# Train-Test Split
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# R2 Score
score = r2_score(y_test, y_pred)

print("\n==============================")
print("Linear Regression Results")
print("==============================")
print("R2 Score :", round(score,4))

print("\nFirst 10 Predictions")
for i in range(10):
    print(f"Actual: {y_test.iloc[i]:.3f} | Predicted: {y_pred[i]:.3f}")

# Save Model
joblib.dump(model, "models/linear_regression_model.pkl")

print("\nModel Saved Successfully!")