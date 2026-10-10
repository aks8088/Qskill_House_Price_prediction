
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("Loading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target * 100000

print("\nDataset shape:", X.shape)
print("\nFeatures:")
print(X.head())

print("\nMissing values:")
print(X.isnull().sum())

X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print(f"MAE: ${mae:,.2f}")
print(f"MSE: {mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
    "Value": [mae, mse, rmse, r2]
})

metrics.to_csv(RESULTS_DIR / "evaluation_metrics.csv", index=False)

comparison = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": y_pred
})

comparison.to_csv(
    RESULTS_DIR / "actual_vs_predicted.csv",
    index=False
)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual House Value ($)")
plt.ylabel("Predicted House Value ($)")
plt.title("Actual vs Predicted House Values")

limits = [
    min(y_test.min(), y_pred.min()),
    max(y_test.max(), y_pred.max())
]

plt.plot(limits, limits, linestyle="--")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "actual_vs_predicted.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test - y_pred, alpha=0.4)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted House Value ($)")
plt.ylabel("Prediction Error ($)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "residual_plot.png")
plt.close()

joblib.dump(model, MODELS_DIR / "house_price_model.joblib")

sample = X_test.iloc[[0]]
sample_prediction = model.predict(sample)[0]

print("\nSample House Features:")
print(sample.to_string(index=False))
print(f"\nPredicted House Value: ${sample_prediction:,.2f}")
print("\nProject completed successfully.")
print("Metrics and plots saved in the results folder.")
print("Trained model saved in the models folder.")
