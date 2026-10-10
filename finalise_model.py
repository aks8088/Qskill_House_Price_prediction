
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target * 100000

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_name = None
best_rmse = float("inf")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name

comparison = pd.DataFrame(results).sort_values("RMSE")
comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

joblib.dump(best_model, MODELS_DIR / "best_house_price_model.joblib")

best_predictions = best_model.predict(X_test)

pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": best_predictions,
    "Error": y_test.to_numpy() - best_predictions
}).to_csv(RESULTS_DIR / "best_model_predictions.csv", index=False)

print("\nModel Comparison")
print(comparison.to_string(index=False))

print(f"\nSelected model: {best_name}")
print(f"Test MAE: ${mean_absolute_error(y_test, best_predictions):,.2f}")
print(f"Test RMSE: ${best_rmse:,.2f}")
print(f"Test R2 Score: {r2_score(y_test, best_predictions):.4f}")
print("\nSaved models/best_house_price_model.joblib")
