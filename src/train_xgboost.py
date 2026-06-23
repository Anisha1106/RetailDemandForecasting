import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor
import joblib

# Load data
df = pd.read_csv("data/processed/features.csv")

X = df[
    [
        "Store",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Year",
        "Month",
        "Day",
        "Week"
    ]
]

y = df["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nXGBoost Performance")
print("-" * 30)
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"R2  : {r2:.4f}")

joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

print("\nXGBoost model saved!")