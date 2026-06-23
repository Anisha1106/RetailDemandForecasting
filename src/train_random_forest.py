import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib

# Load engineered data
df = pd.read_csv("data/processed/features.csv")

# Features
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

# Target
y = df["Weekly_Sales"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-" * 30)
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"R2  : {r2:.4f}")

# Save model
joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("\nModel saved successfully!")