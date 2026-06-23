import pandas as pd
import joblib

# Load data
df = pd.read_csv("data/processed/features.csv")

# Load trained model
model = joblib.load("models/xgboost_model.pkl")

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

# Generate predictions
df["Predicted_Sales"] = model.predict(X)

# Save predictions
df.to_csv(
    "data/processed/predictions.csv",
    index=False
)

print("Predictions file created successfully!")
print(df.head())