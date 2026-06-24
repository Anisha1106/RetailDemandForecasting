import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/xgboost_model.pkl")

features = [
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

importance = model.feature_importances_

plt.figure(figsize=(10,5))
plt.bar(features, importance)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("reports/feature_importance.png")

print("Feature importance chart saved!")