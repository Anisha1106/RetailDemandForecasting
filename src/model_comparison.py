import pandas as pd

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost"
    ],
    "R2": [
        0.9779,  # replace with RF score
        0.9779  # replace with XGB score
    ]
})

print(comparison)