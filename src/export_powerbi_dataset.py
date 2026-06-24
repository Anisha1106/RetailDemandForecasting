import numpy as np
import pandas as pd


INPUT_PATH = "data/processed/predictions.csv"
OUTPUT_PATH = "powerbi/retail_demand_powerbi_dataset.csv"


def build_powerbi_dataset(input_path=INPUT_PATH):
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Sales_Gap"] = df["Predicted_Sales"] - df["Weekly_Sales"]
    df["Sales_Gap_Pct"] = np.where(
        df["Weekly_Sales"] != 0,
        df["Sales_Gap"] / df["Weekly_Sales"],
        np.nan,
    )
    df["Absolute_Error"] = df["Sales_Gap"].abs()
    df["Absolute_Percentage_Error"] = np.where(
        df["Weekly_Sales"] != 0,
        df["Absolute_Error"] / df["Weekly_Sales"],
        np.nan,
    )
    df["Holiday_Label"] = np.where(df["Holiday_Flag"] == 1, "Holiday", "Regular")

    return df[
        [
            "Store",
            "Date",
            "Year",
            "Month",
            "Week",
            "Day",
            "Holiday_Flag",
            "Holiday_Label",
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
            "Weekly_Sales",
            "Predicted_Sales",
            "Sales_Gap",
            "Sales_Gap_Pct",
            "Absolute_Error",
            "Absolute_Percentage_Error",
        ]
    ]


if __name__ == "__main__":
    powerbi_df = build_powerbi_dataset()
    powerbi_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Power BI dataset exported to {OUTPUT_PATH}")
    print(f"Rows: {len(powerbi_df):,}")
