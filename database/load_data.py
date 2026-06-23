import pandas as pd
from sqlalchemy import create_engine

# Read dataset
df = pd.read_csv("data/raw/walmart.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Rename columns to match PostgreSQL table
df = df.rename(columns={
    "Store": "store",
    "Date": "sale_date",
    "Weekly_Sales": "weekly_sales",
    "Holiday_Flag": "holiday_flag",
    "Temperature": "temperature",
    "Fuel_Price": "fuel_price",
    "CPI": "cpi",
    "Unemployment": "unemployment"
})

# PostgreSQL connection
engine = create_engine(
    "postgresql://anisha@localhost/retaildb"
)

# Upload data
df.to_sql(
    "walmart_sales",
    engine,
    if_exists="append",
    index=False
)

print(f"{len(df)} rows uploaded successfully!")