import pandas as pd

# Load data
df = pd.read_csv("data/raw/walmart.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Extract date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Week"] = df["Date"].dt.isocalendar().week

# Show result
print(df.head())

# Save processed data
df.to_csv(
    "data/processed/features.csv",
    index=False
)

print("\nFeature engineering completed successfully!")