import matplotlib.pyplot as plt

models = ["Random Forest", "XGBoost"]

r2_scores = [
    0.9779,      # replace with your RF R²
    0.9779
]

plt.figure(figsize=(8,5))
plt.bar(models, r2_scores)

plt.title("Model Comparison (R² Score)")
plt.ylabel("R² Score")

plt.savefig(
    "reports/model_comparison.png"
)

# plt.show()

print("Chart saved successfully!")