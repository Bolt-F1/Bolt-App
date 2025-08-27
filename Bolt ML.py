import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ========================
# Example dataset
# ========================
# Replace this with your real data
# Columns = factors, Target = lap_time (or energy loss)
data = pd.DataFrame({
    "drag": [0.4],
    "downforce": [0.65],
    "weight": [0.065],
    "friction": [0.05],
    "lap_time": [1.85, 1.92, 1.88, 2.05, 1.99]  # measured result
})

# Features (inputs) and target (output)
X = data[["drag", "downforce", "weight", "friction"]]
y = data["lap_time"]

# ========================
# Train Random Forest
# ========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Feature importances (how much each factor contributes)
importances = model.feature_importances_
factors = X.columns

# ========================
# Plot with Matplotlib
# ========================
plt.figure(figsize=(8, 5))
plt.bar(factors, importances, color="skyblue")
plt.xlabel("Factors Contributing to Lap Time / Energy Loss")
plt.ylabel("Importance (relative influence)")
plt.title("What to Prioritize in Car Design")
plt.show()
