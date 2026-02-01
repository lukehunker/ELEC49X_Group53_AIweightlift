import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------------
file_path = r"C:\School & Projects\Queens\Capstone\LGBM Regressor\weightlifting_dataset_v3.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

print(df.head())
print(df.shape)

# -----------------------------------------------------------
# 2. Add nonlinear interaction features
# -----------------------------------------------------------
df["bar_strain"] = df["bar_speed_change"] * df["facial_strain"]
df["bar_rom"] = df["bar_speed_change"] * df["rom_change"]
df["strain_rom"] = df["facial_strain"] * df["rom_change"]

# -----------------------------------------------------------
# 3. Feature / Target split
# -----------------------------------------------------------
feature_cols = [
    "bar_speed_change",
    "facial_strain",
    "rom_change",
    "bar_strain",
    "bar_rom",
    "strain_rom"
]

X = df[feature_cols]
y = df["RPE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 4. High-capacity LightGBM model (AVOIDS UNDERFITTING)
# -----------------------------------------------------------
model = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.015,
    max_depth=-1,
    num_leaves=120,
    min_child_samples=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------------------------------
# 5. Predict (NO CALIBRATION — avoids flattening)
# -----------------------------------------------------------
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# -----------------------------------------------------------
# 6. MAIN SCATTER PLOT (Predicted vs True)
# -----------------------------------------------------------
plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.85,
    color="blue",
    label="Predictions"
)

# Ideal diagonal (perfect model)
plt.plot([2, 10], [2, 10], color="yellow", linewidth=3, label="Ideal x=y")

# Regression line
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(
    y_test,
    m * y_test + b,
    color="red",
    linewidth=2,
    label="Model Regression Line"
)

plt.xlim(2, 10)
plt.ylim(2, 10)
plt.xlabel("True RPE", fontsize=12)
plt.ylabel("Predicted RPE", fontsize=12)
plt.title("RPE Prediction Performance (High-Capacity LGBM)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 7. RESIDUAL PLOT (Error vs Predicted)
# -----------------------------------------------------------
residuals = y_test - y_pred

plt.figure(figsize=(7, 4))
plt.scatter(y_pred, residuals, alpha=0.75, color="purple")
plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Predicted RPE", fontsize=12)
plt.ylabel("Residual (True - Predicted)", fontsize=12)
plt.title("Residual Plot (Error Analysis)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 8. ERROR DISTRIBUTION HISTOGRAM
# -----------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=10, color="steelblue", edgecolor="black")

plt.title("Distribution of Prediction Errors", fontsize=14)
plt.xlabel("Residual (Error)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 9. FEATURE IMPORTANCE PLOT
# -----------------------------------------------------------
importances = model.feature_importances_
importance_dict = dict(zip(feature_cols, importances))

print("\nFeature Importances:")
for feature, score in importance_dict.items():
    print(f"{feature}: {score}")

plt.figure(figsize=(6, 4))
plt.bar(importance_dict.keys(), importance_dict.values(), color="skyblue")
plt.title("Feature Importance", fontsize=14)
plt.xlabel("Feature", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()