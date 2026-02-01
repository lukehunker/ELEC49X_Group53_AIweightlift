import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
# 2. Train/test split
# -----------------------------------------------------------
X = df[["bar_speed_change", "facial_strain", "rom_change"]]
y = df["RPE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 3. LightGBM (your preferred hyperparameters)
# -----------------------------------------------------------
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=50,
    min_child_samples=3,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------------------------------
# 4. Predict + Calibration
# -----------------------------------------------------------
y_pred = model.predict(X_test)

cal = LinearRegression()
cal.fit(y_pred.reshape(-1, 1), y_test)
y_pred_calibrated = cal.predict(y_pred.reshape(-1, 1))

mse = mean_squared_error(y_test, y_pred_calibrated)
r2 = r2_score(y_test, y_pred_calibrated)
print(f"\nCalibrated MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# -----------------------------------------------------------
# 5. Main Scatter Plot + Regression Line + Ideal Line
# -----------------------------------------------------------
plt.figure(figsize=(7, 6))

# jitter to avoid stacked dots
jitter = np.random.normal(0, 0.06, size=len(y_pred_calibrated))

plt.scatter(
    y_test,
    y_pred_calibrated + jitter,
    alpha=0.85,
    color="blue",
    label="Predictions (calibrated)"
)

# Perfect x=y line
plt.plot([2, 10], [2, 10], color="yellow", linewidth=3, label="Ideal x=y")

# Model regression line
m, b = np.polyfit(y_test, y_pred_calibrated, 1)
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
plt.title("RPE Prediction Performance (LGBM + Calibration)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 6. Residual Plot (Error vs Prediction)
# -----------------------------------------------------------
residuals = y_test - y_pred_calibrated

plt.figure(figsize=(7, 4))
plt.scatter(y_pred_calibrated, residuals, alpha=0.75, color="purple")
plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Predicted RPE", fontsize=12)
plt.ylabel("Residual (True - Predicted)", fontsize=12)
plt.title("Residual Plot (Error Analysis)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 7. Distribution of Prediction Errors
# -----------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=10, color="steelblue", edgecolor="black")

plt.title("Distribution of Prediction Errors", fontsize=14)
plt.xlabel("Residual Error", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 8. Feature Importances
# -----------------------------------------------------------
importances = model.feature_importances_
importance_dict = dict(zip(X.columns, importances))

print("\nFeature Importances:")
for feature, score in importance_dict.items():
    print(f"{feature}: {score}")

plt.figure(figsize=(6, 4))
plt.bar(importance_dict.keys(), importance_dict.values(), color="skyblue")
plt.title("Feature Importance", fontsize=14)
plt.xlabel("Feature", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)
plt.tight_layout()
plt.show()