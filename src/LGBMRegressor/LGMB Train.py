import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------------
file_path = r"C:\Users\ajmal\OneDrive\Desktop\Queen's\Capstone\LGBM Regressor\weightlifting_dataset_v3.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

print(df.head())
print(df.shape)  # should show (200, 4)

# -----------------------------------------------------------
# 2. Split features and target
# -----------------------------------------------------------
X = df[["bar_speed_change", "facial_strain", "rom_change"]]
y = df["RPE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 3. Normalize features (0–10 scale), then apply weights
# -----------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 10))
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Apply relative feature importance weights
weights = {
    "bar_speed_change": 0.60,  # dominant
    "facial_strain": 0.25,
    "rom_change": 0.15
}
for col, w in weights.items():
    X_train_scaled.loc[:, col] *= w
    X_test_scaled.loc[:, col] *= w

# -----------------------------------------------------------
# 4. Train LGBM Regressor
# -----------------------------------------------------------
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -----------------------------------------------------------
# 5. Evaluate model
# -----------------------------------------------------------
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")

# -----------------------------------------------------------
# 6. Visualization
# -----------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.8, color="blue", label="Predictions")

# Yellow diagonal points (3,3 ... 10,10)
rpe_line = np.arange(3, 10.1, 1)
plt.plot(rpe_line, rpe_line, "o", color="yellow", markersize=8, label="Ideal (x=y)")

plt.xlim(2, 10)
plt.ylim(2, 10)
plt.xlabel("True RPE")
plt.ylabel("Predicted RPE")
plt.title("RPE Prediction Performance (LGBM)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 7. Feature importances
# -----------------------------------------------------------
importances = model.feature_importances_
importance_dict = dict(zip(X.columns, importances))
print("\nFeature Importances (after scaling):")
for feature, score in importance_dict.items():
    print(f"{feature}: {score}")

# Optional plot
plt.figure(figsize=(5, 4))
plt.bar(importance_dict.keys(), importance_dict.values(), color="skyblue")
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()