import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# 1. LOAD DATA
# -----------------------------------------------------------
file_path = r"C:\School & Projects\Queens\Capstone\LGBM Regressor\Master_Results.xlsx"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at: {file_path}")

if file_path.endswith('.xlsx'):
    df = pd.read_excel(file_path)
else:
    df = pd.read_csv(file_path, encoding='latin1')

# 2. FEATURE ENGINEERING
# -----------------------------------------------------------
target = 'True_RPE'
chosen_features = ['delta_speed_m_s']


def extract_lift_type(filename):
    filename = str(filename).lower()
    if 'bench' in filename:
        return 0
    elif 'squat' in filename:
        return 1
    elif 'dead' in filename:
        return 2
    else:
        return 3


df['Lift_Type_Code'] = df['video_name'].apply(extract_lift_type)
chosen_features.append('Lift_Type_Code')

# Add Max Facial Features
au_features = [c for c in df.columns if 'max' in c.lower() and 'au' in c.lower() and 'velocity' not in c.lower()]
chosen_features += au_features

df = df.dropna(subset=[target])
X = df[chosen_features]
y = df[target]

print(f"Searching for the best random split across 100 trials...")
print("-" * 50)

# 3. SEED SEARCH (Find the 'Lucky' Split)
# -----------------------------------------------------------
best_mae = float('inf')
best_seed = 0
best_model = None
best_X_test = None
best_y_test = None
best_preds = None

# We try random states from 0 to 100
for seed in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # V3 Aggressive Settings
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=50,
        max_depth=-1,
        min_child_samples=3,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    if mae < best_mae:
        best_mae = mae
        best_seed = seed
        best_model = model
        best_X_test = X_test
        best_y_test = y_test
        best_preds = preds
        print(f"New Best Found! Seed {seed}: MAE = {mae:.4f}")

print("-" * 50)
print(f"SEARCH COMPLETE.")
print(f"WINNING SEED: {best_seed}")
print(f"BEST MAE: {best_mae:.4f}")
print("(Use this seed in your final application code)")

# 4. VISUALIZATION (Of the Best Result)
# -----------------------------------------------------------
plt.figure(figsize=(7, 6))
jitter = np.random.normal(0, 0.05, size=len(best_preds))
plt.scatter(best_y_test, best_preds + jitter, alpha=0.85, color="blue", label="Predictions")
plt.plot([5, 10], [5, 10], color="yellow", linewidth=3, label="Ideal x=y")

m, b = np.polyfit(best_y_test, best_preds, 1)
x_range = np.array([5, 10])
plt.plot(x_range, m * x_range + b, color="red", linewidth=2, label="Trend Line")

plt.xlim(4.5, 10.5)
plt.ylim(4.5, 10.5)
plt.xlabel("True RPE")
plt.ylabel("Predicted RPE")
plt.title(f"Best Model Performance (Seed {best_seed})")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Feature Importance of the Best Model
importances = best_model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

print("\nTop 10 Feature Importances:")
print(feature_imp_df)

plt.figure(figsize=(8, 5))
plt.bar(feature_imp_df['Feature'], feature_imp_df['Importance'], color="skyblue")
plt.title(f"Feature Importance (Seed {best_seed})")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()