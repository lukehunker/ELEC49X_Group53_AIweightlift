import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------
# 1. LOAD DATA (Robust Load)
# -----------------------------------------------------------
file_path = r"C:\School & Projects\Queens\Capstone\LGBM Regressor\Master_Results.xlsx"

print(f"Attempting to load: {file_path}")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at: {file_path}")

# Detect extension and use the correct loader
if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
    print("Detected Excel file. Using read_excel...")
    df = pd.read_excel(file_path)
else:
    print("Detected CSV. Using read_csv...")
    df = pd.read_csv(file_path, encoding='latin1')

print(f"Successfully loaded {len(df)} rows.")

# -----------------------------------------------------------
# 2. FEATURE SELECTION
# -----------------------------------------------------------
target = 'True_RPE'

# A. Start with Bar Speed (The one you want prioritized)
chosen_features = ['delta_speed_m_s']

# B. Add Context (Lift Type) - Helpful for the model to distinguish Bench vs Squat
def extract_lift_type(filename):
    filename = str(filename).lower()
    if 'bench' in filename: return 0  # Bench
    elif 'squat' in filename: return 1 # Squat
    elif 'dead' in filename: return 2  # Deadlift
    else: return 3
df['Lift_Type_Code'] = df['video_name'].apply(extract_lift_type)
chosen_features.append('Lift_Type_Code')

# C. Add Facial Features (All columns ending in "_max" that are Action Units)
# We exclude 'velocity' or 'movement' columns to keep it clean like V3
au_features = [c for c in df.columns if 'max' in c.lower() and 'au' in c.lower() and 'velocity' not in c.lower()]
chosen_features += au_features

print("\n" + "="*50)
print(f"USING THESE {len(chosen_features)} COLUMNS FOR TRAINING:")
print("="*50)
for feature in chosen_features:
    print(f"- {feature}")
print("="*50 + "\n")

# D. Clean Data
df = df.dropna(subset=[target])
X = df[chosen_features]
y = df[target]

# -----------------------------------------------------------
# 3. TRAIN / TEST SPLIT (Like V3)
# -----------------------------------------------------------
# Using 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 4. TRAIN MODEL (Using V3 Aggressive Settings)
# -----------------------------------------------------------
model = LGBMRegressor(
    n_estimators=500,        # Lots of trees
    learning_rate=0.03,      # Slower learning for better precision
    num_leaves=50,           # High leaves (V3 setting) to allow complex learning
    max_depth=-1,            # Unlimited depth (V3 setting)
    min_child_samples=3,     # Allow learning from small groups of data
    random_state=42
)

print("Training Model...")
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"\nModel Results on Test Set (20% of data):")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# -----------------------------------------------------------
# 5. VISUALIZATION (The 4 Plots you asked for)
# -----------------------------------------------------------

# Plot 1: True vs Predicted
plt.figure(figsize=(7, 6))
jitter = np.random.normal(0, 0.05, size=len(preds))
plt.scatter(y_test, preds + jitter, alpha=0.85, color="blue", label="Predictions")
plt.plot([5, 10], [5, 10], color="yellow", linewidth=3, label="Ideal x=y")

# Regression Line
m, b = np.polyfit(y_test, preds, 1)
x_range = np.array([5, 10])
plt.plot(x_range, m * x_range + b, color="red", linewidth=2, label="Trend Line")

plt.xlim(4.5, 10.5)
plt.ylim(4.5, 10.5)
plt.xlabel("True RPE")
plt.ylabel("Predicted RPE")
plt.title(f"True vs Predicted (MAE: {mae:.2f})")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Residuals
residuals = y_test - preds
plt.figure(figsize=(7, 4))
plt.scatter(preds, residuals, alpha=0.75, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted RPE")
plt.ylabel("Residual (True - Predicted)")
plt.title("Residuals")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Plot 3: Distribution
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=10, color="steelblue", edgecolor="black")
plt.title("Error Distribution")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Plot 4: Feature Importance
importances = model.feature_importances_
# Combine feature names and importances
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

print("\nTop 10 Feature Importances:")
print(feature_imp_df)

plt.figure(figsize=(8, 5))
plt.bar(feature_imp_df['Feature'], feature_imp_df['Importance'], color="skyblue")
plt.title("Feature Importance (What the model actually used)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()