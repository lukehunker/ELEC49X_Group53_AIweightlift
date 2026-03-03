import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# SIMPLE ENSEMBLE: LGBM + XGBoost
# Quick way to improve without hyperparameter tuning
# Expected: 0.03-0.05 MAE improvement
# -----------------------------------------------------------

file_path = "../Train_Outputs/Master_Results.xlsx"
df = pd.read_excel(file_path)

print(f"Loaded {len(df)} rows")
target = "RPE"

def extract_lift_type(set_name):
    set_name = str(set_name).lower()
    if "bench" in set_name:
        return 0
    elif "squat" in set_name:
        return 1
    elif "dead" in set_name:
        return 2
    else:
        return 3

df["Lift_Type_Code"] = df["Set"].apply(extract_lift_type)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(target)
df = df.dropna(subset=[target] + numeric_cols)

# Remove leakage
LEAKAGE_FEATURES = ['total_frames', 'detection_count']
for feat in LEAKAGE_FEATURES:
    if feat in numeric_cols:
        numeric_cols.remove(feat)

# Feature engineering (same as Multi-Modal)
au_cols = [col for col in numeric_cols if col.startswith('AU')]
if len(au_cols) > 0:
    df['AU_sum'] = df[au_cols].sum(axis=1)
    df['AU_mean'] = df[au_cols].mean(axis=1)
    df['AU_max'] = df[au_cols].max(axis=1)
    df['AU_std'] = df[au_cols].std(axis=1)
    
    if 'AU04_max' in au_cols and 'AU07_max' in au_cols:
        df['AU04_07_interaction'] = df['AU04_max'] * df['AU07_max']
    if 'AU06_max' in au_cols and 'AU12_max' in au_cols:
        df['AU06_12_interaction'] = df['AU06_max'] * df['AU12_max']
    if 'AU09_max' in au_cols and 'AU10_max' in au_cols:
        df['AU09_10_interaction'] = df['AU09_max'] * df['AU10_max']
    if 'AU25_max' in au_cols and 'AU26_max' in au_cols:
        df['AU25_26_interaction'] = df['AU25_max'] * df['AU26_max']
    if 'AU04_max' in au_cols and 'AU06_max' in au_cols:
        df['AU04_06_interaction'] = df['AU04_max'] * df['AU06_max']

if 'detection_rate' in numeric_cols and 'confidence_mean' in numeric_cols:
    df['quality_score'] = df['detection_rate'] * df['confidence_mean']
    
if 'confidence_mean' in numeric_cols and 'confidence_std' in numeric_cols:
    df['confidence_cv'] = df['confidence_std'] / (df['confidence_mean'] + 1e-6)

if 'd_value' in numeric_cols:
    if 'AU_sum' in df.columns:
        df['body_face_strain'] = df['d_value'] * df['AU_sum']
    if 'AU_mean' in df.columns:
        df['d_value_x_AU_mean'] = df['d_value'] * df['AU_mean']
    if 'au_highly_active_count' in numeric_cols:
        df['d_value_x_active_AUs'] = df['d_value'] * df['au_highly_active_count']

if 'rep_change_s' in numeric_cols and 'rep_count' in numeric_cols:
    df['avg_rep_duration'] = df['rep_change_s'] / (df['rep_count'] + 1)
    if 'AU_sum' in df.columns:
        df['tempo_x_facial_strain'] = df['avg_rep_duration'] * df['AU_sum']
    if 'd_value' in numeric_cols:
        df['tempo_x_body_movement'] = df['avg_rep_duration'] * df['d_value']

if 'rep_count' in numeric_cols:
    df['rep_count_sq'] = df['rep_count'] ** 2
    if 'au_highly_active_count' in numeric_cols:
        df['strain_per_rep'] = df['au_highly_active_count'] / (df['rep_count'] + 1)
    if 'AU_sum' in df.columns:
        df['AU_sum_per_rep'] = df['AU_sum'] / (df['rep_count'] + 1)
    if 'd_value' in numeric_cols:
        df['d_value_per_rep'] = df['d_value'] / (df['rep_count'] + 1)
    df['lift_type_x_reps'] = df['Lift_Type_Code'] * df['rep_count']

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(target)
for leak_feat in LEAKAGE_FEATURES:
    if leak_feat in numeric_cols:
        numeric_cols.remove(leak_feat)

print(f"Feature count: {len(numeric_cols)}")

X = df[numeric_cols].copy()
y = df[target].copy()
y_bins = pd.cut(y, bins=[0, 5, 7, 9, 11], labels=[0, 1, 2, 3])

# -----------------------------------------------------------
# ENSEMBLE TRAINING
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING ENSEMBLE: LGBM + XGBoost")
print("=" * 60)

# Best params from your previous run
lgbm_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 2600,
    'learning_rate': 0.0095,
    'num_leaves': 33,
    'max_depth': 9,
    'min_child_samples': 4,
    'min_child_weight': 0.036,
    'subsample': 0.74,
    'subsample_freq': 6,
    'colsample_bytree': 0.64,
    'reg_alpha': 0.13,
    'reg_lambda': 0.38,
    'min_gain_to_split': 0.0005,
    'max_bin': 216,
    'verbose': -1,
    'n_jobs': -1
}

# XGBoost with similar hyperparameters
xgb_params = {
    'objective': 'reg:absoluteerror',
    'n_estimators': 2600,
    'learning_rate': 0.01,
    'max_depth': 9,
    'min_child_weight': 4,
    'subsample': 0.74,
    'colsample_bytree': 0.64,
    'reg_alpha': 0.13,
    'reg_lambda': 0.38,
    'tree_method': 'hist',
    'n_jobs': -1
}

skf = StratifiedKFold(n_splits=7, shuffle=True)
all_y_true = []
all_lgbm_preds = []
all_xgb_preds = []
all_ensemble_preds = []

print("\nTraining models on CV folds...")
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_bins), 1):
    print(f"  Fold {fold_idx}/7...", end=" ")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train LightGBM
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_train_scaled, y_train)
    lgbm_preds = lgbm_model.predict(X_val_scaled)
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_preds = xgb_model.predict(X_val_scaled)
    
    # Ensemble: weighted average
    ensemble_preds = 0.6 * lgbm_preds + 0.4 * xgb_preds
    ensemble_preds = np.clip(ensemble_preds, 1, 10)
    
    all_y_true.extend(y_val)
    all_lgbm_preds.extend(lgbm_preds)
    all_xgb_preds.extend(xgb_preds)
    all_ensemble_preds.extend(ensemble_preds)
    
    print(f"Done")

# Calculate metrics
lgbm_mae = mean_absolute_error(all_y_true, all_lgbm_preds)
xgb_mae = mean_absolute_error(all_y_true, all_xgb_preds)
ensemble_mae = mean_absolute_error(all_y_true, all_ensemble_preds)
ensemble_r2 = r2_score(all_y_true, all_ensemble_preds)

print("\n" + "=" * 60)
print("RESULTS:")
print(f"  LGBM alone:      MAE = {lgbm_mae:.4f}")
print(f"  XGBoost alone:   MAE = {xgb_mae:.4f}")
print(f"  Ensemble (60/40): MAE = {ensemble_mae:.4f}  R² = {ensemble_r2:.4f}")
print(f"  Improvement:     {lgbm_mae - ensemble_mae:.4f}")
print("=" * 60)

# -----------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Ensemble predictions
ax1 = axes[0]
jitter = np.random.normal(0, 0.05, size=len(all_ensemble_preds))
ax1.scatter(all_y_true, np.array(all_ensemble_preds) + jitter, alpha=0.6, color="purple", s=30)
ax1.plot([1, 10], [1, 10], color="orange", linewidth=3, label="Perfect")

m, b = np.polyfit(all_y_true, all_ensemble_preds, 1)
x_range = np.array([1, 10])
ax1.plot(x_range, m * x_range + b, color="red", linewidth=3, linestyle='--', label=f"Fit: y={m:.2f}x+{b:.2f}")

ax1.set_xlim(0.5, 10.5)
ax1.set_ylim(0.5, 10.5)
ax1.set_xlabel("True RPE", fontsize=13, fontweight='bold')
ax1.set_ylabel("Predicted RPE", fontsize=13, fontweight='bold')
ax1.set_title(f"Ensemble: LGBM(60%) + XGBoost(40%)\nMAE: {ensemble_mae:.4f} | R²: {ensemble_r2:.4f}", 
             fontsize=14, fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=11)

# Plot 2: Model comparison
ax2 = axes[1]
models = ['LGBM', 'XGBoost', 'Ensemble']
maes = [lgbm_mae, xgb_mae, ensemble_mae]
colors = ['steelblue', 'forestgreen', 'purple']

bars = ax2.bar(models, maes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('MAE', fontsize=13, fontweight='bold')
ax2.set_title('Model Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# Add values on bars
for bar, mae in zip(bars, maes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{mae:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement annotation
improvement = lgbm_mae - ensemble_mae
ax2.annotate(f'↓ {improvement:.4f}', 
            xy=(2, ensemble_mae), xytext=(2, lgbm_mae),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', fontweight='bold',
            ha='center')

plt.tight_layout()
plt.savefig('lgbm_ensemble_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Ensemble achieved MAE: {ensemble_mae:.4f}")
print(f"{'   Getting closer to 0.50-0.65 target!' if ensemble_mae < 0.67 else '   Try LGBMMod2_Advanced.py for further improvements'}")
