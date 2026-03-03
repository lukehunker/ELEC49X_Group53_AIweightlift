import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import optuna
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# 1. LOAD DATA & ENHANCED FEATURE ENGINEERING
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

# Get base numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(target)

df = df.dropna(subset=[target] + numeric_cols)

# -----------------------------------------------------------
# FEATURE ENGINEERING: Add Interaction and Derived Features
# -----------------------------------------------------------
print("\n" + "=" * 50)
print("CREATING ENHANCED FEATURES")
print("=" * 50)

# Action Unit interactions (facial strain combinations)
au_cols = [col for col in numeric_cols if col.startswith('AU')]
if len(au_cols) >= 2:
    df['AU_sum'] = df[au_cols].sum(axis=1)
    df['AU_mean'] = df[au_cols].mean(axis=1)
    df['AU_std'] = df[au_cols].std(axis=1)
    df['AU_max'] = df[au_cols].max(axis=1)
    
    # Specific AU interactions (common strain patterns)
    if 'AU04_max' in au_cols and 'AU07_max' in au_cols:
        df['AU04_07_interaction'] = df['AU04_max'] * df['AU07_max']  # Brow + lid tightener
    if 'AU06_max' in au_cols and 'AU12_max' in au_cols:
        df['AU06_12_interaction'] = df['AU06_max'] * df['AU12_max']  # Cheek + lip corner raiser
    if 'AU09_max' in au_cols and 'AU10_max' in au_cols:
        df['AU09_10_interaction'] = df['AU09_max'] * df['AU10_max']  # Nose wrinkler + upper lip

# Detection quality features
if 'detection_rate' in numeric_cols and 'confidence_mean' in numeric_cols:
    df['quality_score'] = df['detection_rate'] * df['confidence_mean']
    
if 'confidence_mean' in numeric_cols and 'confidence_std' in numeric_cols:
    df['confidence_cv'] = df['confidence_std'] / (df['confidence_mean'] + 1e-6)  # Coefficient of variation

# Rep-based features
if 'rep_count' in numeric_cols and 'rep_change_s' in numeric_cols:
    df['rep_intensity'] = df['rep_count'] / (df['rep_change_s'] + 1e-6)  # Reps per second
    
if 'au_highly_active_count' in numeric_cols and 'rep_count' in numeric_cols:
    df['strain_per_rep'] = df['au_highly_active_count'] / (df['rep_count'] + 1)

# Polynomial features for key metrics
if 'confidence_mean' in numeric_cols:
    df['confidence_mean_sq'] = df['confidence_mean'] ** 2
    
if 'detection_rate' in numeric_cols:
    df['detection_rate_sq'] = df['detection_rate'] ** 2

# Update feature list
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(target)

print(f"Enhanced to {len(numeric_cols)} features (from 21 baseline)")

X = df[numeric_cols].copy()
y = df[target].copy()

# Create RPE bins for stratified splitting
y_bins = pd.cut(y, bins=[0, 5, 7, 9, 11], labels=[0, 1, 2, 3])

print(f"\nFeature list: {numeric_cols[:10]}... (showing first 10)")


# -----------------------------------------------------------
# 2. ENHANCED OPTUNA OBJECTIVE FUNCTION
# -----------------------------------------------------------
def objective(trial):
    # Expanded search space with additional LightGBM parameters
    param = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "n_estimators": trial.suggest_int("n_estimators", 1500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.04, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 60),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "n_jobs": -1
    }
    
    # Add dart-specific parameters
    if param["boosting_type"] == "dart":
        param["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.2)
        param["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)

    # Use Stratified K-Fold for better handling of imbalanced RPE distribution
    skf = StratifiedKFold(n_splits=7, shuffle=True)
    fold_maes = []

    for train_idx, val_idx in skf.split(X, y_bins):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Optional: Scale features (helps with regularization)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = lgb.LGBMRegressor(**param)

        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
        )

        preds = model.predict(X_val_scaled)
        fold_maes.append(mean_absolute_error(y_val, preds))

    return np.mean(fold_maes)


# -----------------------------------------------------------
# 3. RUN THE OPTIMIZATION (More trials for better exploration)
# -----------------------------------------------------------
print("\n" + "=" * 50)
print("STARTING OPTUNA HYPERPARAMETER TUNING (300 Trials)")
print("=" * 50)

study = optuna.create_study(direction="minimize", 
                           pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
study.optimize(objective, n_trials=300, show_progress_bar=True)

print("\n" + "=" * 50)
print("OPTIMIZATION COMPLETE")
print(f"Best MAE Found: {study.best_value:.4f}")
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print("=" * 50)

# -----------------------------------------------------------
# 4. TRAIN FINAL MODEL AND EVALUATE
# -----------------------------------------------------------
best_params = study.best_params.copy()
best_params["objective"] = "regression"
best_params["metric"] = "mae"
best_params["n_jobs"] = -1
best_params["verbosity"] = -1

skf = StratifiedKFold(n_splits=7, shuffle=True)
all_y_true = []
all_y_pred = []
feature_importance_list = []

for train_idx, val_idx in skf.split(X, y_bins):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_val_scaled)
    all_y_true.extend(y_val)
    all_y_pred.extend(preds)
    
    feature_importance_list.append(model.feature_importances_)

final_mae = mean_absolute_error(all_y_true, all_y_pred)
final_r2 = r2_score(all_y_true, all_y_pred)

# Average feature importances
avg_importance = np.mean(feature_importance_list, axis=0)
feature_importance_df = pd.DataFrame({
    'feature': numeric_cols,
    'importance': avg_importance
}).sort_values('importance', ascending=False)

print(f"\n{'='*50}")
print("TOP 15 MOST IMPORTANT FEATURES:")
print(feature_importance_df.head(15).to_string(index=False))
print('='*50)

# -----------------------------------------------------------
# 5. VISUALIZATION
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Predictions vs Truth
ax1 = axes[0]
jitter = np.random.normal(0, 0.05, size=len(all_y_pred))
ax1.scatter(all_y_true, np.array(all_y_pred) + jitter, alpha=0.6, color="blue", s=30)
ax1.plot([1, 10], [1, 10], color="orange", linewidth=3, label="Perfect Prediction")

m, b = np.polyfit(all_y_true, all_y_pred, 1)
x_range = np.array([1, 10])
ax1.plot(x_range, m * x_range + b, color="red", linewidth=3, label=f"Fit: y={m:.2f}x+{b:.2f}")

ax1.set_xlim(0.5, 10.5)
ax1.set_ylim(0.5, 10.5)
ax1.set_xlabel("True RPE", fontsize=12)
ax1.set_ylabel("Predicted RPE", fontsize=12)
ax1.set_title(f"Enhanced Model ({len(numeric_cols)} Features)\nMAE: {final_mae:.4f} | R²: {final_r2:.4f}", fontsize=13, fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Plot 2: Feature Importance
ax2 = axes[1]
top_features = feature_importance_df.head(15)
ax2.barh(range(len(top_features)), top_features['importance'], color='steelblue')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'])
ax2.invert_yaxis()
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Top 15 Feature Importances', fontsize=13, fontweight='bold')
ax2.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('lgbm_enhanced_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*50}")
print(f"FINAL RESULTS:")
print(f"  MAE: {final_mae:.4f}")
print(f"  R²:  {final_r2:.4f}")
print(f"  Improvement over baseline (0.6952): {0.6952 - final_mae:.4f}")
print('='*50)
