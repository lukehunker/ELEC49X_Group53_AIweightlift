import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# MULTI-MODAL VERSION
# Uses: OpenFace (facial) + MMPose (body angles) + Bar tracking (kinematics)
# Removes: Time-based leakage (total_frames, rep_change_s)
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

# -----------------------------------------------------------
# REMOVE FRAME COUNT LEAKAGE (keep bar speed/tempo features)
# -----------------------------------------------------------
LEAKAGE_FEATURES = [
    'total_frames',      # Absolute frame count - redundant with rep_change_s
    'detection_count',   # Frame count-based
]

print(f"\n{'='*60}")
print("MULTI-MODAL FEATURE ANALYSIS:")
print('='*60)

# Categorize existing features
openface_features = [c for c in numeric_cols if ('AU' in c or 'confidence' in c or 'detection_rate' in c or 'au_highly' in c)]
mmpose_features = [c for c in numeric_cols if 'd_value' in c]
bar_tracking_features = [c for c in numeric_cols if c == 'rep_count']  # Add more if available

print(f"\n✓ OpenFace (facial): {len(openface_features)} features")
print(f"  {openface_features[:5]}...")
print(f"\n✓ MMPose (body pose): {len(mmpose_features)} features") 
print(f"  {mmpose_features}")
print(f"\n✓ Bar tracking: {len(bar_tracking_features)} features")
print(f"  {bar_tracking_features}")

# Check if we have rep_change_s (bar speed/tempo indicator)
if 'rep_change_s' in numeric_cols:
    print(f"\n✓ Keeping rep_change_s (bar speed/tempo)")

# Remove frame count leakage
print(f"\n❌ Removing frame count leakage:")
for feat in LEAKAGE_FEATURES:
    if feat in numeric_cols:
        print(f"  - {feat}")
        numeric_cols.remove(feat)

print(f"\nBase features remaining: {len(numeric_cols)}")
print('='*60)

# -----------------------------------------------------------
# FEATURE ENGINEERING (Multi-Modal)
# -----------------------------------------------------------
print("\nEngineering multi-modal features...")

# 1. OPENFACE (Facial Strain)
au_cols = [col for col in numeric_cols if col.startswith('AU')]
if len(au_cols) > 0:
    df['AU_sum'] = df[au_cols].sum(axis=1)
    df['AU_mean'] = df[au_cols].mean(axis=1)
    df['AU_max'] = df[au_cols].max(axis=1)
    df['AU_std'] = df[au_cols].std(axis=1)
    
    # Critical AU interactions (strain patterns)
    if 'AU04_max' in au_cols and 'AU07_max' in au_cols:
        df['AU04_07_interaction'] = df['AU04_max'] * df['AU07_max']  # Brow lowerer + lid tightener
    if 'AU06_max' in au_cols and 'AU12_max' in au_cols:
        df['AU06_12_interaction'] = df['AU06_max'] * df['AU12_max']  # Cheek raiser + lip corner pull
    if 'AU09_max' in au_cols and 'AU10_max' in au_cols:
        df['AU09_10_interaction'] = df['AU09_max'] * df['AU10_max']  # Nose wrinkler + upper lip
    if 'AU25_max' in au_cols and 'AU26_max' in au_cols:
        df['AU25_26_interaction'] = df['AU25_max'] * df['AU26_max']  # Lip part + jaw drop
    if 'AU04_max' in au_cols and 'AU06_max' in au_cols:
        df['AU04_06_interaction'] = df['AU04_max'] * df['AU06_max']  # Upper face strain

# 2. DETECTION QUALITY (OpenFace reliability)
if 'detection_rate' in numeric_cols and 'confidence_mean' in numeric_cols:
    df['quality_score'] = df['detection_rate'] * df['confidence_mean']
    
if 'confidence_mean' in numeric_cols and 'confidence_std' in numeric_cols:
    df['confidence_cv'] = df['confidence_std'] / (df['confidence_mean'] + 1e-6)

# 3. MMPOSE + BAR TRACKING INTERACTIONS (Body kinematics + facial strain)
if 'd_value' in numeric_cols:
    # d_value is body angle change metric from MMPose
    # Higher d_value = more body angle change during lift
    
    # Interaction: Body movement × Facial strain
    if 'AU_sum' in df.columns:
        df['body_face_strain'] = df['d_value'] * df['AU_sum']  # Combined movement + facial effort
    if 'AU_mean' in df.columns:
        df['d_value_x_AU_mean'] = df['d_value'] * df['AU_mean']
    if 'au_highly_active_count' in numeric_cols:
        df['d_value_x_active_AUs'] = df['d_value'] * df['au_highly_active_count']

# 4. BAR SPEED / TEMPO FEATURES (rep_change_s)
if 'rep_change_s' in numeric_cols and 'rep_count' in numeric_cols:
    # Average time per rep (slower = more effort)
    df['avg_rep_duration'] = df['rep_change_s'] / (df['rep_count'] + 1)
    
    # Interaction: slow reps + high facial strain
    if 'AU_sum' in df.columns:
        df['tempo_x_facial_strain'] = df['avg_rep_duration'] * df['AU_sum']
    
    # Interaction: slow reps + body angle change
    if 'd_value' in numeric_cols:
        df['tempo_x_body_movement'] = df['avg_rep_duration'] * df['d_value']

if 'rep_count' in numeric_cols:
    # Rep count = number of reps performed (NOT time-based)
    # More reps with same weight = different fatigue patterns
    
    # Strain per rep
    if 'au_highly_active_count' in numeric_cols:
        df['strain_per_rep'] = df['au_highly_active_count'] / (df['rep_count'] + 1)
    if 'AU_sum' in df.columns:
        df['AU_sum_per_rep'] = df['AU_sum'] / (df['rep_count'] + 1)
    
    # Body movement per rep
    if 'd_value' in numeric_cols:
        df['d_value_per_rep'] = df['d_value'] / (df['rep_count'] + 1)
    
    # Lift type × rep count interactions (different lifts handle volume differently)
    df['rep_count_sq'] = df['rep_count'] ** 2  # Non-linear rep effects
    df['lift_type_x_reps'] = df['Lift_Type_Code'] * df['rep_count']

# Update features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(target)

# Double-check no frame count leakage remains
for leak_feat in LEAKAGE_FEATURES:
    if leak_feat in numeric_cols:
        numeric_cols.remove(leak_feat)

print(f"\nFinal feature count: {len(numeric_cols)} (multi-modal + engineered)")
print(f"Includes rep_change_s: {'rep_change_s' in numeric_cols} (bar speed/tempo)")

X = df[numeric_cols].copy()
y = df[target].copy()
y_bins = pd.cut(y, bins=[0, 5, 7, 9, 11], labels=[0, 1, 2, 3])


# -----------------------------------------------------------
# OPTUNA OPTIMIZATION
# -----------------------------------------------------------
def objective(trial):
    param = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "n_estimators": trial.suggest_int("n_estimators", 1500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.05, log=True),
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
    
    if param["boosting_type"] == "dart":
        param["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.2)
        param["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)

    skf = StratifiedKFold(n_splits=7, shuffle=True)
    fold_maes = []

    for train_idx, val_idx in skf.split(X, y_bins):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
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
# RUN OPTIMIZATION
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("STARTING MULTI-MODAL OPTIMIZATION (300 Trials)")
print("Using: OpenFace + MMPose + Bar Tracking")
print("=" * 60)

study = optuna.create_study(direction="minimize", 
                           pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
study.optimize(objective, n_trials=300, show_progress_bar=True)

print("\n" + "=" * 60)
print("OPTIMIZATION COMPLETE")
print(f"Best MAE Found: {study.best_value:.4f}")
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print("=" * 60)

# -----------------------------------------------------------
# FINAL MODEL EVALUATION
# -----------------------------------------------------------
best_params = study.best_params.copy()
best_params.update({"objective": "regression", "metric": "mae", 
                   "n_jobs": -1, "verbosity": -1})

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

# Feature importance analysis
avg_importance = np.mean(feature_importance_list, axis=0)
feature_importance_df = pd.DataFrame({
    'feature': numeric_cols,
    'importance': avg_importance
}).sort_values('importance', ascending=False)

print(f"\n{'='*60}")
print("TOP 20 MOST IMPORTANT FEATURES (MULTI-MODAL):")
print(feature_importance_df.head(20).to_string(index=False))
print('='*60)

# Categorize top features by source
top_20 = feature_importance_df.head(20)
openface_count = sum('AU' in f or 'confidence' in f or 'detection' in f for f in top_20['feature'])
mmpose_count = sum('d_value' in f or 'body' in f for f in top_20['feature'])
bar_count = sum('rep' in f for f in top_20['feature'])
lift_type_count = sum('Lift_Type' in f or 'lift_type' in f for f in top_20['feature'])

print(f"\nTop 20 Feature Sources:")
print(f"  OpenFace (facial): {openface_count}")
print(f"  MMPose (body): {mmpose_count}")
print(f"  Bar tracking: {bar_count}")
print(f"  Lift type: {lift_type_count}")
print('='*60)

# -----------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Predictions
ax1 = axes[0]
jitter = np.random.normal(0, 0.05, size=len(all_y_pred))
ax1.scatter(all_y_true, np.array(all_y_pred) + jitter, alpha=0.6, color="blue", s=30)
ax1.plot([1, 10], [1, 10], color="orange", linewidth=3, label="Perfect")

m, b = np.polyfit(all_y_true, all_y_pred, 1)
x_range = np.array([1, 10])
ax1.plot(x_range, m * x_range + b, color="red", linewidth=3, linestyle='--', label=f"Fit: y={m:.2f}x+{b:.2f}")

ax1.set_xlim(0.5, 10.5)
ax1.set_ylim(0.5, 10.5)
ax1.set_xlabel("True RPE", fontsize=12)
ax1.set_ylabel("Predicted RPE", fontsize=12)
ax1.set_title(f"Multi-Modal Model ({len(numeric_cols)} Features)\nFace + Body + Kinematics\nMAE: {final_mae:.4f} | R²: {final_r2:.4f}", 
             fontsize=13, fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Plot 2: Feature Importance
ax2 = axes[1]
top_features = feature_importance_df.head(20)
colors = []
for feat in top_features['feature']:
    if 'AU' in feat or 'confidence' in feat or 'detection' in feat:
        colors.append('steelblue')  # OpenFace
    elif 'd_value' in feat or 'body' in feat:
        colors.append('forestgreen')  # MMPose
    elif 'rep' in feat:
        colors.append('coral')  # Bar tracking
    else:
        colors.append('gray')  # Other

ax2.barh(range(len(top_features)), top_features['importance'], color=colors)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'], fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Top 20 Features\n🔵 Facial | 🟢 Body | 🟠 Kinematics', fontsize=13, fontweight='bold')
ax2.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('lgbm_multimodal_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*60}")
print(f"FINAL RESULTS (MULTI-MODAL):")
print(f"  MAE: {final_mae:.4f}")
print(f"  R²:  {final_r2:.4f}")
print(f"\n  Features used:")
print(f"    - OpenFace: Facial expressions (AU units)")
print(f"    - MMPose: Body angles/form (d_value)")
print(f"    - Bar speed: Tempo/speed (rep_change_s, avg_rep_duration)")
print(f"    - Kinematics: Rep count & volume")
print(f"    - Removed: total_frames (frame count leakage)")
print('='*60)
