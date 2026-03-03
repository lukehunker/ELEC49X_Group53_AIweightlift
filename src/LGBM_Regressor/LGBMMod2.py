import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import optuna
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# 1. LOAD DATA & FEATURES (Exact logic from LGBMMod.py)
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

# Use ALL original numeric columns (No dropping)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(target)

df = df.dropna(subset=[target] + numeric_cols)

X = df[numeric_cols]
y = df[target]

print(f"Using {len(numeric_cols)} features (Back to the 21 feature baseline)")


# -----------------------------------------------------------
# 2. OPTUNA OBJECTIVE FUNCTION (Tightly constrained)
# -----------------------------------------------------------
def objective(trial):
    # Search space anchored tightly around your winning 0.72 setup
    param = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 20, 40),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 15),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 2.0),  # Strong L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),  # Strong L2 regularization
        "random_state": 42,
        "n_jobs": -1
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**param)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )

        preds = model.predict(X_val)
        fold_maes.append(mean_absolute_error(y_val, preds))

    return np.mean(fold_maes)


# -----------------------------------------------------------
# 3. RUN THE OPTIMIZATION
# -----------------------------------------------------------
print("\n" + "=" * 50)
print("STARTING OPTUNA HYPERPARAMETER TUNING (50 Trials)")
print("=" * 50)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

print("\n" + "=" * 50)
print("OPTIMIZATION COMPLETE")
print(f"Best MAE Found: {study.best_value:.4f}")
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print("=" * 50)

# -----------------------------------------------------------
# 4. TRAIN AND PLOT THE ULTIMATE WINNER
# -----------------------------------------------------------
best_params = study.best_params
best_params["random_state"] = 42
best_params["n_jobs"] = -1

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_y_true = []
all_y_pred = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    all_y_true.extend(y_val)
    all_y_pred.extend(preds)

final_mae = mean_absolute_error(all_y_true, all_y_pred)
final_r2 = r2_score(all_y_true, all_y_pred)

plt.figure(figsize=(8, 7))
jitter = np.random.normal(0, 0.05, size=len(all_y_pred))
plt.scatter(all_y_true, np.array(all_y_pred) + jitter, alpha=0.7, color="blue")
plt.plot([4, 10], [4, 10], color="yellow", linewidth=3)

m, b = np.polyfit(all_y_true, all_y_pred, 1)
x_range = np.array([4, 10])
plt.plot(x_range, m * x_range + b, color="red", linewidth=3)

plt.xlim(3.5, 10.5)
plt.ylim(3.5, 10.5)
plt.xlabel("True RPE")
plt.ylabel("Predicted RPE")
plt.title(f"Optuna Optimized Model (21 Features)\nMAE: {final_mae:.4f} | R2: {final_r2:.4f}")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()