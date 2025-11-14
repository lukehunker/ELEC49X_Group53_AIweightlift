import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# ---------------------------
# 1) Load
# ---------------------------
file_path = r"C:\Users\ajmal\OneDrive\Desktop\Queen's\Capstone\LGBM Regressor\weightlifting_dataset_v3.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")
print(df.head()); print(df.shape)

X = df[["bar_speed_change", "facial_strain", "rom_change"]]
y = df["RPE"].astype(float)

# ---------------------------
# 2) Stratified split by RPE bins (helps bias)
# ---------------------------
bins = pd.cut(y, bins=[2.5,3.5,4.5,5.5,6.5,7.5,8.5,10.5], labels=False)  # 3..10 -> 0..7
X_train, X_tmp, y_train, y_tmp, bins_train, bins_tmp = train_test_split(
    X, y, bins, test_size=0.3, random_state=42, stratify=bins
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42,
    stratify=bins_tmp
)

# ---------------------------
# 3) Scale inputs (trees don’t need it, but harmless for your pipeline)
# ---------------------------
scaler = MinMaxScaler(feature_range=(0, 10))
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_val_s   = pd.DataFrame(scaler.transform(X_val),   columns=X.columns, index=X_val.index)
X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=X.columns, index=X_test.index)

# (Note) Multiplying "weights" into features won’t influence trees much; skip it.

# ---------------------------
# 4) LightGBM with monotone constraints + regularization
#     Order of constraints must match feature order.
#     +1 means prediction must be non-decreasing as feature increases.
# ---------------------------
model = LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=15,            # smaller = smoother
    max_depth=-1,
    min_child_samples=40,     # more conservative splits
    min_split_gain=0.0,
    reg_lambda=1.5,           # L2
    reg_alpha=0.1,            # L1
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    monotone_constraints=[1, 0, 0],  # bar_speed_change ↑ -> RPE ↑ ; others free
    random_state=42
)

model.fit(
    X_train_s, y_train,
    eval_set=[(X_val_s, y_val)],
    eval_metric="l1",              # MAE is robust and reduces big positive errors
    callbacks=[],                  # (you can add early_stopping here if using sklearn API <2.0)
)

# If your LightGBM version supports it, use:
# from lightgbm import early_stopping, log_evaluation
# model.fit(..., callbacks=[early_stopping(50), log_evaluation(50)])

# ---------------------------
# 5) Predict
# ---------------------------
y_pred_val = model.predict(X_val_s)
y_pred     = model.predict(X_test_s)

# Optional bias correction: fit a line on val residuals and correct test preds
# (Pulls predictions down if model is systematically high)
coef = np.polyfit(y_pred_val, (y_val - y_pred_val), 1)  # residual = a*x + b
residual_correction = np.poly1d(coef)
y_pred_corr = y_pred + residual_correction(y_pred)

# Optional quantile blend to prefer slight underprediction (p=0.45)
# Train a second model for quantile and blend; quick proxy below:
quantile_weight = 0.25  # set 0.0 to disable
y_pred_final = (1 - quantile_weight) * y_pred_corr + quantile_weight * np.minimum(y_pred_corr, y_pred)

# Clip to valid RPE range and (optionally) round
y_pred_final = np.clip(y_pred_final, 3, 10)
# y_pred_final = np.rint(y_pred_final)  # uncomment if you want integer outputs

# ---------------------------
# 6) Metrics
# ---------------------------
mse = mean_squared_error(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
bias = float(np.mean(y_pred_final - y_test))  # >0 means overpredicting on average
print(f"\nTest MSE: {mse:.4f}  |  MAE: {mae:.4f}  |  Mean Bias (pred-true): {bias:+.3f}")

# ---------------------------
# 7) Plot: predictions with ideal points (3..10) and fixed axes
# ---------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_final, alpha=0.8, label="Predictions")

diag = np.arange(3, 11, 1)
plt.plot(diag, diag, "o", color="yellow", markersize=8, label="Ideal (x=y)")

plt.xlim(2, 10); plt.ylim(2, 10)
plt.xlabel("True RPE"); plt.ylabel("Predicted RPE")
plt.title("RPE Prediction Performance (LGBM, debiased)")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.tight_layout()
plt.show()

# ---------------------------
# 8) Feature importance
# ---------------------------
importances = model.feature_importances_
for f, s in zip(X.columns, importances):
    print(f"{f}: {s}")
plt.figure(figsize=(5, 4))
plt.bar(X.columns, importances)
plt.title("Feature Importance"); plt.tight_layout(); plt.show()
