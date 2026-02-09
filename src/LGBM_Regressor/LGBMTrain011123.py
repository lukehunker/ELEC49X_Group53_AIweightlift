import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import itertools
import os
import matplotlib.pyplot as plt

def run():
    # 1. LOAD DATA
    # ---------------------------------------------------------
    file_path = "../Train_Outputs/Master_Results.xlsx"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    # Load correct file type
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, encoding='latin1')

    print(f"Loaded {len(df)} rows.")

    # 2. PREPARE FEATURES
    # ---------------------------------------------------------
    def extract_lift_type(filename):
        filename = str(filename).lower()
        if 'bench' in filename:
            return 'Bench Press'
        elif 'squat' in filename:
            return 'Squat'
        elif 'dead' in filename:
            return 'Deadlift'
        else:
            return 'Unknown'

    df['Lift_Type'] = df['video_name'].apply(extract_lift_type)

    # FIX 1: Change target to match the CSV header
    target = 'RPE'

    # FIX 2: Add rep_count to the base features
    feature_cols = ['delta_speed_m_s', 'rep_count', 'Lift_Type']

    au_max_cols = [c for c in df.columns if 'max' in c.lower() and 'au' in c.lower() and 'velocity' not in c.lower()]
    feature_cols += au_max_cols
    df = df.dropna(subset=[target])

    X = df[feature_cols].copy()
    y = df[target]

    le = LabelEncoder()
    X['Lift_Type'] = le.fit_transform(X['Lift_Type'])

    # 3. DEFINE THE GRID (The settings we will test)
    # ---------------------------------------------------------
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [7, 15, 31],
        'max_depth': [3, 5, 7, 10],
        'min_child_samples': [3, 5, 7]
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Testing {len(combinations)} different model configurations...")
    print("-" * 60)

    # 4. RUN THE GRID SEARCH
    # ---------------------------------------------------------
    best_mae = float('inf')
    best_params = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, params in enumerate(combinations):
        # Add fixed params
        params['objective'] = 'regression'
        params['metric'] = 'mae'
        params['boosting_type'] = 'gbdt'
        params['verbosity'] = -1
        params['seed'] = 42

        mae_scores = []

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['Lift_Type'])
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Train WITH validation set passed in
            model = lgb.train(
                params,
                train_data,
                num_boost_round=300,
                valid_sets=[val_data],  # <--- FIXED HERE
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            mae_scores.append(mean_absolute_error(y_val, preds))

        avg_mae = np.mean(mae_scores)

        # Print progress every 10 iterations
        if i % 10 == 0:
            print(f"Tested {i}/{len(combinations)}... Current Best MAE: {best_mae:.4f}")

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params

    print("-" * 60)
    print(f"SEARCH COMPLETE.")
    print(f"BEST MAE: {best_mae:.4f}")
    print("BEST PARAMETERS:")
    print(best_params)

    # 5. GENERATE PLOT WITH BEST PARAMS
    # ---------------------------------------------------------
    print("\nGenerating plot with WINNING parameters...")

    all_y_true = []
    all_y_pred = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['Lift_Type'])
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],  # <--- AND HERE
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        all_y_true.extend(y_val)
        all_y_pred.extend(preds)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Plotting
    plt.figure(figsize=(7, 6))
    jitter = np.random.normal(0, 0.06, size=len(all_y_pred))
    plt.scatter(all_y_true, all_y_pred + jitter, alpha=0.85, color="blue", label="Predictions")
    plt.plot([5, 10], [5, 10], color="yellow", linewidth=3, label="Ideal x=y")

    # Regression Line
    m, b = np.polyfit(all_y_true, all_y_pred, 1)
    x_range = np.array([5, 10])
    plt.plot(x_range, m * x_range + b, color="red", linewidth=2, label=f"Trend (R2={r2_score(all_y_true, all_y_pred):.2f})")

    plt.xlim(4.5, 10.5)
    plt.ylim(4.5, 10.5)
    plt.xlabel("True RPE")
    plt.ylabel("Predicted RPE")
    plt.title(f"Optimized Model (MAE: {best_mae:.3f})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. SAVE FINAL MODEL FOR THE APP
    # ---------------------------------------------------------
    import joblib  # Standard tool for saving python objects

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL FOR DEPLOYMENT")
    print("=" * 60)

    # Train on 100% of the data (X and y) using the best found parameters
    full_train_data = lgb.Dataset(X, label=y, categorical_feature=['Lift_Type'])

    final_model = lgb.train(
        best_params,
        full_train_data,
        num_boost_round=500  # Matches your plotting rounds
    )

    # Save the LightGBM Model
    model_save_path = "../Train_Outputs/lgb_rpe_predictor.txt"
    final_model.save_model(model_save_path)
    print(f"✅ Model saved to: {model_save_path}")

    # CRITICAL: Save the LabelEncoder
    # The app needs this to know that "Bench Press" = 0 (or whatever number it learned)
    encoder_save_path = "../Train_Outputs/lift_type_encoder.pkl"
    joblib.dump(le, encoder_save_path)
    print(f"✅ LabelEncoder saved to: {encoder_save_path}")

if __name__ == "__main__":
    run()