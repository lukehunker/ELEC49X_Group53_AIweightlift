import pandas as pd
import os
import re


def run():
    # ---------------------------------------------------------
    # CONFIG: Exact Paths
    # ---------------------------------------------------------
    rpe_path = "./dataset_labelled.csv"
    bar_speed_path = "./Train_Outputs/barSpeed.xlsx"
    openface_path = "./Train_Outputs/openface_features_all.csv"
    angleChange_path = "./Train_Outputs/dmetrics.xlsx"
    output_excel_path = "./Train_Outputs/Master_Results.xlsx"

    print(f"Loading RPE data from: {rpe_path}")
    print(f"Loading Bar Speed data from: {bar_speed_path}")
    print(f"Loading OpenFace data from: {openface_path}")
    print(f"Loading D-Metrics data from: {angleChange_path}")

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------

    # --- Load RPE ---
    try:
        rpe_df = pd.read_csv(rpe_path)
        # Normalize column names (strip whitespace)
        rpe_df.columns = rpe_df.columns.str.strip()

        # Verify columns exist
        if 'Video' not in rpe_df.columns or 'RPE' not in rpe_df.columns:
            print(f"[ERROR] RPE file missing 'Video' or 'RPE' columns. Found: {rpe_df.columns.tolist()}")
            return

        # Rename 'Video' to 'video_stem' for clear merging later
        rpe_df['video_stem'] = rpe_df['Video'].astype(str).str.strip()
        rpe_subset = rpe_df[['video_stem', 'RPE']].copy()

    except FileNotFoundError:
        print(f"[ERROR] Could not find {rpe_path}")
        return

    # --- Load Bar Speed ---
    try:
        bs_df = pd.read_excel(bar_speed_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {bar_speed_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to read Bar Speed Excel: {e}")
        return

    # --- Load OpenFace ---
    try:
        of_df = pd.read_csv(openface_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {openface_path}")
        return

    # --- Load D-Metrics (Angle Change) ---
    try:
        dmetrics_df = pd.read_excel(angleChange_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {angleChange_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to read D-Metrics Excel: {e}")
        return

    # ---------------------------------------------------------
    # 2. Prepare DataFrames (create matching keys)
    # ---------------------------------------------------------

    def get_stem(filename):
        return os.path.splitext(str(filename))[0].strip()

    # --- Filter & Prep Bar Speed ---
    req_bs_cols = ['video_name', 'delta_speed_m_s', 'rep_count']
    if not all(c in bs_df.columns for c in req_bs_cols):
        print(f"[ERROR] Bar Speed file missing columns. Needed: {req_bs_cols}")
        return

    bs_subset = bs_df[req_bs_cols].copy()
    bs_subset['video_name'] = bs_subset['video_name'].astype(str).str.strip()
    bs_subset['video_stem'] = bs_subset['video_name'].apply(get_stem)

    # --- Filter & Prep OpenFace ---
    if 'video_name' not in of_df.columns:
        print("[ERROR] OpenFace file is missing 'video_name' column")
        return

    au_cols = [c for c in of_df.columns if re.search(r"AU\d+_max", c)]
    print(f"Found {len(au_cols)} AU max columns")

    req_of_cols = ['video_name'] + au_cols
    of_subset = of_df[req_of_cols].copy()
    of_subset['video_name'] = of_subset['video_name'].astype(str).str.strip()

    # --- Filter & Prep D-Metrics ---
    if 'video_name' not in dmetrics_df.columns:
        print("[ERROR] D-Metrics file is missing 'video_name' column")
        return

    # We take all columns from dmetrics, just clean the merge key
    dmetrics_df['video_name'] = dmetrics_df['video_name'].astype(str).str.strip()

    # ---------------------------------------------------------
    # 3. Merge Strategy
    # ---------------------------------------------------------
    print("Merging data...")

    # Step A: Merge Bar Speed and OpenFace on 'video_name'
    master_df = pd.merge(bs_subset, of_subset, on='video_name', how='outer')

    # Step B: Merge D-Metrics into master on 'video_name'
    master_df = pd.merge(master_df, dmetrics_df, on='video_name', how='outer')

    # Step C: Fill missing 'video_stem' if needed
    if 'video_stem' not in master_df.columns or master_df['video_stem'].isnull().any():
        master_df['video_stem'] = master_df['video_name'].apply(get_stem)

    # Step D: Merge RPE data on 'video_stem'
    final_df = pd.merge(master_df, rpe_subset, on='video_stem', how='left')

    # ---------------------------------------------------------
    # 4. Reorder Columns
    # ---------------------------------------------------------
    cols = list(final_df.columns)

    head_cols = ['video_name', 'RPE', 'delta_speed_m_s', 'rep_count']
    rest_cols = [c for c in cols if c not in head_cols and c != 'video_stem']
    rest_cols.sort()

    final_cols = head_cols + rest_cols
    final_df = final_df[final_cols]

    # ---------------------------------------------------------
    # 5. Save Output
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)

    try:
        final_df.to_excel(output_excel_path, index=False)
        print(f"\n[SUCCESS] Master Results saved to: {output_excel_path}")
        print(f"Total videos: {len(final_df)}")
        print(f"Videos with RPE found: {final_df['RPE'].notna().sum()}")
        print("First 5 columns:", final_df.columns.tolist()[:5])

    except PermissionError:
        print(f"[ERROR] Could not save to {output_excel_path}. Is the file open?")
    except Exception as e:
        print(f"[ERROR] Generic save error: {e}")


if __name__ == "__main__":
    run()