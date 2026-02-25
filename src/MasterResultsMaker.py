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

    try:
        rpe_df = pd.read_csv(rpe_path)
        rpe_df.columns = rpe_df.columns.str.strip()
        if 'Video' not in rpe_df.columns or 'RPE' not in rpe_df.columns:
            print(f"[ERROR] RPE file missing 'Video' or 'RPE'. Found: {rpe_df.columns.tolist()}")
            return
        rpe_df['video_stem'] = rpe_df['Video'].astype(str).str.strip()
        rpe_subset = rpe_df[['video_stem', 'RPE']].copy()
    except FileNotFoundError:
        print(f"[ERROR] Could not find {rpe_path}");
        return

    try:
        bs_df = pd.read_excel(bar_speed_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {bar_speed_path}");
        return

    try:
        of_df = pd.read_csv(openface_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {openface_path}");
        return

    try:
        dmetrics_df = pd.read_excel(angleChange_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {angleChange_path}");
        return

    # ---------------------------------------------------------
    # 2. Prepare DataFrames (create matching keys)
    # ---------------------------------------------------------

    def get_stem(filename):
        if pd.isna(filename): return ""
        return os.path.splitext(str(filename))[0].strip()

    # --- Prep Bar Speed ---
    req_bs_cols = ['video_name', 'delta_speed_m_s', 'rep_count']
    if not all(c in bs_df.columns for c in req_bs_cols):
        print(f"[ERROR] Bar Speed missing columns. Needed: {req_bs_cols}")
        return
    bs_subset = bs_df[req_bs_cols].copy()
    bs_subset['video_name'] = bs_subset['video_name'].astype(str).str.strip()
    bs_subset['video_stem'] = bs_subset['video_name'].apply(get_stem)

    # --- Prep OpenFace ---
    if 'video_name' not in of_df.columns:
        print("[ERROR] OpenFace missing 'video_name' column");
        return
    au_cols = [c for c in of_df.columns if re.search(r"AU\d+_max", c)]
    req_of_cols = ['video_name'] + au_cols
    of_subset = of_df[req_of_cols].copy()
    of_subset['video_stem'] = of_subset['video_name'].astype(str).apply(get_stem)
    of_subset = of_subset.drop(columns=['video_name'])

    # --- Prep D-Metrics (Filtered to d_value only) ---
    if 'video_name' not in dmetrics_df.columns:
        print("[ERROR] D-Metrics missing 'video_name' column");
        return
    if 'd_value' not in dmetrics_df.columns:
        print(f"[ERROR] D-Metrics missing 'd_value' column. Found: {dmetrics_df.columns.tolist()}")
        return

    dmetrics_df['video_stem'] = dmetrics_df['video_name'].astype(str).apply(get_stem)
    # Subset to ONLY keep the stem and d_value
    dmetrics_df = dmetrics_df[['video_stem', 'd_value']].copy()

    # ---------------------------------------------------------
    # 3. Merge Strategy
    # ---------------------------------------------------------
    print("Merging data...")

    # Strip overlapping non-key columns just in case files share generic column names
    of_overlap = [c for c in of_subset.columns if c in bs_subset.columns and c != 'video_stem']
    of_subset = of_subset.drop(columns=of_overlap)

    master_df = pd.merge(bs_subset, of_subset, on='video_stem', how='outer')

    dmetrics_overlap = [c for c in dmetrics_df.columns if c in master_df.columns and c != 'video_stem']
    dmetrics_df = dmetrics_df.drop(columns=dmetrics_overlap)

    master_df = pd.merge(master_df, dmetrics_df, on='video_stem', how='outer')

    # Fill in the final video_name for any rows that were missing from Bar Speed
    if 'video_name' in master_df.columns:
        master_df['video_name'] = master_df['video_name'].fillna(master_df['video_stem'] + ".mp4")
    else:
        master_df['video_name'] = master_df['video_stem']

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
        print(f"Total unique videos: {len(final_df)}")
        print(f"Videos with RPE found: {final_df['RPE'].notna().sum()}")
    except PermissionError:
        print(f"[ERROR] Could not save to {output_excel_path}. Is the file open?")
    except Exception as e:
        print(f"[ERROR] Generic save error: {e}")


if __name__ == "__main__":
    run()