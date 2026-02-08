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
    output_excel_path = "./Train_Outputs/Master_Results.xlsx"

    print(f"Loading RPE data from: {rpe_path}")
    print(f"Loading Bar Speed data from: {bar_speed_path}")
    print(f"Loading OpenFace data from: {openface_path}")

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
        # We assume RPE file has "Bench Press 1" (no extension)
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

    # ---------------------------------------------------------
    # 2. Prepare DataFrames (create matching keys)
    # ---------------------------------------------------------

    # Helper to remove extension from video_name to match RPE file
    # e.g. "Bench Press 1.mp4" -> "Bench Press 1"
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

    # Find AU columns
    au_cols = [c for c in of_df.columns if re.search(r"AU\d+_max", c)]
    print(f"Found {len(au_cols)} AU max columns")

    req_of_cols = ['video_name'] + au_cols
    of_subset = of_df[req_of_cols].copy()
    of_subset['video_name'] = of_subset['video_name'].astype(str).str.strip()
    # We don't strictly need video_stem here if we merge with BarSpeed on video_name first,
    # but let's do it for consistency in case BarSpeed is missing some files.

    # ---------------------------------------------------------
    # 3. Merge Strategy
    # ---------------------------------------------------------
    print("Merging data...")

    # Step A: Merge Bar Speed and OpenFace on 'video_name' (Exact file match)
    # Use outer merge to keep files that might have failed in one but not the other
    master_df = pd.merge(bs_subset, of_subset, on='video_name', how='outer')

    # Step B: If 'video_stem' is missing (because rows came only from OpenFace), fill it
    # (OpenFace didn't have video_stem computed above, so we compute it now for the merged frame)
    if 'video_stem' not in master_df.columns or master_df['video_stem'].isnull().any():
        master_df['video_stem'] = master_df['video_name'].apply(get_stem)

    # Step C: Merge RPE data on 'video_stem'
    # Use left merge? No, 'outer' is safest to see what we are missing,
    # but typically we want the videos we have processed. Let's use left join onto the processed videos.
    # If a video is in RPE list but wasn't processed, it won't show up here.
    # If you want ALL videos even unprocessed ones, use 'outer'.
    # I will use 'left' merge onto the master_df (videos we actually have data for).

    final_df = pd.merge(master_df, rpe_subset, on='video_stem', how='left')

    # ---------------------------------------------------------
    # 4. Reorder Columns
    #    Desired: video_name, RPE, delta_speed, rep_count, AUs...
    # ---------------------------------------------------------
    cols = list(final_df.columns)

    # Start with our fixed headers
    head_cols = ['video_name', 'RPE', 'delta_speed_m_s', 'rep_count']

    # The rest (AUs, etc), excluding the ones we just listed and the helper 'video_stem'
    rest_cols = [c for c in cols if c not in head_cols and c != 'video_stem']

    # specific sort for AUs if you want them ordered (optional)
    rest_cols.sort()

    final_cols = head_cols + rest_cols

    # Re-create DataFrame with explicit order
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