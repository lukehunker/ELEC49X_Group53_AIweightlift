"""
Demo mode for barspeed tracking - processes videos from a single folder without movement subfolders.
Detects movement type from filename.
"""
import os
from . import barspeed_to_excel as bst

def run_demo(demo_folder="../../lifting_videos/Demo_Videos", output_dir=None):
    """
    Process all videos in a single demo folder (no movement subfolders).
    Detects movement type from video filename.
    """
    print("\n==========================================")
    print("      DEMO MODE: SINGLE FOLDER PROCESSING")
    print("==========================================\n")

    if not os.path.exists(demo_folder):
        print(f"[ERROR] Demo folder not found: {demo_folder}")
        return

    # Find all video files
    all_files = os.listdir(demo_folder)
    video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    video_files.sort(key=bst.extract_number)

    if not video_files:
        print(f"[ERROR] No videos found in {demo_folder}")
        return

    print(f"Found {len(video_files)} videos in demo folder")
    print(f"Demo folder: {demo_folder}\n")

    # Prepare output directory
    if output_dir is None:
        demo_out = os.path.join(bst.OUTPUT_ROOT, "Demo_Results")
    else:
        demo_out = output_dir
    os.makedirs(demo_out, exist_ok=True)

    # Process each video
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(demo_folder, video_file)
        print(f"\n[{i + 1}/{len(video_files)}] Processing: {video_file} ...")

        # Detect movement type from filename
        video_lower = video_file.lower()
        if "bench" in video_lower:
            movement_name = "Bench Press"
        elif "squat" in video_lower:
            movement_name = "Squat"
        elif "deadlift" in video_lower:
            movement_name = "Deadlift"
        else:
            print(f"[WARN] Unknown movement type, defaulting to 'Unknown'")
            movement_name = "Unknown"

        print(f"Detected movement: {movement_name}")

        try:
            feats = bst.process_video(video_path, movement_name, demo_out)

            if feats:
                # Update master Excel
                import pandas as pd
                if os.path.exists(bst.MASTER_EXCEL_PATH):
                    try:
                        master_df = pd.read_excel(bst.MASTER_EXCEL_PATH)
                        master_df = master_df[master_df["video_name"] != feats["video_name"]]
                    except Exception:
                        master_df = pd.DataFrame()
                else:
                    master_df = pd.DataFrame()

                new_row_df = pd.DataFrame([feats])
                final_df = pd.concat([master_df, new_row_df], ignore_index=True)

                try:
                    final_df.to_excel(bst.MASTER_EXCEL_PATH, index=False)
                    print(f"[SAVED] Master Excel updated successfully.")
                except PermissionError:
                    print(f"[ERROR] Could not save to {bst.MASTER_EXCEL_PATH}.")
                    print("        Please close the Excel file if it is open!")
            else:
                print(f"[SKIP] No valid data found for {video_file}")

        except Exception as e:
            print(f"[CRASH] Critical error processing {video_file}: {e}")
            import traceback
            traceback.print_exc()

    print("\n==========================================")
    print("        DEMO PROCESSING COMPLETE")
    print("==========================================")


if __name__ == "__main__":
    run_demo()
