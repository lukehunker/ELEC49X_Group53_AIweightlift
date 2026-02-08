from OpenFace.openface_feature_extractor import OpenFaceExtractor
from OpenFace.impute_missing_features import identify_missing_videos, impute_by_rpe_average, load_rpe_labels
import pandas as pd
import glob
import os

def extract_openface_batch(video_paths_or_folder, rpe_labels_csv=None, output_csv=None):
    """
    Extract OpenFace features from multiple videos.
    Automatically handles failures with imputation.
    
    Args:
        video_paths_or_folder: Either a list of video paths OR a folder path (string)
        rpe_labels_csv: Path to RPE labels for imputation (optional)
        output_csv: Path to save results CSV (optional)
    
    Returns: DataFrame with features for all videos
    
    Output DataFrame contains:
    - video_name: Original video filename
    - detection_rate, confidence_mean: Face detection quality
    - AU04_max, AU06_max, etc.: Max AU intensities (10 AUs × 3 metrics = 30 features)
    - rep_*, landmark_*, temporal features: ~34 additional features
    - imputed: Boolean flag (True if features were imputed due to failed extraction)
    
    Total: ~64 features per video, extracted using:
    - Pose-guided face cropping (MediaPipe)
    - Max values only (not full statistics)
    - 10 key AUs only (not all 17)
    - Downsampled to 10 FPS (faster processing)
    """
    # Handle folder input
    if isinstance(video_paths_or_folder, str) and os.path.isdir(video_paths_or_folder):
        print(f"Scanning folder: {video_paths_or_folder}")
        patterns = ['**/*.mp4', '**/*.mov', '**/*.MOV', '**/*.MP4']
        video_paths = []
        for pattern in patterns:
            video_paths.extend(glob.glob(os.path.join(video_paths_or_folder, pattern), recursive=True))
        video_paths = sorted(list(set(video_paths)))
        print(f"Found {len(video_paths)} videos")
    else:
        video_paths = video_paths_or_folder
    
    if len(video_paths) == 0:
        raise ValueError("No videos found!")
    
    extractor = OpenFaceExtractor(
        use_pose_guidance=True,      # MediaPipe face cropping
        max_only=True,                # Extract max values only
        load_minimal_columns=True,    # Load only 10 key AUs
        sample_fps=10,                # Downsample to 10 FPS
        verbose=True                  # Show progress
    )
    
    # Extract features from all videos
    print(f"\n{'='*80}")
    print(f"EXTRACTING FEATURES FROM {len(video_paths)} VIDEOS")
    print(f"{'='*80}\n")
    
    results = []
    for idx, video_path in enumerate(video_paths, 1):
        print(f"[{idx}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")
        try:
            features = extractor.extract_from_video(video_path)
            result = {'video_name': os.path.basename(video_path)}
            result.update({k: v for k, v in features.items() if k != 'metadata'})
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'video_name': os.path.basename(video_path), 'error': str(e)})
    
    df = pd.DataFrame(results)
    
    # Impute missing features if RPE labels provided
    if rpe_labels_csv and 'error' in df.columns:
        print(f"\n{'='*80}")
        print("IMPUTATION")
        print(f"{'='*80}\n")
        failed_videos = identify_missing_videos(df)
        if len(failed_videos) > 0:
            rpe_df = load_rpe_labels(rpe_labels_csv)
            # impute_by_rpe_average returns (df, summary_df)
            df, imputation_summary = impute_by_rpe_average(df, rpe_df, failed_videos)
            print(f"\nImputed {len(failed_videos)} videos using RPE-based averaging")
        else:
            print("No failed videos - no imputation needed")
            df['imputed'] = False
    
    # Save to CSV if requested
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to: {output_csv}")
    
    return df

def run():
    # Option 1: Process a single video
    # video_list = ["../lifting_videos/Augmented_h264/Bench_Press_h264/Bench Press 1.mp4"]

    # Option 2: Process entire folder (recommended)
    video_folder = "../lifting_videos/Augmented/Bench Press"

    openface_df = extract_openface_batch(
        video_folder,
        rpe_labels_csv="./dataset_labelled.csv",
        output_csv="./Train_Outputs/openface_features_test.csv"
    )

    print(f"\n{'=' * 80}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nTotal videos: {len(openface_df)}")
    print(f"Total features per video: {len(openface_df.columns) - 1}")  # -1 for video_name
    if 'imputed' in openface_df.columns:
        imputed_count = openface_df['imputed'].sum()
        print(f"Imputed videos: {imputed_count}")
        print(f"Original videos: {len(openface_df) - imputed_count}")

    print(f"\nSample features: {list(openface_df.columns)[:10]}...")
    print(f"\nFirst video features:")
    print(openface_df.iloc[0][['video_name', 'detection_rate', 'AU04_max', 'AU12_max']])

# Example usage:
if __name__ == "__main__":
    run()