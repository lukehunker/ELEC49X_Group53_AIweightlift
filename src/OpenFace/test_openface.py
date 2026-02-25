from OpenFace.openface_feature_extractor import OpenFaceExtractor
from OpenFace.impute_missing_features import identify_missing_videos, impute_by_rpe_average, load_rpe_labels
import OpenFace.openface_utils as ofu
import pandas as pd
import glob
import os
import cv2
import numpy as np


def create_visualization_video(video_path, csv_path, output_path):
    """
    Create a visualization video with OpenFace landmarks overlaid.
    Standalone function for the official pipeline.
    
    Args:
        video_path: Path to input video (pose-guided video)
        csv_path: Path to OpenFace CSV output with landmarks
        output_path: Path to save visualization video
    """
    # Load landmark data
    if not os.path.exists(csv_path):
        print(f"  WARNING: CSV file not found for visualization: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Could not open video for visualization: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"  WARNING: Could not create output video writer")
        cap.release()
        return None
    
    frame_idx = 0
    frames_with_landmarks = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find corresponding row in CSV
        if frame_idx < len(df):
            row = df.iloc[frame_idx]
            success = row.get('success', 0) == 1
            confidence = row.get('confidence', 0)
            
            # Draw detection status
            status_text = f"Frame {frame_idx}/{total_frames}"
            conf_text = f"Detection: {'SUCCESS' if success else 'FAILED'} (conf: {confidence:.2f})"
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if success else (0, 0, 255), 2)
            
            if success:
                frames_with_landmarks += 1
                
                # Draw 68 facial landmarks
                for i in range(68):
                    x_col = f'x_{i}'
                    y_col = f'y_{i}'
                    if x_col in row and y_col in row:
                        x = int(row[x_col])
                        y = int(row[y_col])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Draw AU intensities
                y_offset = 90
                for au in ['AU04_r', 'AU06_r', 'AU12_r', 'AU07_r']:
                    if au in row:
                        au_val = row[au]
                        au_text = f"{au}: {au_val:.2f}"
                        cv2.putText(frame, au_text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 25
        
        writer.write(frame)
        frame_idx += 1
    
    cap.release()
    writer.release()
    
    print(f"  Visualization: {frames_with_landmarks} frames with landmarks -> {os.path.basename(output_path)}")
    
    return output_path


def extract_openface_batch(video_paths_or_folder, rpe_labels_csv=None, output_csv=None, create_visualizations=False, output_dir=None):
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
        use_pose_guidance=True,
        max_only=True,
        load_minimal_columns=True,
        sample_fps=None,
        verbose=True
    )
    
    # Extract features from all videos
    print(f"\n{'='*80}")
    print(f"EXTRACTING FEATURES FROM {len(video_paths)} VIDEOS")
    print(f"{'='*80}\n")
    
    # Prepare directories
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    openface_cache_dir = os.path.join(project_root, 'output', 'openface')  # Where pose-guided videos are cached
    
    if create_visualizations:
        if output_dir is None:
            vis_output_dir = os.path.join(project_root, 'Train_Outputs')  # Default output
        else:
            vis_output_dir = output_dir  # Use provided output directory
        os.makedirs(vis_output_dir, exist_ok=True)
    
    results = []
    for idx, video_path in enumerate(video_paths, 1):
        print(f"[{idx}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")
        try:
            features = extractor.extract_from_video(video_path)
            result = {'video_name': os.path.basename(video_path)}
            result.update({k: v for k, v in features.items() if k != 'metadata'})
            results.append(result)
            
            # Create visualization if requested
            if create_visualizations:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                
                # Find pose-guided video and CSV in cache directory
                pose_guided_video = os.path.join(openface_cache_dir, f"{base_name}_pose_guided.mp4")
                
                # Use glob to find the actual CSV (might be cached with different name)
                csv_pattern = os.path.join(openface_cache_dir, f"{base_name}*_pose_guided.csv")
                csv_files = glob.glob(csv_pattern)
                openface_csv = csv_files[0] if csv_files else os.path.join(openface_cache_dir, f"{base_name}_pose_guided.csv")
                
                if os.path.exists(pose_guided_video) and os.path.exists(openface_csv):
                    # Output visualization to Train_Outputs
                    vis_output = os.path.join(vis_output_dir, f"{base_name}_landmarks_visualized.mp4")
                    create_visualization_video(pose_guided_video, openface_csv, vis_output)
                
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'video_name': os.path.basename(video_path), 'error': str(e)})
    
    df = pd.DataFrame(results)
    
    if rpe_labels_csv and 'error' in df.columns:
        print(f"\n{'='*80}")
        print("IMPUTATION")
        print(f"{'='*80}\n")
        failed_videos = identify_missing_videos(df)
        if len(failed_videos) > 0:
            rpe_df = load_rpe_labels(rpe_labels_csv)
            df, imputation_summary = impute_by_rpe_average(df, rpe_df, failed_videos)
            print(f"\nImputed {len(failed_videos)} videos using RPE-based averaging")
        else:
            print("No failed videos - no imputation needed")
            df['imputed'] = False
    
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to: {output_csv}")
    
    return df

def extract_single_video(video_path, rpe_labels_csv=None, output_csv=None, create_visualization=False):
    """Extract features from a single video - convenience wrapper for demos."""
    return extract_openface_batch([video_path], rpe_labels_csv, output_csv, create_visualizations=create_visualization)

def run(create_visualizations=False, output_dir=None, output_csv=None):
    video_folder = "../lifting_videos/Demo_Videos/"
    
    if output_csv is None:
        output_csv = "../Train_Outputs/openface_features_all.csv"

    openface_df = extract_openface_batch(
        video_folder,
        rpe_labels_csv="../lifting_videos/Augmented/dataset_labelled.csv",
        output_csv=output_csv,
        create_visualizations=create_visualizations,
        output_dir=output_dir
    )

    print(f"\n{'=' * 80}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nTotal videos: {len(openface_df)}")
    print(f"Total features per video: {len(openface_df.columns) - 1}")
    if 'imputed' in openface_df.columns:
        imputed_count = openface_df['imputed'].sum()
        print(f"Imputed videos: {imputed_count}")
        print(f"Original videos: {len(openface_df) - imputed_count}")

    print(f"\nSample features: {list(openface_df.columns)[:10]}...")
    print(f"\nFirst video features:")
    print(openface_df.iloc[0][['video_name', 'detection_rate', 'AU04_max', 'AU12_max']])

if __name__ == "__main__":
    run()