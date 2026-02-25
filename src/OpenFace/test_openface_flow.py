#!/usr/bin/env python3
"""
OpenFace Feature Extraction Pipeline

Complete automated pipeline:
1. Video input (recursive directory search)
2. MediaPipe pose-guided face cropping
3. OpenFace feature extraction  
4. Extract max values from important AU columns
5. Automatic imputation for failed videos using RPE averaging

Outputs:
- output/openface_max_features.csv (raw features, may have failed videos)
- output/openface_max_features_imputed.csv (complete dataset with imputation)

Usage:
    python test_openface_flow.py                          # Process all videos in Augmented_h264
    python test_openface_flow.py --max-videos 50          # Limit to 50 videos
    python test_openface_flow.py --video-dir path/to/vids # Custom directory
    python test_openface_flow.py --visualize              # Show video with OpenFace landmarks
    python test_openface_flow.py -v --max-videos 1        # Visualize first video only
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OpenFace.openface_feature_extractor import OpenFaceExtractor
import OpenFace.openface_utils as ofu
from impute_missing_features import load_rpe_labels, impute_by_rpe_average, identify_missing_videos
import cv2
import numpy as np


def create_visualization_video(video_path, csv_path, output_path):
    """
    Create a visualization video with OpenFace landmarks overlaid.
    Standalone function - doesn't affect the core extractor.
    
    Args:
        video_path: Path to input video (pose-guided video)
        csv_path: Path to OpenFace CSV output with landmarks
        output_path: Path to save visualization video
    """
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATION VIDEO")
    print(f"{'='*80}")
    
    # Load landmark data
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input: {os.path.basename(video_path)}")
    print(f"Frames: {total_frames}, FPS: {fps:.1f}, Resolution: {width}x{height}")
    print(f"Output: {output_path}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"ERROR: Could not create output video writer")
        cap.release()
        return None
    
    frame_idx = 0
    frames_with_landmarks = 0
    
    print("\nProcessing frames...")
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
        
        # Progress indicator
        if frame_idx % 30 == 0:
            print(".", end="", flush=True)
        
        frame_idx += 1
    
    cap.release()
    writer.release()
    
    print(f"\n\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed: {frame_idx} frames")
    print(f"Landmarks drawn: {frames_with_landmarks} frames")
    print(f"Saved to: {output_path}")
    print(f"\nDownload this file to view the visualization!")
    
    return output_path


def test_single_video(video_path, verbose=True, visualize=False):
    """Extract features from a single video."""
    extractor = OpenFaceExtractor(
        use_pose_guidance=True,    # MediaPipe face cropping
        max_only=True,              # Extract max values only
        load_minimal_columns=True,  # Load only important AU columns
        sample_fps=10,              # Process every 3rd frame for speed
        verbose=verbose,
        visualize=False             # Don't use built-in visualization
    )
    
    try:
        features = extractor.extract_from_video(video_path)
        
        if verbose:
            metadata = features.get('metadata', {})
            detection_rate = features.get('detection_rate', 0)
            detection_count = features.get('detection_count', 0)
            total_frames = metadata.get('total_frames', 0)
            
            print(f"\nSUCCESS: {os.path.basename(video_path)}")
            print(f"  Detection: {detection_count}/{total_frames} frames ({detection_rate*100:.1f}%)")
            print(f"  Confidence: {features.get('confidence_mean', 0):.3f}")
            print(f"  Sample AUs: AU04={features.get('AU04_max', 0):.3f}, AU12={features.get('AU12_max', 0):.3f}")
        
        # Create standalone visualization if requested
        if visualize:
            # Find the pose-guided video and CSV that were created
            # Look in output/openface for files matching this video
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            openface_output = os.path.join(project_root, 'output', 'openface')
            
            pose_guided_video = os.path.join(openface_output, f"{base_name}_pose_guided.mp4")
            
            # Find the actual CSV using glob since it might have cached a different file
            import glob
            csv_pattern = os.path.join(openface_output, f"{base_name}*_pose_guided.csv")
            csv_files = glob.glob(csv_pattern)
            
            if csv_files:
                openface_csv = csv_files[0]  # Use the first match
            else:
                openface_csv = os.path.join(openface_output, f"{base_name}_pose_guided.csv")
            
            if os.path.exists(pose_guided_video) and os.path.exists(openface_csv):
                vis_output = os.path.join(openface_output, f"{base_name}_landmarks_visualized.mp4")
                create_visualization_video(pose_guided_video, openface_csv, vis_output)
            else:
                print(f"\nWARNING: Could not create visualization")
                print(f"  Pose-guided video: {pose_guided_video} (exists: {os.path.exists(pose_guided_video)})")
                print(f"  OpenFace CSV: {openface_csv} (exists: {os.path.exists(openface_csv)})")
        
        # Return all features (excluding metadata) for CSV export
        result = {'video_name': os.path.basename(video_path)}
        for key, value in features.items():
            if key != 'metadata':
                result[key] = value
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"\nERROR: {e}")
        
        # Return error row for CSV
        return {
            'video_name': os.path.basename(video_path),
            'error': str(e)
        }


def test_video_directory(video_dir, max_videos=None, pattern='*.mp4', recursive=True, visualize=False):
    """Process all videos in a directory and extract features."""
    # Find all video files
    if recursive:
        patterns = [pattern, '*.mov', '*.MOV', '*.MP4']
        video_files = []
        for pat in patterns:
            video_files.extend(glob.glob(os.path.join(video_dir, '**', pat), recursive=True))
    else:
        patterns = [pattern, '*.mov', '*.MOV', '*.MP4']
        video_files = []
        for pat in patterns:
            video_files.extend(glob.glob(os.path.join(video_dir, pat)))
    
    video_files = sorted(list(set(video_files)))  # Remove duplicates and sort
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    if len(video_files) == 0:
        print(f"ERROR: No videos found in {video_dir}")
        return None
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(video_files)} VIDEOS")
    print(f"{'='*80}")
    
    results = []
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
        print(f"{'='*80}")
        result = test_single_video(video_path, verbose=True, visualize=visualize)
        results.append(result)
    
    # Create DataFrame with all features per video
    df = pd.DataFrame(results)
    
    # Save raw results to output folder
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(project_root, 'output', 'openface_max_features.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    
    has_errors = 'error' in df.columns and df['error'].notna().any()
    successful = len(df) if not has_errors else len(df[df['error'].isna()])
    failed = 0 if not has_errors else len(df[df['error'].notna()])
    
    print(f"\nTotal Videos: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    print(f"\nRaw features saved to: {output_path}")
    
    return output_path


def run_imputation(features_csv, rpe_labels_csv):
    """Automatically impute missing features using RPE-based averaging."""
    print(f"\n{'='*80}")
    print("AUTOMATIC IMPUTATION")
    print(f"{'='*80}")
    
    if not os.path.exists(rpe_labels_csv):
        print(f"WARNING: RPE labels not found: {rpe_labels_csv}")
        print("  Cannot impute without RPE labels.")
        print(f"  Raw features saved to: {features_csv}")
        return
    
    try:
        print(f"\nLoading features: {os.path.basename(features_csv)}")
        features_df = pd.read_csv(features_csv)
        
        print(f"Loading RPE labels: {os.path.basename(rpe_labels_csv)}")
        rpe_df = load_rpe_labels(rpe_labels_csv)
        
        # Check for missing videos (those with error column)
        if 'error' not in features_df.columns:
            print("\nAll videos processed successfully - no imputation needed!")
            # Copy to imputed file anyway for consistency
            output_path = features_csv.replace('.csv', '_imputed.csv')
            features_df['imputed'] = False
            features_df.to_csv(output_path, index=False)
            print(f"Complete dataset saved to: {output_path}")
            return
        
        missing_videos = identify_missing_videos(features_df)
        
        if len(missing_videos) == 0:
            print("\nAll videos processed successfully - no imputation needed!")
            output_path = features_csv.replace('.csv', '_imputed.csv')
            features_df['imputed'] = False
            features_df.to_csv(output_path, index=False)
            print(f"Complete dataset saved to: {output_path}")
            return
        
        # Impute missing features
        print(f"\nImputing {len(missing_videos)} failed videos using RPE averaging...")
        imputed_df, summary_df = impute_by_rpe_average(features_df, rpe_df, missing_videos)
        
        # Save imputed version
        output_path = features_csv.replace('.csv', '_imputed.csv')
        imputed_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*80}")
        print("IMPUTATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nComplete dataset saved to: {output_path}")
        print(f"\nDataset Composition:")
        print(f"  Successfully processed: {len(features_df) - len(missing_videos)} videos")
        print(f"  Imputed (failed): {len(missing_videos)} videos")
        print(f"  Total: {len(imputed_df)} videos")
        print(f"\nReady for LGBM training!")
        
    except Exception as e:
        print(f"\nERROR during imputation: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nRaw features still available at: {features_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='OpenFace Feature Extraction Pipeline with Automatic Imputation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in Augmented_h264 (default)
  python test_openface_flow.py
  
  # Process first 50 videos
  python test_openface_flow.py --max-videos 50
  
  # Process custom directory
  python test_openface_flow.py --video-dir ../../path/to/videos
  
Outputs:
  output/openface_max_features.csv          - Raw features (may have failed videos)
  output/openface_max_features_imputed.csv  - Complete with imputation (ready for training)
        """
    )
    
    parser.add_argument('--video-dir', 
                       default='../../lifting_videos/Augmented_h264',
                       help='Directory containing videos (default: ../../lifting_videos/Augmented_h264)')
    parser.add_argument('--max-videos', type=int, 
                       help='Maximum number of videos to process')
    parser.add_argument('--pattern', default='*.mp4', 
                       help='Video file pattern (default: *.mp4)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       default=True,
                       help='Search recursively in subdirectories (default: True)')
    parser.add_argument('--rpe-labels', 
                       default='../../lifting_videos/Augmented/dataset_labelled.csv',
                       help='Path to RPE labels CSV for imputation (default: ../../lifting_videos/Augmented/dataset_labelled.csv)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show video with OpenFace landmarks drawn in real-time')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("OPENFACE FEATURE EXTRACTION PIPELINE")
    print(f"{'='*80}")
    print("\nPipeline Steps:")
    print("  1. Video input (recursive directory search)")
    print("  2. MediaPipe pose-guided face cropping")
    print("  3. OpenFace feature extraction")
    print("  4. Load important AU columns only (10 AUs)")
    print("  5. Extract maximum values from each AU")
    print("  6. Automatic imputation for failed videos")
    print(f"{'='*80}")
    
    if not os.path.exists(args.video_dir):
        print(f"ERROR: Path not found: {args.video_dir}")
        return
    
    # Check if path is a file or directory
    if os.path.isfile(args.video_dir):
        # Single file mode
        print(f"\nProcessing single video file: {os.path.basename(args.video_dir)}\n")
        result = test_single_video(args.video_dir, verbose=True, visualize=args.visualize)
        
        # Save to CSV
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path = os.path.join(project_root, 'output', 'openface_max_features.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df = pd.DataFrame([result])
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*80}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"Features saved to: {output_path}")
        
        # Run imputation
        run_imputation(output_path, args.rpe_labels)
    else:
        # Directory mode (original behavior)
        output_csv = test_video_directory(args.video_dir, args.max_videos, args.pattern, args.recursive, args.visualize)
        
        # Always run imputation
        if output_csv:
            run_imputation(output_csv, args.rpe_labels)


if __name__ == '__main__':
    main()
