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


def test_single_video(video_path, verbose=True, visualize=False):
    """Extract features from a single video."""
    extractor = OpenFaceExtractor(
        use_pose_guidance=True,    # MediaPipe face cropping
        max_only=True,              # Extract max values only
        load_minimal_columns=True,  # Load only important AU columns
        sample_fps=10,              # Process every 3rd frame for speed
        verbose=verbose,
        visualize=visualize         # Show video with landmarks
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
        print(f"ERROR: Directory not found: {args.video_dir}")
        return
    
    # Extract features from all videos
    output_csv = test_video_directory(args.video_dir, args.max_videos, args.pattern, args.recursive, args.visualize)
    
    # Always run imputation
    if output_csv:
        run_imputation(output_csv, args.rpe_labels)


if __name__ == '__main__':
    main()
