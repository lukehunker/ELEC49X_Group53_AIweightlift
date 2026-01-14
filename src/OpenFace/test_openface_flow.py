#!/usr/bin/env python3
"""
Test OpenFace Feature Extraction Flow

Tests the complete flow:
1. Video input
2. MediaPipe pose-guided face cropping
3. OpenFace processes video → CSV
4. Load only important AU columns
5. Extract maximum values from those columns

Usage:
    python test_openface_flow.py
    python test_openface_flow.py --video path/to/video.mp4
    python test_openface_flow.py --video-dir path/to/videos/ --max-videos 5
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


def test_single_video(video_path, verbose=True):
    """Test OpenFace flow on a single video."""
    print(f"\n{'='*80}")
    print(f"TESTING: {os.path.basename(video_path)}")
    print(f"{'='*80}")
    
    # Initialize extractor with max_only mode
    extractor = OpenFaceExtractor(
        use_pose_guidance=True,    # Step 1: MediaPipe cropping
        max_only=True,              # Step 4: Extract max values only
        load_minimal_columns=True,  # Step 3: Load only important columns
        sample_fps=10,              # Optional: sample frames for speed
        verbose=verbose
    )
    
    try:
        # Extract features (runs the full flow)
        features = extractor.extract_from_video(video_path)
        
        # Show the results
        print(f"\n{'='*80}")
        print("EXTRACTED FEATURES (MAX VALUES)")
        print(f"{'='*80}")
        
        # Detection quality
        print(f"\nDetection Quality:")
        print(f"  Detection Rate: {features.get('detection_rate', 0)*100:.1f}%")
        print(f"  Confidence Mean: {features.get('confidence_mean', 0):.3f}")
        
        # AU Maximum Intensities (the key features for RPE)
        print(f"\nAction Unit Maximum Intensities:")
        au_features = {k: v for k, v in features.items() 
                      if k.startswith('AU') and '_max' in k}
        
        for au_name in sorted(au_features.keys()):
            print(f"  {au_name}: {au_features[au_name]:.3f}")
        
        # Peak ratios
        print(f"\nPeak-to-Baseline Ratios:")
        ratio_features = {k: v for k, v in features.items() 
                         if k.startswith('AU') and '_peak_ratio' in k}
        
        for ratio_name in sorted(ratio_features.keys()):
            print(f"  {ratio_name}: {ratio_features[ratio_name]:.3f}")
        
        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Overall Max AU: {features.get('au_overall_max', 0):.3f}")
        print(f"  High Peak Count: {features.get('au_high_peak_count', 0)}")
        
        # Metadata
        metadata = features.get('metadata', {})
        print(f"\nVideo Metadata:")
        print(f"  Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
        print(f"  FPS: {metadata.get('fps', 0):.1f}")
        print(f"  Valid Frames: {metadata.get('valid_frames', 0)}/{metadata.get('total_frames', 0)}")
        
        print(f"\n✓ SUCCESS: Flow completed for {os.path.basename(video_path)}")
        
        # Return all features (excluding metadata) for CSV export
        result = {'video_name': os.path.basename(video_path)}
        for key, value in features.items():
            if key != 'metadata':  # Skip metadata dict
                result[key] = value
        
        return result
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error row for CSV
        return {
            'video_name': os.path.basename(video_path),
            'error': str(e)
        }


def test_video_directory(video_dir, max_videos=None, pattern='*.mp4', recursive=False):
    """Test OpenFace flow on multiple videos."""
    if recursive:
        # Search recursively in subdirectories (for Augmented structure)
        video_files = []
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith(('.mp4', '.mov', '.MP4', '.MOV')):
                    video_files.append(os.path.join(root, file))
        video_files = sorted(video_files)
    else:
        # Support multiple patterns (e.g., *.mp4 and *.mov)
        if pattern == '*.mp4':
            # Default: search for both mp4 and mov
            video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
            video_files += glob.glob(os.path.join(video_dir, '*.mov'))
            video_files += glob.glob(os.path.join(video_dir, '*.MP4'))
            video_files += glob.glob(os.path.join(video_dir, '*.MOV'))
            video_files = sorted(video_files)
        else:
            video_files = sorted(glob.glob(os.path.join(video_dir, pattern)))
    
    if not video_files:
        print(f"ERROR: No videos found matching {pattern} in {video_dir}")
        return
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    print(f"\n{'='*80}")
    print(f"BATCH TEST: {len(video_files)} videos")
    print(f"{'='*80}")
    
    results = []
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}]")
        result = test_single_video(video_path, verbose=False)
        results.append(result)
    
    # Create DataFrame with all features per video
    df = pd.DataFrame(results)
    
    # Save detailed results (each row = video, columns = all max features)
    # Save to project output folder (not src/output)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(project_root, 'output', 'openface_max_features.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH TEST COMPLETE")
    print(f"{'='*80}")
    
    successful = len(df[~df['video_name'].str.contains('error', na=False)])
    failed = len(df) - successful
    
    print(f"\nTotal Videos: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print(f"\nFeatures Extracted Per Video: {len(df.columns) - 1}")  # -1 for video_name
        
        # Show sample of key features
        au_max_cols = [col for col in df.columns if col.startswith('AU') and '_max' in col]
        if au_max_cols:
            print(f"\nSample AU Max Values (first video):")
            for col in sorted(au_max_cols)[:5]:
                val = df[col].iloc[0] if not pd.isna(df[col].iloc[0]) else 0
                print(f"  {col}: {val:.3f}")
    
    print(f"\n✓ Detailed results saved to: {output_path}")
    print(f"  Format: Each row = video, Each column = feature max value")


def verify_csv_columns(video_path):
    """Verify the CSV has correct columns after OpenFace processing."""
    print(f"\n{'='*80}")
    print("CSV COLUMN VERIFICATION")
    print(f"{'='*80}")
    
    # Run OpenFace to get CSV
    csv_path = ofu.run_openface(video_path, use_pose_guidance=True)
    
    print(f"\nOpenFace CSV: {csv_path}")
    
    # Load full CSV
    df_full = pd.read_csv(csv_path)
    print(f"\nFull CSV Columns: {len(df_full.columns)}")
    print(f"Sample columns: {list(df_full.columns[:10])}")
    
    # Load with minimal columns filter
    df_minimal = ofu.load_landmark_data(csv_path, columns_only='rpe_minimal')
    print(f"\nMinimal Columns Loaded: {len(df_minimal.columns)}")
    print(f"Columns: {list(df_minimal.columns)}")
    
    # Check for important AUs
    important_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 
                     'AU12_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r']
    
    print(f"\nImportant AU Columns Present:")
    for au in important_aus:
        present = au in df_minimal.columns
        symbol = "✓" if present else "✗"
        print(f"  {symbol} {au}")
    
    # Show sample max values
    if all(au in df_minimal.columns for au in important_aus):
        print(f"\nSample Maximum Values:")
        for au in important_aus:
            max_val = df_minimal[au].max()
            print(f"  {au}: {max_val:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Test OpenFace feature extraction flow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single video
  python test_openface_flow.py --video ../../videos/workout1.mp4
  
  # Test all videos in Augmented directory (recursive)
  python test_openface_flow.py --video-dir ../../lifting_videos/Augmented --recursive
  
  # Test first 10 videos from Augmented
  python test_openface_flow.py --video-dir ../../lifting_videos/Augmented --recursive --max-videos 10
  
  # Verify CSV columns
  python test_openface_flow.py --video ../../videos/workout1.mp4 --verify-csv
        """
    )
    
    parser.add_argument('--video', help='Path to single video file')
    parser.add_argument('--video-dir', help='Directory containing videos')
    parser.add_argument('--max-videos', type=int, help='Maximum videos to test')
    parser.add_argument('--pattern', default='*.mp4', help='Video file pattern (default: *.mp4, also searches *.mov)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search recursively in subdirectories (for Augmented structure)')
    parser.add_argument('--verify-csv', action='store_true', 
                       help='Verify CSV columns (only with --video)')
    
    args = parser.parse_args()
    
    if not args.video and not args.video_dir:
        parser.error("Must provide either --video or --video-dir")
    
    print(f"\n{'='*80}")
    print("OPENFACE FEATURE EXTRACTION FLOW TEST")
    print(f"{'='*80}")
    print("\nFlow Steps:")
    print("  1. Video input")
    print("  2. MediaPipe pose-guided face cropping")
    print("  3. OpenFace processes video → CSV")
    print("  4. Load only important AU columns (10 AUs)")
    print("  5. Extract maximum values from those columns")
    print(f"{'='*80}")
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"ERROR: Video not found: {args.video}")
            return
        
        if args.verify_csv:
            verify_csv_columns(args.video)
        else:
            test_single_video(args.video)
    
    elif args.video_dir:
        if not os.path.exists(args.video_dir):
            print(f"ERROR: Directory not found: {args.video_dir}")
            return
        
        test_video_directory(args.video_dir, args.max_videos, args.pattern, args.recursive)


if __name__ == '__main__':
    main()
