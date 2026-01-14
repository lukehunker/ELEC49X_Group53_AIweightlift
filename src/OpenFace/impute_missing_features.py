#!/usr/bin/env python3
"""
Impute Missing OpenFace Features Using RPE-Based Averaging

For videos that failed to process (no valid frames), this script fills in their features
by averaging the features from other videos with the same RPE value and exercise type.

Usage:
    python impute_missing_features.py --features openface_max_features.csv --rpe rpe_labels.csv --output openface_max_features_imputed.csv
    
RPE labels CSV format:
    video_name,exercise_type,rpe
    Squat 1.mp4,squat,7
    Squat 10.mp4,squat,8
    ...
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys


def load_rpe_labels(rpe_csv_path):
    """Load RPE labels for each video."""
    df = pd.read_csv(rpe_csv_path)
    
    # Expected columns: Video, RPE (from Dataset_labelled.csv)
    required_cols = ['Video', 'RPE']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"RPE CSV must have column: {col}. Found: {list(df.columns)}")
    
    # Normalize video names to match feature CSV format
    # Dataset_labelled.csv has "Bench Press 1", features CSV has "Bench Press 1.mp4"
    df['video_name'] = df['Video'].apply(lambda x: f"{x}.mp4" if not x.endswith('.mp4') else x)
    df['rpe'] = df['RPE']
    
    # Extract exercise type from video name
    def extract_exercise(video_name):
        video_lower = video_name.lower()
        if 'squat' in video_lower:
            return 'squat'
        elif 'deadlift' in video_lower:
            return 'deadlift'
        elif 'bench' in video_lower:
            return 'bench_press'
        else:
            # Try to extract first word(s) before number
            parts = video_name.split()
            if len(parts) >= 2:
                return ' '.join(parts[:-1]).lower()
            return 'unknown'
    
    df['exercise_type'] = df['video_name'].apply(extract_exercise)
    
    return df[['video_name', 'exercise_type', 'rpe']]


def identify_missing_videos(features_df):
    """Identify videos that failed to process."""
    # Videos with error messages
    missing_mask = features_df['error'].notna()
    missing_videos = features_df[missing_mask]['video_name'].tolist()
    
    print(f"\nFound {len(missing_videos)} videos with missing features:")
    for video in missing_videos:
        error = features_df[features_df['video_name'] == video]['error'].iloc[0]
        print(f"  - {video}: {error}")
    
    return missing_videos


def impute_by_rpe_average(features_df, rpe_df, missing_videos):
    """
    Impute missing features using average of videos with same RPE and exercise type.
    
    Strategy:
    1. For each missing video, get its RPE and exercise type
    2. Find all successful videos with same RPE and exercise type
    3. Average their features
    4. Fill in the missing video's features with these averages
    """
    features_df = features_df.copy()
    
    # Get numeric columns (features to impute)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove error column
    numeric_cols = [col for col in numeric_cols if col != 'error']
    
    print(f"\nImputing {len(numeric_cols)} numeric features...")
    
    imputation_summary = []
    
    for video in missing_videos:
        # Get RPE and exercise type for this video
        rpe_row = rpe_df[rpe_df['video_name'] == video]
        
        if len(rpe_row) == 0:
            print(f"  WARNING: {video} not found in RPE labels, skipping")
            continue
        
        video_rpe = rpe_row['rpe'].iloc[0]
        video_exercise = rpe_row['exercise_type'].iloc[0].lower()
        
        print(f"\n  Processing: {video}")
        print(f"    Exercise: {video_exercise}, RPE: {video_rpe}")
        
        # Strategy 1: Find videos with SAME EXERCISE + SAME RPE (most accurate)
        similar_videos_mask = (
            (rpe_df['exercise_type'].str.lower() == video_exercise) &
            (rpe_df['rpe'] == video_rpe) &
            (~rpe_df['video_name'].isin(missing_videos))
        )
        similar_video_names = rpe_df[similar_videos_mask]['video_name'].tolist()
        similar_features = features_df[features_df['video_name'].isin(similar_video_names)]
        imputation_method = f"{video_exercise} + RPE={video_rpe}"
        
        if len(similar_features) < 2:  # Need at least 2 videos for good average
            print(f"    Only {len(similar_features)} video(s) with {video_exercise} RPE={video_rpe}")
            print(f"    Trying: Any exercise with RPE={video_rpe}...")
            
            # Strategy 2: Find videos with SAME RPE (any exercise)
            similar_videos_mask = (
                (rpe_df['rpe'] == video_rpe) &
                (~rpe_df['video_name'].isin(missing_videos))
            )
            similar_video_names = rpe_df[similar_videos_mask]['video_name'].tolist()
            similar_features = features_df[features_df['video_name'].isin(similar_video_names)]
            imputation_method = f"RPE={video_rpe} (any exercise)"
            
            if len(similar_features) == 0:
                print(f"    WARNING: No videos found with RPE={video_rpe}, trying same exercise...")
                
                # Strategy 3: Fallback to same exercise only (ignore RPE)
                similar_videos_mask = (
                    (rpe_df['exercise_type'].str.lower() == video_exercise) &
                    (~rpe_df['video_name'].isin(missing_videos))
                )
                similar_video_names = rpe_df[similar_videos_mask]['video_name'].tolist()
                similar_features = features_df[features_df['video_name'].isin(similar_video_names)]
                imputation_method = f"{video_exercise} (all RPE levels)"
                
                if len(similar_features) == 0:
                    print(f"    ERROR: No videos found for exercise '{video_exercise}', cannot impute")
                    continue
                
                print(f"    Using average of {len(similar_features)} videos from '{video_exercise}' (all RPE levels)")
            else:
                print(f"    Using average of {len(similar_features)} videos with RPE={video_rpe} (any exercise)")
        else:
            print(f"    Using average of {len(similar_features)} videos with {video_exercise} RPE={video_rpe}")
        
        # Calculate mean of features from similar videos
        mean_features = similar_features[numeric_cols].mean()
        
        # Fill in the missing video's features
        video_idx = features_df[features_df['video_name'] == video].index[0]
        for col in numeric_cols:
            features_df.loc[video_idx, col] = mean_features[col]
        
        # Clear error column and mark as imputed
        features_df.loc[video_idx, 'error'] = np.nan
        
        # Add imputation metadata column if it doesn't exist
        if 'imputed' not in features_df.columns:
            features_df['imputed'] = False
        if 'imputation_method' not in features_df.columns:
            features_df['imputation_method'] = ''
        
        features_df.loc[video_idx, 'imputed'] = True
        features_df.loc[video_idx, 'imputation_method'] = imputation_method
        
        # Track imputation
        imputation_summary.append({
            'video': video,
            'exercise': video_exercise,
            'rpe': video_rpe,
            'similar_videos_count': len(similar_features),
            'imputed_features': len(numeric_cols),
            'imputation_method': imputation_method
        })
        
        # Show sample of imputed values
        print(f"    Sample imputed values:")
        print(f"      AU04_max: {mean_features['AU04_max']:.3f}")
        print(f"      AU26_max: {mean_features['AU26_max']:.3f}")
        print(f"      detection_rate: {mean_features['detection_rate']:.3f}")
    
    return features_df, pd.DataFrame(imputation_summary)


def main():
    parser = argparse.ArgumentParser(
        description='Impute missing OpenFace features using RPE-based averaging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python impute_missing_features.py \\
        --features ../../output/openface_max_features.csv \\
        --rpe ../../lifting_videos/Augmented/dataset_labelled.csv \\
        --output ../../output/openface_max_features_imputed.csv

The script will:
1. Load RPE labels from dataset_labelled.csv (Video, RPE columns)
2. Identify videos with insufficient frames (errors in features CSV)
3. For each failed video, find all successful videos with the same RPE
4. Average their features and fill in the missing video's data
        """
    )
    
    parser.add_argument('--features', required=True,
                       help='Path to OpenFace max features CSV (with missing videos)')
    parser.add_argument('--rpe', required=True,
                       help='Path to RPE labels CSV (video_name, exercise_type, rpe)')
    parser.add_argument('--output', required=True,
                       help='Path to save imputed features CSV')
    parser.add_argument('--summary', help='Path to save imputation summary CSV (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.features):
        print(f"ERROR: Features file not found: {args.features}")
        return
    
    if not os.path.exists(args.rpe):
        print(f"ERROR: RPE labels file not found: {args.rpe}")
        return
    
    print("="*80)
    print("OPENFACE FEATURE IMPUTATION")
    print("="*80)
    print(f"\nInput features: {args.features}")
    print(f"RPE labels: {args.rpe}")
    print(f"Output: {args.output}")
    
    # Load data
    print("\nLoading data...")
    features_df = pd.read_csv(args.features)
    rpe_df = load_rpe_labels(args.rpe)
    
    print(f"  Features: {len(features_df)} videos, {len(features_df.columns)} columns")
    print(f"  RPE labels: {len(rpe_df)} videos")
    
    # Identify missing videos
    missing_videos = identify_missing_videos(features_df)
    
    if len(missing_videos) == 0:
        print("\n✓ No missing videos found! All videos processed successfully.")
        return
    
    # Impute missing features
    print("\n" + "="*80)
    print("IMPUTING MISSING FEATURES")
    print("="*80)
    
    imputed_df, summary_df = impute_by_rpe_average(features_df, rpe_df, missing_videos)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    imputed_df.to_csv(args.output, index=False)
    print(f"\n✓ Imputed features saved to: {args.output}")
    
    if args.summary:
        os.makedirs(os.path.dirname(args.summary), exist_ok=True)
        summary_df.to_csv(args.summary, index=False)
        print(f"✓ Imputation summary saved to: {args.summary}")
    
    # Final summary
    print("\n" + "="*80)
    print("IMPUTATION COMPLETE")
    print("="*80)
    print(f"\nFinal Dataset Composition:")
    print(f"  Original successful videos: {len(features_df) - len(missing_videos)}")
    print(f"  Videos with imputed features: {len(summary_df)}")
    print(f"  Total videos in output: {len(imputed_df)}")
    print(f"\nAll rows are included in the final CSV:")
    print(f"  ✓ Successfully processed videos (real features)")
    print(f"  ✓ Imputed videos (generated features from averaging)")
    
    if len(summary_df) > 0:
        print(f"\nImputation breakdown by exercise:")
        for exercise in summary_df['exercise'].unique():
            count = len(summary_df[summary_df['exercise'] == exercise])
            print(f"  {exercise.capitalize()}: {count} videos imputed")
        
        print(f"\nImputation methods used:")
        for method in summary_df['imputation_method'].unique():
            count = len(summary_df[summary_df['imputation_method'] == method])
            print(f"  {method}: {count} videos")
    
    print("\n✓ Ready for LGBM training!")
    print(f"\nNote: Check 'imputed' column in output CSV to identify which videos")
    print(f"      had features generated vs. extracted from actual video analysis.")


if __name__ == '__main__':
    main()
