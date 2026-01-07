#!/usr/bin/env python3
"""
Run RPE Prediction Pipeline

Processes videos and predicts RPE scores using trained LGBM model.

Usage Examples:

    # Predict RPE for all videos in a folder:
    python run_rpe_prediction.py videos/ --model models/rpe_model.pkl --output results.csv
    
    # Predict for single video:
    python run_rpe_prediction.py videos/workout1.mp4 --model models/rpe_model.pkl
    
    # Extract features only (no prediction):
    python run_rpe_prediction.py videos/ --extract-only --output features.csv
    
    # With expected reps for validation:
    python run_rpe_prediction.py videos/ --model models/rpe_model.pkl --reps 10
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_pipeline import RPEPipeline


def predict_single_video(pipeline, video_path, expected_reps=None):
    """Predict RPE for a single video."""
    print(f"\n{'='*80}")
    print(f"VIDEO: {os.path.basename(video_path)}")
    print(f"{'='*80}")
    
    try:
        rpe = pipeline.predict_rpe(video_path, expected_reps=expected_reps)
        print(f"\n✓ Predicted RPE: {rpe:.2f}")
        
        return {
            'video_name': Path(video_path).stem,
            'video_path': str(video_path),
            'predicted_rpe': rpe,
            'expected_reps': expected_reps
        }
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return {
            'video_name': Path(video_path).stem,
            'video_path': str(video_path),
            'predicted_rpe': None,
            'error': str(e)
        }


def predict_batch(pipeline, video_dir, expected_reps=None):
    """Predict RPE for all videos in a directory."""
    video_files = sorted(Path(video_dir).glob('*.mp4'))
    
    if not video_files:
        print(f"ERROR: No .mp4 files found in {video_dir}")
        return []
    
    print(f"\nFound {len(video_files)} videos to process\n")
    
    results = []
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing...")
        
        result = predict_single_video(pipeline, str(video_path), expected_reps)
        results.append(result)
    
    return results


def extract_features_only(pipeline, video_dir, output_csv):
    """Extract features without prediction (for training data)."""
    print(f"\nExtracting features from all videos in: {video_dir}")
    print("(No RPE prediction - features only)\n")
    
    df = pipeline.extract_features_batch(video_dir, output_csv=output_csv)
    
    print(f"\n{'='*80}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Videos processed: {len(df)}")
    print(f"Features extracted: {len(df.columns) - 1}")  # -1 for video_name
    print(f"Output saved to: {output_csv}")
    
    return df


def load_reps_file(reps_file):
    """Load expected reps from JSON file."""
    with open(reps_file, 'r') as f:
        reps_data = json.load(f)
    return reps_data


def main():
    parser = argparse.ArgumentParser(
        description='Run RPE prediction pipeline on videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Predict RPE for all videos in a folder:
  python run_rpe_prediction.py videos/ --model models/rpe_model.pkl --output results.csv
  
  # Predict for single video:
  python run_rpe_prediction.py videos/workout1.mp4 --model models/rpe_model.pkl
  
  # Extract features only (for training):
  python run_rpe_prediction.py videos/ --extract-only --output features.csv
  
  # With expected reps (single value for all):
  python run_rpe_prediction.py videos/ --model models/rpe_model.pkl --reps 10
  
  # With per-video reps (from JSON file):
  python run_rpe_prediction.py videos/ --model models/rpe_model.pkl --reps-file reps.json
  
  Example reps.json format:
  {
    "workout1": 10,
    "workout2": 8,
    "workout3": 12
  }
        """
    )
    
    # Required arguments
    parser.add_argument('input', help='Input video file or directory')
    
    # Model arguments
    parser.add_argument('--model', '-m', help='Path to trained LGBM model (.pkl)')
    parser.add_argument('--extract-only', action='store_true',
                       help='Only extract features, no prediction')
    
    # Output arguments
    parser.add_argument('--output', '-o', help='Output CSV file for results/features')
    
    # Reps arguments
    parser.add_argument('--reps', type=int, help='Expected reps (same for all videos)')
    parser.add_argument('--reps-file', help='JSON file with per-video expected reps')
    
    # Processing arguments
    parser.add_argument('--fps', type=int, default=10,
                       help='Target FPS for OpenFace sampling (default: 10)')
    parser.add_argument('--no-preprocess', action='store_true',
                       help='Skip video preprocessing')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.extract_only and not args.model:
        parser.error("--model required unless --extract-only is specified")
    
    if not os.path.exists(args.input):
        parser.error(f"Input not found: {args.input}")
    
    # Load reps data if provided
    expected_reps = None
    if args.reps_file:
        expected_reps = load_reps_file(args.reps_file)
        print(f"Loaded expected reps from: {args.reps_file}")
    elif args.reps:
        expected_reps = args.reps
        print(f"Using expected reps: {expected_reps} (for all videos)")
    
    # Initialize pipeline
    print("\n" + "="*80)
    print("RPE PREDICTION PIPELINE")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Model: {args.model if args.model else 'N/A (feature extraction only)'}")
    print(f"OpenFace FPS: {args.fps}")
    print("="*80 + "\n")
    
    pipeline = RPEPipeline(
        lgbm_model_path=args.model if not args.extract_only else None,
        sample_fps=args.fps
    )
    
    # Determine if input is file or directory
    input_path = Path(args.input)
    
    if args.extract_only:
        # Feature extraction only
        if not input_path.is_dir():
            parser.error("--extract-only requires a directory, not a single file")
        
        output_csv = args.output or 'output/features.csv'
        extract_features_only(pipeline, str(input_path), output_csv)
    
    elif input_path.is_file():
        # Single video prediction
        result = predict_single_video(
            pipeline, 
            str(input_path), 
            expected_reps=expected_reps
        )
        
        # Save result if output specified
        if args.output:
            df = pd.DataFrame([result])
            df.to_csv(args.output, index=False)
            print(f"\n✓ Result saved to: {args.output}")
    
    else:
        # Batch prediction
        results = predict_batch(pipeline, str(input_path), expected_reps=expected_reps)
        
        # Save results
        output_csv = args.output or 'output/rpe_predictions.csv'
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"BATCH PREDICTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total videos: {len(results)}")
        print(f"Successful: {len([r for r in results if r.get('predicted_rpe') is not None])}")
        print(f"Failed: {len([r for r in results if r.get('predicted_rpe') is None])}")
        
        if df['predicted_rpe'].notna().any():
            print(f"\nRPE Statistics:")
            print(f"  Mean: {df['predicted_rpe'].mean():.2f}")
            print(f"  Std:  {df['predicted_rpe'].std():.2f}")
            print(f"  Min:  {df['predicted_rpe'].min():.2f}")
            print(f"  Max:  {df['predicted_rpe'].max():.2f}")
        
        print(f"\n✓ Results saved to: {output_csv}")


if __name__ == '__main__':
    main()
