"""
Example usage of the OpenFace Feature Extractor

This script demonstrates how to use the production OpenFace module
to extract features from exercise videos for LGBM regression.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from openface_feature_extractor import OpenFaceExtractor, extract_features, extract_features_batch

# Get repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VIDEOS_DIR = os.path.join(REPO_ROOT, 'videos')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')


def example_single_video():
    """Example: Extract features from a single video."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Video Feature Extraction")
    print("="*60)
    
    # Find a video
    videos = [f for f in os.listdir(VIDEOS_DIR) 
              if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        print("No videos found in videos/ directory")
        return None
    
    video_path = os.path.join(VIDEOS_DIR, videos[0])
    
    # Extract features
    print(f"\nExtracting features from: {videos[0]}")
    features = extract_features(video_path, expected_reps=5, verbose=True)
    
    # Show some key features
    print("\n" + "-"*60)
    print("Key Features Extracted:")
    print("-"*60)
    print(f"Detection Rate: {features['detection_rate']:.2%}")
    print(f"Detected Reps: {features['detected_reps']}")
    print(f"Rep Consistency: {features['rep_consistency']:.3f}")
    print(f"AU Overall Mean: {features['au_overall_mean']:.3f}")
    print(f"AU Highly Active Count: {features['au_highly_active_count']}")
    
    # Show AU04 (Brow Lowerer) features
    au04_features = {k: v for k, v in features.items() if 'AU04' in k and k != 'metadata'}
    print(f"\nAU04 (Brow Lowerer) Features:")
    for k, v in sorted(au04_features.items())[:5]:
        print(f"  {k}: {v:.3f}")
    
    return features


def example_batch_extraction():
    """Example: Extract features from all videos in directory."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Feature Extraction")
    print("="*60)
    
    # Create output CSV path
    output_csv = os.path.join(OUTPUT_DIR, 'training_features.csv')
    
    # Extract from all videos
    print(f"\nExtracting features from all videos in: {VIDEOS_DIR}")
    
    # Define expected reps for each video (if known)
    # This can be a dictionary mapping video names to rep counts
    expected_reps = {
        # Example: 'squat_low_3reps': 3,
        # Or just use a default for all: 5
    }
    
    try:
        df = extract_features_batch(
            VIDEOS_DIR, 
            output_csv=output_csv,
            expected_reps=None,  # Or use expected_reps dict
            verbose=True
        )
        
        print("\n" + "-"*60)
        print("Batch Extraction Results:")
        print("-"*60)
        print(f"Videos Processed: {len(df)}")
        print(f"Features per Video: {len(df.columns) - 1}")
        print(f"Output CSV: {output_csv}")
        
        print("\nFirst 3 videos:")
        print(df[['video_name', 'detection_rate', 'detected_reps', 'au_overall_mean']].head(3))
        
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def example_prepare_for_lgbm():
    """Example: Prepare features for LGBM training."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Prepare Features for LGBM")
    print("="*60)
    
    # Extract features
    extractor = OpenFaceExtractor(verbose=False)
    
    videos = [f for f in os.listdir(VIDEOS_DIR) 
              if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if len(videos) < 2:
        print("Need at least 2 videos for this example")
        return
    
    # Simulate extracting features from multiple videos
    all_features = []
    for video_file in videos[:3]:  # Use first 3 videos
        video_path = os.path.join(VIDEOS_DIR, video_file)
        try:
            features = extractor.extract_from_video(video_path)
            
            # Flatten features
            flat_features = {'video_name': video_file}
            for key, value in features.items():
                if key != 'metadata':
                    flat_features[key] = value
            
            all_features.append(flat_features)
        except Exception as e:
            print(f"Skipping {video_file}: {e}")
    
    if not all_features:
        print("No features extracted")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    print(f"\nExtracted features from {len(df)} videos")
    print(f"Feature columns: {len(df.columns) - 1}")
    
    # Prepare for LGBM
    print("\n" + "-"*60)
    print("Preparing for LGBM Model:")
    print("-"*60)
    
    # Feature matrix (X)
    feature_cols = [col for col in df.columns if col != 'video_name']
    X = df[feature_cols].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Sample feature vector (first video, first 10 features):")
    print(X[0, :10])
    
    # In real usage, you would also have target values (y)
    # For example, RPE (Rate of Perceived Exertion) labels
    # y = np.array([3.5, 7.2, 5.8])  # RPE scores
    
    print("\nTo train LGBM:")
    print("  import lightgbm as lgb")
    print("  model = lgb.LGBMRegressor()")
    print("  model.fit(X_train, y_train)")
    print("  predictions = model.predict(X_test)")
    
    return df


def example_custom_extractor():
    """Example: Using OpenFaceExtractor class with custom settings."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Extractor Usage")
    print("="*60)
    
    # Create extractor instance
    extractor = OpenFaceExtractor(verbose=True)
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    print(f"\nTotal features available: {len(feature_names)}")
    
    # Show feature categories
    print("\nFeature categories:")
    categories = {}
    for fname in feature_names:
        if fname.startswith('AU'):
            cat = 'Action Units'
        elif fname.startswith('rep_'):
            cat = 'Repetition'
        elif fname.startswith('detection') or fname.startswith('confidence'):
            cat = 'Detection Quality'
        elif fname.startswith('landmark') or fname.startswith('head'):
            cat = 'Landmark/Head'
        elif 'velocity' in fname or 'change' in fname:
            cat = 'Temporal Dynamics'
        else:
            cat = 'Other'
        
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} features")
    
    # Process a video if available
    videos = [f for f in os.listdir(VIDEOS_DIR) 
              if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if videos:
        video_path = os.path.join(VIDEOS_DIR, videos[0])
        print(f"\nProcessing: {videos[0]}")
        
        features = extractor.extract_from_video(video_path, expected_reps=5)
        
        # Show metadata
        print("\nVideo Metadata:")
        for key, value in features['metadata'].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OpenFace Feature Extractor - Usage Examples")
    print("="*80)
    
    # Run examples
    try:
        # Example 1: Single video
        example_single_video()
        
        # Example 2: Batch extraction
        # example_batch_extraction()
        
        # Example 3: LGBM preparation
        # example_prepare_for_lgbm()
        
        # Example 4: Custom usage
        # example_custom_extractor()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")
