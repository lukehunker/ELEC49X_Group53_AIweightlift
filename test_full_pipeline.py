#!/usr/bin/env python3
"""
Complete End-to-End Pipeline Test

This script tests the entire RPE prediction pipeline:
1. Bar speed feature extraction
2. Facial feature extraction (OpenFace)
3. Posture feature extraction (MMPose)
4. Feature combination and engineering
5. Ensemble model prediction (LGBM + XGBoost)

Usage:
    python test_full_pipeline.py path/to/video.mp4 Squat
    python test_full_pipeline.py  # Uses demo video if available
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rpe_pipeline import RPEPredictor


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_model_loading():
    """Test that all ensemble models load correctly"""
    print_section("STEP 1: Loading Ensemble Models")
    
    predictor = RPEPredictor(verbose=True)
    
    # Check LGBM model
    if predictor.lgbm_model:
        print(f"LGBM model loaded successfully")
        print(f"   Weight: {predictor.ensemble_weights['lgbm']:.0%}")
    else:
        print("LGBM model not loaded")
        return None
    
    # Check XGBoost model
    if predictor.xgb_model:
        print(f"XGBoost model loaded successfully")
        print(f"   Weight: {predictor.ensemble_weights['xgb']:.0%}")
    else:
        print("XGBoost model not loaded")
        return None
    
    # Check scaler
    if predictor.scaler:
        print(f"Feature scaler loaded")
        print(f"   Features: {len(predictor.feature_names)}")
    else:
        print("Scaler not loaded")
        return None
    
    # Check metadata
    if predictor.metadata:
        print(f"Metadata loaded")
        cv_perf = predictor.metadata.get('cv_performance', {})
        if cv_perf:
            print(f"   CV MAE: {cv_perf.get('ensemble_mae', 0):.4f}")
            print(f"   CV R²: {cv_perf.get('ensemble_r2', 0):.4f}")
    
    print("\nAll ensemble components loaded successfully!")
    return predictor


def test_feature_extraction(video_path, movement):
    """Test feature extraction from all three modules"""
    print_section("STEP 2: Feature Extraction")
    
    from Bar_Tracking.extract_features import extract_bar_speed
    from OpenFace.extract_features import extract_facial_features
    from MMPose.extract_features import extract_posture_features
    
    print(f"\n📹 Video: {video_path}")
    print(f"Movement: {movement}\n")
    
    # 1. Bar Speed Tracking
    print("1️⃣  Extracting Bar Speed Features...")
    try:
        bar_features = extract_bar_speed(video_path, movement)
        print(f"   Rep count: {bar_features.get('rep_count', 0)}")
        print(f"   Fatigue: {bar_features.get('fatigue_s', 0):.3f}s")
        print(f"   First rep: {bar_features.get('first_rep_duration_s', 0):.3f}s")
        print(f"   Last rep: {bar_features.get('last_rep_duration_s', 0):.3f}s")
    except Exception as e:
        print(f"   Error: {e}")
        bar_features = None
    
    # 2. Facial Features (OpenFace)
    print("\n2️⃣  Extracting Facial Features (OpenFace)...")
    try:
        facial_features = extract_facial_features(video_path)
        if facial_features:
            # Features are returned directly, not nested
            au_count = sum(1 for k in facial_features.keys() if k.startswith('AU'))
            print(f"   Action Units extracted: {au_count}")
            print(f"   Detection rate: {facial_features.get('detection_rate', 0):.1%}")
            # Get confidence from metadata if available
            metadata = facial_features.get('metadata', {})
            confidence = metadata.get('confidence_mean', 0) if isinstance(metadata, dict) else 0
            print(f"   Confidence: {confidence:.3f}")
        else:
            print(f"   No facial features extracted")
    except Exception as e:
        print(f"   Error: {e}")
        facial_features = None
    
    # 3. Posture Features (MMPose)
    print("\n3️⃣  Extracting Posture Features (MMPose)...")
    try:
        posture_features = extract_posture_features(video_path, movement)
        print(f"   D-metric: {posture_features.get('d_value', 0):.4f}")
        print(f"   Body tracking successful")
    except Exception as e:
        print(f"   Error: {e}")
        posture_features = None
    
    return {
        'bar_speed': bar_features,
        'facial': facial_features,
        'posture': posture_features
    }


def test_prediction(predictor, video_path, movement):
    """Test the complete prediction pipeline"""
    print_section("STEP 3: Making Prediction")
    
    print("Running complete pipeline...")
    print("(This combines all features and runs through ensemble model)\n")
    
    try:
        result = predictor.predict(video_path, movement=movement)
        
        if result['success']:
            print("PREDICTION SUCCESSFUL!\n")
            print(f"Results:")
            print(f"   Predicted RPE: {result['predicted_rpe']:.1f}")
            print(f"   Confidence: {result['confidence']:.1%}")
            
            # Show model details
            model_details = result.get('model_details', {})
            if model_details:
                print(f"\n🤖 Ensemble Breakdown:")
                print(f"   LGBM prediction: {model_details.get('lgbm_prediction', 0):.2f} (60% weight)")
                print(f"   XGBoost prediction: {model_details.get('xgb_prediction', 0):.2f} (40% weight)")
                print(f"   Final ensemble: {result['predicted_rpe']:.2f}")
                
                # Model agreement
                lgbm_pred = model_details.get('lgbm_prediction', 0)
                xgb_pred = model_details.get('xgb_prediction', 0)
                diff = abs(lgbm_pred - xgb_pred)
                agreement = "High" if diff < 0.5 else "Medium" if diff < 1.0 else "Low"
                print(f"   Model agreement: {agreement} (diff: {diff:.2f})")
            
            # Show feature summary
            features = result.get('features', {})
            if features:
                print(f"\n📋 Feature Summary:")
                combined = features.get('combined', {})
                print(f"   Total features: {len(combined)}")
                print(f"   Rep count: {combined.get('rep_count', 0)}")
                print(f"   Fatigue: {combined.get('rep_change_s', 0):.3f}s")
                print(f"   Detection rate: {combined.get('detection_rate', 0):.1%}")
                print(f"   D-metric: {combined.get('d_value', 0):.4f}")
            
            # Show warnings
            warnings = result.get('warnings', [])
            if warnings:
                print(f"\nWarnings:")
                for warning in warnings:
                    print(f"   • {warning}")
            else:
                print(f"\nNo warnings")
            
            return result
        else:
            print("PREDICTION FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_demo_video():
    """Find a demo video to use for testing"""
    demo_paths = [
        Path('lifting_videos/Demo_Videos'),
        Path('examples'),
        Path('data')
    ]
    
    for demo_path in demo_paths:
        if demo_path.exists():
            videos = list(demo_path.glob('*.mp4')) + list(demo_path.glob('*.mov'))
            if videos:
                return videos[0]
    
    return None


def main():
    """Run complete pipeline test"""
    
    # Parse arguments
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
        movement = sys.argv[2] if len(sys.argv) >= 3 else 'Squat'
    else:
        # Try to find demo video
        demo_video = find_demo_video()
        if demo_video:
            video_path = str(demo_video)
            movement = 'Squat'
            print(f"\n📹 Using demo video: {demo_video.name}")
            print(f"   (You can specify a video: python {sys.argv[0]} path/to/video.mp4 Squat)")
        else:
            print(f"\nNo video specified and no demo video found")
            print(f"\nUsage:")
            print(f"   python {sys.argv[0]} path/to/video.mp4 Squat")
            print(f"   python {sys.argv[0]} path/to/video.mp4 'Bench Press'")
            print(f"   python {sys.argv[0]} path/to/video.mp4 Deadlift")
            return 1
    
    # Check video exists
    if not os.path.exists(video_path):
        print(f"\nVideo not found: {video_path}")
        return 1
    
    # Validate movement type
    valid_movements = ['Bench Press', 'Squat', 'Deadlift']
    if movement not in valid_movements:
        print(f"\nWarning: '{movement}' not in standard movements: {valid_movements}")
        print(f"   Continuing anyway...\n")
    
    print("\n" + "="*70)
    print("  COMPLETE RPE PREDICTION PIPELINE TEST")
    print("="*70)
    print(f"\n📹 Video: {video_path}")
    print(f"Movement: {movement}")
    
    # Step 1: Test model loading
    predictor = test_model_loading()
    if not predictor:
        print("\nModel loading failed. Run training first:")
        print("   python src/LGBM_Regressor/LGBMMod2_Ensemble.py")
        return 1
    
    # Step 2: Test individual feature extraction
    print("\n⏳ This will take 3-5 minutes (feature extraction is slow)...")
    input("\nPress Enter to continue with feature extraction...")
    
    features = test_feature_extraction(video_path, movement)
    
    # Check if all features extracted
    success_count = sum(1 for v in features.values() if v is not None)
    print(f"\nFeature extraction: {success_count}/3 modules successful")
    
    if success_count == 0:
        print("\nNo features extracted. Cannot make prediction.")
        return 1
    
    # Step 3: Test complete prediction (this will re-run feature extraction)
    print("\n⏳ Running complete pipeline with ensemble prediction...")
    input("\nPress Enter to continue with prediction...")
    
    result = test_prediction(predictor, video_path, movement)
    
    if result:
        print_section("TEST COMPLETE")
        print("\nAll pipeline steps completed successfully!")
        print(f"\nFinal Result: RPE {result['predicted_rpe']:.1f} (Confidence: {result['confidence']:.0%})")
        
        # Save detailed results
        output_file = Path('test_pipeline_result.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
        
        return 0
    else:
        print_section("TEST FAILED")
        print("\nPipeline test failed. See errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
