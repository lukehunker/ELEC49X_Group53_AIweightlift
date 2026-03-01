#!/usr/bin/env python3
"""
OpenFace Single Video Inference

Extracts facial expression features from a SINGLE video.
This is for testing/app use, NOT batch training.

Usage:
    # As a script
    python openface_inference.py path/to/video.mp4
    
    # As a module
    from OpenFace.openface_inference import extract_features
    features = extract_features('video.mp4')
"""

import os
import sys
import json

try:
    from .openface_feature_extractor import OpenFaceExtractor
except ImportError:
    from openface_feature_extractor import OpenFaceExtractor


def extract_features(video_path, verbose=True, use_cache=True):
    """
    Extract facial expression features from a single video.
    
    Args:
        video_path: Path to video file
        verbose: Print progress messages
        use_cache: Use cached OpenFace results if available
        
    Returns:
        dict: Feature dictionary with keys:
            - video_name: Video filename
            - detection_rate: Face detection success rate
            - detection_count: Number of frames with detected faces
            - total_frames: Total frames processed
            - AU04_max, AU06_max, etc.: Action Unit intensities (max values)
            - error: Error message if extraction failed (only present on failure)
            
    Example:
        >>> features = extract_features('squat_video.mp4')
        >>> print(f"RPE predictors: AU max = {features['AU12_max']:.2f}")
        >>> print(f"Detection rate: {features['detection_rate']*100:.1f}%")
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    video_name = os.path.basename(video_path)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"OpenFace Feature Extraction: {video_name}")
        print(f"{'='*60}\n")
    
    try:
        # Create extractor
        extractor = OpenFaceExtractor(
            use_pose_guidance=True,    # MediaPipe face cropping for better detection
            max_only=True,              # Extract max values only (sufficient for RPE)
            load_minimal_columns=True,  # Load only important AU columns
            sample_fps=None,            # Process all frames (or set to 10 for faster)
            verbose=verbose
        )
        
        # Extract features
        features = extractor.extract_from_video(video_path)
        
        # Convert to flat dictionary (remove nested metadata)
        result = {'video_name': video_name}
        for key, value in features.items():
            if key != 'metadata':
                result[key] = value
        
        if verbose:
            print(f"\n✓ Extraction successful!")
            print(f"  Detection rate: {result.get('detection_rate', 0)*100:.1f}%")
            print(f"  Frames detected: {result.get('detection_count', 0)}/{result.get('total_frames', 0)}")
            
            # Show some important AUs
            au_keys = [k for k in result.keys() if 'AU' in k and 'max' in k]
            if au_keys:
                print(f"  Action Units extracted: {len(au_keys)}")
                print(f"    Sample: AU12_max={result.get('AU12_max', 0):.2f}, "
                      f"AU04_max={result.get('AU04_max', 0):.2f}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"\n✗ Extraction failed: {e}")
        
        return {
            'video_name': video_name,
            'error': str(e)
        }


def extract_features_to_json(video_path, output_json=None, verbose=True):
    """
    Extract features and save to JSON file.
    
    Args:
        video_path: Path to video file
        output_json: Output JSON path (default: video_name_features.json)
        verbose: Print messages
        
    Returns:
        str: Path to output JSON file
    """
    features = extract_features(video_path, verbose=verbose)
    
    if output_json is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_json = f"{base_name}_openface_features.json"
    
    with open(output_json, 'w') as f:
        json.dump(features, f, indent=2)
    
    if verbose:
        print(f"\n✓ Features saved to: {output_json}")
    
    return output_json


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python openface_inference.py <video_path> [--json output.json]")
        print("\nExamples:")
        print("  python openface_inference.py squat_video.mp4")
        print("  python openface_inference.py squat_video.mp4 --json features.json")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Check for --json flag
    if '--json' in sys.argv:
        json_idx = sys.argv.index('--json')
        if json_idx + 1 < len(sys.argv):
            output_json = sys.argv[json_idx + 1]
        else:
            output_json = None
        extract_features_to_json(video_path, output_json)
    else:
        # Just extract and print
        features = extract_features(video_path)
        
        print(f"\n{'='*60}")
        print("EXTRACTED FEATURES:")
        print(f"{'='*60}")
        
        # Print readable summary
        for key, value in sorted(features.items()):
            if key == 'video_name':
                print(f"  {key}: {value}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
