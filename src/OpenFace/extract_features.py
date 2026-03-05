"""
OpenFace Feature Extraction - Single Video API

This module provides a simple function to extract facial expression features from a single video.
Returns a dictionary with AU (Action Unit) intensity features, detection rates, and metadata.

Usage:
    from OpenFace.extract_features import extract_facial_features
    
    features = extract_facial_features('workout_video.mp4')
    print(features['detection_rate'])
    print(features['AU04_r_max'])  # Brow lowerer max intensity
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .openface_feature_extractor import OpenFaceExtractor
except ImportError:
    from openface_feature_extractor import OpenFaceExtractor


def extract_facial_features(video_path, verbose=True, use_pose_guidance=True, 
                            sample_fps=10, max_only=True, visualize=False, save_overlay=True):
    """
    Extract facial expression features from a single video using OpenFace.
    
    Args:
        video_path (str): Path to the video file
        verbose (bool): Print progress messages (default: True)
        use_pose_guidance (bool): Use pose-guided face detection for better tracking (default: True)
        sample_fps (int): Sample rate for feature extraction (default: 10)
        max_only (bool): Extract only max features (default: True)
        visualize (bool): Display video with landmarks overlay (default: False)
        save_overlay (bool): Save overlay video with landmarks (default: True)
    
    Returns:
        dict: Feature dictionary with keys:
            - detection_rate: Face detection success rate (0.0 to 1.0)
            - AU features: Action Unit intensity features (mean, max, std, range, etc.)
                          Examples: 'AU04_r_max', 'AU06_r_mean', 'AU12_r_std'
            - overlay_video: Path to overlay video (if save_overlay=True)
            - metadata: Dictionary with video metadata
                - video_name: Name of the video file
                - width, height: Video dimensions
                - fps: Video frame rate
                - total_frames: Total frames in video
                - valid_frames: Frames with successful face detection
    
    Key Action Units (AUs):
        - AU04: Brow lowerer (effort/concentration)
        - AU06: Cheek raiser (pain/exertion)
        - AU07: Lid tightener (squinting/strain)
        - AU09: Nose wrinkler (disgust/effort)
        - AU10: Upper lip raiser (grimace)
        - AU12: Lip corner puller (smile/grimace)
        - AU25: Lips part (mouth opening)
        - AU26: Jaw drop (mouth wide open)
    
    Example:
        >>> features = extract_facial_features('squat_video.mp4')
        >>> print(f"Detection rate: {features['detection_rate']:.1%}")
        >>> print(f"Max brow lowerer: {features['AU04_r_max']:.2f}")
        >>> print(f"Video: {features['metadata']['video_name']}")
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or insufficient frames detected
        RuntimeError: If OpenFace binary is not found
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Initialize extractor
    extractor = OpenFaceExtractor(
        verbose=verbose,
        use_pose_guidance=use_pose_guidance,
        sample_fps=sample_fps,
        max_only=max_only,
        visualize=visualize,
        save_overlay=save_overlay
    )
    
    # Extract features
    features = extractor.extract_from_video(video_path)
    
    return features


def flatten_features(features):
    """
    Flatten nested feature dictionary into a single-level dictionary.
    Useful for converting to DataFrame or API responses.
    
    Args:
        features (dict): Feature dictionary from extract_facial_features()
    
    Returns:
        dict: Flattened dictionary with all features at top level
    
    Example:
        >>> features = extract_facial_features('video.mp4')
        >>> flat = flatten_features(features)
        >>> # flat contains all features + metadata fields at top level
    """
    flat = {}
    
    for key, value in features.items():
        if key == 'metadata' and isinstance(value, dict):
            # Flatten metadata dict
            for meta_key, meta_value in value.items():
                flat[f'metadata_{meta_key}'] = meta_value
        else:
            flat[key] = value
    
    return flat
