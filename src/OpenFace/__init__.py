"""
OpenFace Integration Module

Provides facial expression feature extraction for RPE prediction.
"""

from .openface_feature_extractor import OpenFaceExtractor
from .impute_missing_features import (
    identify_missing_videos,
    impute_by_rpe_average,
    load_rpe_labels
)
from .openface_inference import extract_features

__all__ = [
    'OpenFaceExtractor',
    'extract_features',  # Simple inference function for single videos
    'identify_missing_videos',
    'impute_by_rpe_average',
    'load_rpe_labels',
]
