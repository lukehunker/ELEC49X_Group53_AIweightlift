"""
OpenFace Integration Module

Provides facial expression feature extraction for RPE prediction.
"""

from .openface_feature_extractor import OpenFaceExtractor
from .extract_features import extract_facial_features, flatten_features
from .impute_missing_features import (
    identify_missing_videos,
    impute_by_rpe_average,
    load_rpe_labels
)

__all__ = [
    'OpenFaceExtractor',
    'extract_facial_features',
    'flatten_features',
    'identify_missing_videos',
    'impute_by_rpe_average',
    'load_rpe_labels',
]
