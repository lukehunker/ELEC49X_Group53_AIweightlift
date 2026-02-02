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

__all__ = [
    'OpenFaceExtractor',
    'identify_missing_videos',
    'impute_by_rpe_average',
    'load_rpe_labels',
]
