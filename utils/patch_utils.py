"""
Patch extraction and aggregation utilities for deepfake detection pipeline.

This module provides unified functions for:
- Extracting patches from images
- Aggregating patch-level features using multi-stat pooling (ALWAYS 100D)

NOTE: This pipeline ONLY supports multi-stat pooling (mean+std+max+min).
Single-stat pooling has been removed as it's insufficient for deepfake detection.
"""

import cv2
import numpy as np


def extract_patches(image: np.ndarray, patch_size: int, stride: int):
    """
    Extract patches from an image with location tracking.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        patch_size: Size of square patches in pixels
        stride: Stride for patch extraction
    
    Returns:
        patches: List of patch arrays
        patch_locations: List of (y, x) coordinates for each patch
    """
    h, w = image.shape[:2]
    patches = []
    patch_locations = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            patch_locations.append((y, x))
    
    # Fallback for images smaller than patch_size
    if not patches:
        patches.append(cv2.resize(image, (patch_size, patch_size)))
        patch_locations.append((0, 0))
    
    return patches, patch_locations


def aggregate_patch_features(patch_feats, use_multi_stat: bool = True):
    """
    Aggregate patch-level features to image-level features using multi-stat pooling.
    
    IMPORTANT: This pipeline REQUIRES multi-stat pooling (mean+std+max+min).
    The use_multi_stat parameter is kept for backward compatibility but will
    raise an error if set to False.
    
    Args:
        patch_feats: Array of shape (n_patches, n_base_features) with features for each patch
                    Typically (n_patches, 25) for base features
        use_multi_stat: MUST be True.
    
    Returns:
        Aggregated feature vector of shape (n_base_features * 4,)
        For 25 base features, returns 100D vector (25 mean + 25 std + 25 max + 25 min)
    
    Raises:
        ValueError: If use_multi_stat is False
    
    Example:
        >>> patch_features = np.random.rand(4, 25)  # 4 patches, 25 features each
        >>> image_features = aggregate_patch_features(patch_features)
        >>> print(image_features.shape)  # (100,)
    """
    if not use_multi_stat:
        raise ValueError(
            "Multi-stat pooling (mean+std+max+min) is required for accurate "
            "deepfake detection. Please set use_multi_stat=True or remove the parameter."
        )
    
    # Multi-stat pooling: compute statistics across patches
    mean_vec = np.mean(patch_feats, axis=0)  # (n_base_features,)
    std_vec = np.std(patch_feats, axis=0)    # (n_base_features,)
    max_vec = np.max(patch_feats, axis=0)    # (n_base_features,)
    min_vec = np.min(patch_feats, axis=0)    # (n_base_features,)
    
    # Concatenate: [mean | std | max | min]
    # For 25 base features: [25 mean | 25 std | 25 max | 25 min] = 100D
    return np.concatenate([mean_vec, std_vec, max_vec, min_vec])

