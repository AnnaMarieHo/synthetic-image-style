"""
Feature naming and domain similarity utilities for deepfake detection.

This module provides:
- Base feature names used across the pipeline
- Functions to build full feature names based on aggregation strategy
- Domain similarity computation for feature interaction analysis
"""

# Base feature names (25 features extracted per patch)
BASE_FEATURE_NAMES = [
    "color_correlation_gb", "color_correlation_rb", "color_correlation_rg",
    "color_saturation_var", "edge_coherence", "edge_density", "freq_falloff",
    "glcm_contrast_1", "glcm_contrast_3", "glcm_contrast_5", "glcm_energy_1",
    "glcm_homogeneity_1", "gradient_mean", "gradient_skewness", "gradient_std",
    "high_freq_energy", "lab_a_skewness", "lab_b_skewness", "lbp_entropy",
    "mid_freq_energy", "noise_kurtosis", "noise_local_var", "noise_skewness",
    "noise_variance", "spectral_entropy"
]


def build_feature_names(style_dim):
    """
    Build full feature names based on style dimension.
    
    Args:
        style_dim: Feature dimension (25 for single-stat, 100 for multi-stat)
    
    Returns:
        List of feature names
    """
    if style_dim == 100:
        # Multi-stat aggregation: mean, std, max, min for each base feature
        feature_names = []
        for base in BASE_FEATURE_NAMES:
            feature_names.extend([
                f"{base}_mean",
                f"{base}_std",
                f"{base}_max",
                f"{base}_min"
            ])
        return feature_names
    else:
        # Single-stat aggregation: just the base features
        return BASE_FEATURE_NAMES.copy()


def features_to_dict(feature_vector, style_dim):
    """
    Convert feature vector to named dictionary.
    
    Args:
        feature_vector: Numpy array of feature values
        style_dim: Feature dimension (25 or 100)
    
    Returns:
        Dictionary mapping feature names to values
    """
    feature_names = build_feature_names(style_dim)
    return {name: float(val) for name, val in zip(feature_names, feature_vector)}


def get_feature_domain(feature_name):
    """
    Extract domain prefix from feature name.
    
    Args:
        feature_name: Full feature name (e.g., "noise_variance_mean")
    
    Returns:
        Domain prefix (e.g., "noise")
    """
    return feature_name.split("_")[0]


def compute_domain_similarity(feature1, feature2):
    """
    Calculate semantic domain similarity between two features.
    
    Features from the same domain (e.g., both noise features) have similarity 1.0.
    Related domains (e.g., frequency and texture) have similarity 0.8.
    Unrelated domains have similarity 0.3.
    
    Args:
        feature1: First feature name
        feature2: Second feature name
    
    Returns:
        Similarity score between 0.3 and 1.0
    """
    d1 = get_feature_domain(feature1)
    d2 = get_feature_domain(feature2)
    
    # Same domain
    if d1 == d2:
        return 1.0
    
    # Related domains (texture/frequency relationships)
    related_pairs = {
        ("glcm", "mid"), ("mid", "glcm"),
        ("glcm", "freq"), ("freq", "glcm"),
        ("spectral", "freq"), ("freq", "spectral")
    }
    
    if (d1, d2) in related_pairs:
        return 0.8
    
    # Weak relation
    return 0.3

