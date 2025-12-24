"""
Model loading and checkpoint utilities for deepfake detection pipeline.

This module provides unified functions for loading models and checkpoints
to ensure consistency across training, evaluation, and inference scripts.
"""

import torch
from models.mlp_classifier import PureStyleClassifier
from models.style_extractor_pure import PureStyleExtractor


def load_classifier(checkpoint_path, device="cpu"):
    """
    Load pre-trained PureStyleClassifier from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        classifier: Loaded classifier in eval mode
        style_dim: Feature dimension from checkpoint (25 or 100)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    style_dim = checkpoint.get("style_dim", 25)
    
    classifier = PureStyleClassifier(style_dim=style_dim).to(device)
    classifier.load_state_dict(checkpoint["model"])
    classifier.eval()
    
    return classifier, style_dim


def load_style_extractor(device="cpu"):
    """
    Load PureStyleExtractor for feature extraction.
    
    Args:
        device: Device for computation ('cpu' or 'cuda')
    
    Returns:
        style_extractor: Initialized PureStyleExtractor
    """
    return PureStyleExtractor(device)


def get_feature_names_from_extractor(device="cpu"):
    """
    Get canonical feature names from style extractor.
    
    Args:
        device: Device for extractor
    
    Returns:
        List of base feature names (25 features)
    """
    extractor = PureStyleExtractor(device)
    return extractor.get_feature_names()

