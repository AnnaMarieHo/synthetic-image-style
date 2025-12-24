import sys
import json
import torch
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from models.style_extractor_pure import PureStyleExtractor
from models.mlp_classifier import PureStyleClassifier
from utils.patch_utils import extract_patches, aggregate_patch_features
from utils.feature_utils import build_feature_names, compute_domain_similarity
from utils.model_utils import load_classifier
from utils.config_loader import Config, ensure_multi_stat


def load_pair_frequencies(file_path=None):
    """Loads the pre-calculated pair frequency map."""
    if file_path is None:
        file_path = Config.pair_frequencies()
    
    if not os.path.exists(file_path):
        print(f"Warning: Frequency file not found at {file_path}. Using default freq=0.")
        return {}
    with open(file_path, "r") as f:
        return json.load(f)
        


def extract_style_features_and_interactions(image_path, device):
    """Extract features and compute feature importance using gradients"""
    print("Extracting style features...\n")
    
    ensure_multi_stat()
    
    # Get config values
    checkpoint_path = Config.checkpoint()
    patch_size = Config.patch_size()
    stride = Config.stride()
    top_features_count = Config.top_features()
    
    GLOBAL_PAIR_FREQ = load_pair_frequencies()

    # Load classifier using utility function
    classifier, style_dim = load_classifier(checkpoint_path, device)
    
    if style_dim != 100:
        raise ValueError(f"Expected style_dim=100 (multi-stat), got {style_dim}")
    
    style_extractor = PureStyleExtractor(device)
    
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    original_shape = img_array.shape[:2]  # (H, W)
    
    patches, patch_locations = extract_patches(img_array, patch_size, stride)
    patch_feats = [style_extractor(p, normalize=True) for p in patches]
    patch_feats = np.stack(patch_feats, axis=0)
    
    style_vec = aggregate_patch_features(patch_feats, use_multi_stat=True)
    
    # Get probability
    style_tensor = torch.tensor(style_vec, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = classifier(style_tensor)
        prob_real = torch.sigmoid(logits).item()
    
    prob_fake = 1 - prob_real
    
    print(f"Classifier Result: {prob_fake:.1%} probability FAKE")
    print(f"Prediction: {'FAKE' if prob_fake > 0.5 else 'REAL'}\n")
    
    # GRADIENT-BASED FEATURE IMPORTANCE 
    print("Computing feature importance using gradients...")
    
    x = torch.tensor(style_vec, dtype=torch.float32, device=device).unsqueeze(0)
    x.requires_grad_(True)
    
    # Forward pass
    logit_fake = classifier(x)
    loss = logit_fake.sum()
    
    # Backward to get gradients
    classifier.zero_grad()
    loss.backward()
    
    # Compute importance: grad * input
    grads = x.grad.detach()[0]
    importance = (grads * x[0]).abs().detach().cpu().numpy()
    
    # Build feature names
    feature_names = build_feature_names(style_dim)
    
    # Get top features by gradient importance (from config)
    top_idx = np.argsort(-importance)[:top_features_count]
    
    # Extract the 25D mean values for display
    features_25d = style_vec[:25]
    

    pair_scores = []

    for a in range(len(top_idx)):
        for b in range(a + 1, len(top_idx)):
            i1, i2 = top_idx[a], top_idx[b]
            name1 = feature_names[i1]
            name2 = feature_names[i2]

            # Magnitude Score Calculation 
            mag = importance[i1] * importance[i2]
            mag_score = abs(mag) / (abs(mag) + 1e-6)

            # Domain Score Calculation
            dom_score = compute_domain_similarity(name1, name2)

            # Frequency Score Calculation 
            # Create a standardized key for lookup (index sorted)
            key_tuple = tuple(sorted((i1, i2)))
            key_str = f"{key_tuple[0]}_{key_tuple[1]}"
            
            # Look up frequency score, default to 0 if pair was never seen in training
            freq_score = GLOBAL_PAIR_FREQ.get(key_str, 0.0)

            # Final Full Bounded Coherency
            coherency = freq_score * dom_score * mag_score

            # Append result to pair_scores
            pair_scores.append({
                "features": [name1, name2],
                "coherency": float(coherency), # Bounded between 0.0 and 1.0
                "values": [float(style_vec[i1]), float(style_vec[i2])]
            })
    
    pair_scores.sort(key=lambda x: x["coherency"], reverse=True)
    pair_scores = pair_scores[:5]  # Use top 5 pairs

    return features_25d, prob_fake, pair_scores, img_array, patch_locations, patch_feats, top_idx, importance
