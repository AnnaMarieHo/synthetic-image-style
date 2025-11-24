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


def extract_patches(image: np.ndarray, patch_size: int, stride: int):
    h, w = image.shape[:2]
    patches = []
    patch_locations = []  # Store (y, x) coordinates for each patch
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            patch_locations.append((y, x))
    if not patches:
        patches.append(cv2.resize(image, (patch_size, patch_size)))
        patch_locations.append((0, 0))
    return patches, patch_locations


def aggregate_patch_features(patch_feats):
    """
    Aggregate patch-level features to image-level features.
    
    Args:
        patch_feats: Array of shape (n_patches, 25) with features for each patch
        use_multi_stat: If True, compute mean/std/max/min (100D), else just mean (25D)
    
    Returns:
        Aggregated feature vector (100D or 25D)
    """
    mean_vec = np.mean(patch_feats, axis=0)
    std_vec = np.std(patch_feats, axis=0)
    max_vec = np.max(patch_feats, axis=0)
    min_vec = np.min(patch_feats, axis=0)
    return np.concatenate([mean_vec, std_vec, max_vec, min_vec])



def domain(feature_name):
    """Extracts the base domain of a feature."""
    return feature_name.split("_")[0]

def domain_similarity(f1, f2):
    """Calculates domain similarity based on feature type."""
    d1 = domain(f1)
    d2 = domain(f2)

    if d1 == d2:
        return 1.0
    # Add rules for related domains (e.g., texture/frequency)
    if (d1, d2) in [
        ("glcm", "mid"), ("mid", "glcm"),
        ("glcm", "freq"), ("freq", "glcm"),
        ("spectral", "freq"), ("freq", "spectral")
    ]:
        return 0.8
    return 0.3  # weak relation

def load_pair_frequencies(file_path="feature_importance/pair_freq_norm.json"):
    """Loads the pre-calculated pair frequency map."""
    if not os.path.exists(file_path):
        print(f"Warning: Frequency file not found at {file_path}. Using default freq=0.")
        return {}
    with open(file_path, "r") as f:
        return json.load(f)
        


def extract_style_features_and_interactions(image_path, device):
    """Extract features and compute feature importance using gradients"""
    print("Extracting style features...\n")
    GLOBAL_PAIR_FREQ = load_pair_frequencies()

    checkpoint = torch.load("checkpoints/pure_style_512.pt", map_location=device)
    style_dim = checkpoint.get("style_dim", 25)
    
    style_extractor = PureStyleExtractor(device)
    
    
    classifier = PureStyleClassifier(style_dim=style_dim).to(device)
    classifier.load_state_dict(checkpoint["model"])
    classifier.eval()
    
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    original_shape = img_array.shape[:2]  # (H, W)
    
    patches, patch_locations = extract_patches(img_array, 512, 512)
    patch_feats = [style_extractor(p, normalize=True) for p in patches]
    patch_feats = np.stack(patch_feats, axis=0)
    
    use_multi_stat = (style_dim == 100)
    style_vec = aggregate_patch_features(patch_feats)
    
    # Get probability
    style_tensor = torch.tensor(style_vec, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = classifier(style_tensor)
        prob_real = torch.sigmoid(logits).item()
    
    prob_fake = 1 - prob_real
    
    print(f"Classifier Result: {prob_fake:.1%} probability FAKE")
    print(f"Prediction: {'FAKE' if prob_fake > 0.5 else 'REAL'}\n")
    
    # ============================================================
    # GRADIENT-BASED FEATURE IMPORTANCE 
    # ============================================================
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
    
    # Base feature names (for the first 25 mean values)
    base_feature_names = [
        "color_correlation_gb", "color_correlation_rb", "color_correlation_rg",
        "color_saturation_var", "edge_coherence", "edge_density", "freq_falloff",
        "glcm_contrast_1", "glcm_contrast_3", "glcm_contrast_5", "glcm_energy_1",
        "glcm_homogeneity_1", "gradient_mean", "gradient_skewness", "gradient_std",
        "high_freq_energy", "lab_a_skewness", "lab_b_skewness", "lbp_entropy",
        "mid_freq_energy", "noise_kurtosis", "noise_local_var", "noise_skewness",
        "noise_variance", "spectral_entropy"
    ]
    
    # Build full feature names (mean, std, max, min for each base feature)
    if style_dim == 100:
        feature_names = []
        for base in base_feature_names:
            feature_names.extend([f"{base}_mean", f"{base}_std", f"{base}_max", f"{base}_min"])
    else:
        feature_names = base_feature_names
    
    # Get top features by gradient importance
    top_idx = np.argsort(-importance)[:10]
    
    # Extract the 25D mean values for display
    features_25d = style_vec[:25]
    
    # # Map top features to their values (use mean values primarily)
    # top_features = []
    # for i in top_idx[:5]:
    #     if i < 25:
    #         # This is a mean value
    #         top_features.append((feature_names[i], style_vec[i], importance[i]))
    #     else:
    #         # This is std/max/min, map back to base feature
    #         base_idx = (i - 25) % 25
    #         stat_type = ["std", "max", "min"][(i - 25) // 25]
    #         top_features.append((f"{base_feature_names[base_idx]}_{stat_type}", style_vec[i], importance[i]))
    
    # Compute top pairs (like training script)
   # ... (inside extract_style_features_and_interactions) ...
    # Compute top pairs (like training script)
    # --- CODE TO REPLACE TOP_PAIRS CALCULATION IN INFERENCE FUNCTION ---
    pair_scores = []

    for a in range(len(top_idx)):
        for b in range(a + 1, len(top_idx)):
            i1, i2 = top_idx[a], top_idx[b]
            name1 = feature_names[i1]
            name2 = feature_names[i2]

            # 1. Magnitude Score Calculation 
            mag = importance[i1] * importance[i2]
            mag_score = abs(mag) / (abs(mag) + 1e-6)

            # 2. Domain Score Calculation
            dom_score = domain_similarity(name1, name2)

            # 3. Frequency Score Calculation 
            # Create a standardized key for lookup (index sorted)
            key_tuple = tuple(sorted((i1, i2)))
            key_str = f"{key_tuple[0]}_{key_tuple[1]}"
            
            # Look up frequency score, default to 0 if pair was never seen in training
            freq_score = GLOBAL_PAIR_FREQ.get(key_str, 0.0)

            # 4. Final Full Bounded Coherency
            coherency = freq_score * dom_score * mag_score

            # Append result to pair_scores...
            # ...
            pair_scores.append({
                "features": [name1, name2],
                "coherency": float(coherency), # Now bounded between 0.0 and 1.0
                "values": [float(style_vec[i1]), float(style_vec[i2])]
            })
    
    pair_scores.sort(key=lambda x: x["coherency"], reverse=True)
    pair_scores = pair_scores[:5]  # Use top 5 pairs to match training format
    

    # print(f"[OK] Identified top {len(top_features)} important features\n")
    
    # return features_25d, prob_fake, top_features, pair_scores, img_array, patch_locations, patch_feats, top_idx, importance
    return features_25d, prob_fake, pair_scores, img_array, patch_locations, patch_feats, top_idx, importance
