
import sys
import json
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def compute_patch_gradcam(classifier, patch_feats, top_idx, importance, style_dim, device):
    """
    Compute GradCAM heatmap showing which patches contribute most to ALL feature interactions.
    Uses all features weighted by their importance (not just top ones).
    
    Returns a heatmap array with same spatial dimensions as patches.
    """
    print("Computing patch-level GradCAM (using ALL features weighted by importance)...")
    
    use_multi_stat = (style_dim == 100)
    n_patches = len(patch_feats)
    
    # Convert patch features to tensor
    patch_feats_tensor = torch.tensor(patch_feats, dtype=torch.float32, device=device)
    patch_feats_tensor.requires_grad_(True)
    
    # Compute aggregated features from patches (same as in extraction)
    if use_multi_stat:
        mean_vec = torch.mean(patch_feats_tensor, dim=0)
        std_vec = torch.std(patch_feats_tensor, dim=0)
        max_vec = torch.max(patch_feats_tensor, dim=0)[0]
        min_vec = torch.min(patch_feats_tensor, dim=0)[0]
        style_vec = torch.cat([mean_vec, std_vec, max_vec, min_vec])
    else:
        style_vec = torch.mean(patch_feats_tensor, dim=0)
    
    # Forward pass
    logit_fake = classifier(style_vec.unsqueeze(0))
    
    # Backward pass to get gradients w.r.t. patch features
    classifier.zero_grad()
    logit_fake.backward()
    
    # Get gradients for each patch
    patch_grads = patch_feats_tensor.grad  # Shape: (n_patches, 25)
    
    # Compute importance for each patch based on ALL features (weighted by importance)
    # This captures all feature interactions, not just top ones
    patch_importance = torch.zeros(n_patches, device=device)
    
    n_features = len(importance)
    
    for feat_idx in range(n_features):
        # Skip features with zero importance to avoid noise
        if importance[feat_idx] < 1e-6:
            continue
            
        if feat_idx < 25:
            # This is a mean value - check how much each patch contributes
            feat_grad = patch_grads[:, feat_idx]  # Gradient for this feature across patches
            feat_value = patch_feats_tensor[:, feat_idx]  # Feature value for each patch
            # Importance = gradient * value (like grad * input), weighted by feature importance
            patch_importance += (feat_grad * feat_value).abs() * importance[feat_idx]
        elif use_multi_stat:
            # For std/max/min, we need to compute how patches contribute
            base_idx = (feat_idx - 25) % 25
            stat_type_idx = (feat_idx - 25) // 25
            
            if stat_type_idx == 0:  # std
                # Contribution to std: how much does this patch deviate from mean?
                mean_val = torch.mean(patch_feats_tensor[:, base_idx])
                patch_contrib = (patch_feats_tensor[:, base_idx] - mean_val) ** 2
            elif stat_type_idx == 1:  # max
                # Contribution to max: is this patch the max?
                max_val = torch.max(patch_feats_tensor[:, base_idx])
                patch_contrib = (patch_feats_tensor[:, base_idx] == max_val).float()
            else:  # min
                # Contribution to min: is this patch the min?
                min_val = torch.min(patch_feats_tensor[:, base_idx])
                patch_contrib = (patch_feats_tensor[:, base_idx] == min_val).float()
            
            patch_importance += patch_contrib * importance[feat_idx]
    
    # Normalize
    patch_importance = patch_importance.detach().cpu().numpy()
    if patch_importance.max() > 0:
        patch_importance = patch_importance / patch_importance.max()
    
    return patch_importance


    
def create_gradcam_heatmap(image_array, patch_locations, patch_importance, patch_size=512, stride=512):
    """
    Create a GradCAM heatmap overlay on the original image.
    
    Args:
        image_array: Original image as numpy array (H, W, 3)
        patch_locations: List of (y, x) coordinates for each patch
        patch_importance: Importance scores for each patch (n_patches,)
        patch_size: Size of patches
        stride: Stride used for patch extraction
    
    Returns:
        Heatmap image with overlay
    """
    h, w = image_array.shape[:2]
    
    # Create a heatmap array
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Fill in importance values for each patch location
    for (y, x), importance in zip(patch_locations, patch_importance):
        y_end = min(y + patch_size, h)
        x_end = min(x + patch_size, w)
        heatmap[y:y_end, x:x_end] = importance
    
    # Apply Gaussian blur for smoother visualization
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply colormap (jet colormap: blue=low, red=high)
    heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Overlay on original image
    overlay = cv2.addWeighted(image_array, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay, heatmap


def save_gradcam_visualization(image_path, image_array, overlay, heatmap, output_path=None):
    """Save GradCAM visualization"""
    if output_path is None:
        # Create output filename based on input
        base_name = image_path.rsplit('.', 1)[0] if '.' in image_path else image_path
        output_path = f"{base_name}_gradcam.png"
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image_array)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("GradCAM Heatmap\n(Red = High Importance)", fontsize=14)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay\n(Red regions = High contribution to feature interactions)", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  GradCAM visualization saved to: {output_path}")
    return output_path


def get_important_patch_locations(patch_locations, patch_importance, image_shape, patch_size=512):
    """
    Identify patches with medium to high importance and return their locations.
    
    Returns a dictionary with:
    - 'high': List of (location_desc, y, x, importance) for red/high importance patches (>0.7)
    - 'medium_high': List of (location_desc, y, x, importance) for yellow/medium-high patches (0.4-0.7)
    - 'medium_low': List of (location_desc, y, x, importance) for green/medium-low patches (0.2-0.4)
    
    location_desc is a string like "upper-left", "middle-center", "lower-right"
    """
    h, w = image_shape[:2]
    
    # Categorize patches by importance
    high_patches = []      # Red: > 0.7
    medium_high_patches = []  # Yellow: 0.4-0.7
    medium_low_patches = []   # Green: 0.2-0.4
    
    for (y, x), imp in zip(patch_locations, patch_importance):
        # Convert pixel coordinates to relative positions
        center_y = y + patch_size // 2
        center_x = x + patch_size // 2
        
        # Describe location in image
        if center_y < h * 0.33:
            vert_pos = "upper"
        elif center_y < h * 0.67:
            vert_pos = "middle"
        else:
            vert_pos = "lower"
            
        if center_x < w * 0.33:
            horiz_pos = "left"
        elif center_x < w * 0.67:
            horiz_pos = "center"
        else:
            horiz_pos = "right"
        
        location_desc = f"{vert_pos}-{horiz_pos}"
        
        if imp > 0.7:
            high_patches.append((location_desc, y, x, imp))
        elif imp > 0.4:
            medium_high_patches.append((location_desc, y, x, imp))
        elif imp > 0.2:
            medium_low_patches.append((location_desc, y, x, imp))
    
    # Sort by importance (descending)
    high_patches.sort(key=lambda x: x[3], reverse=True)
    medium_high_patches.sort(key=lambda x: x[3], reverse=True)
    medium_low_patches.sort(key=lambda x: x[3], reverse=True)
    
    return {
        'high': high_patches[:5],  # Top 5 high importance
        'medium_high': medium_high_patches[:5],  # Top 5 medium-high
        'medium_low': medium_low_patches[:3]  # Top 3 medium-low
    }


def format_patch_locations(patch_info):
    """Format patch location information for the prompt"""
    if not any(patch_info.values()):
        return ""
    
    lines = []
    
    if patch_info['high']:
        lines.append("HIGH IMPORTANCE QUADRANTS:")
        for loc_desc, y, x, imp in patch_info['high']:
            lines.append(f"  - {loc_desc} quadrant (importance: {imp:.2f}, y={y}, x={x})")
        lines.append("")
    
    if patch_info['medium_high']:
        lines.append("MEDIUM-HIGH IMPORTANCE QUADRANTS:")
        for loc_desc, y, x, imp in patch_info['medium_high']:
            lines.append(f"  - {loc_desc} quadrant (importance: {imp:.2f}, y={y}, x={x})")
        lines.append("")
    
    if patch_info['medium_low']:
        lines.append("MEDIUM-LOW IMPORTANCE QUADRANTS:")
        for loc_desc, y, x, imp in patch_info['medium_low']:
            lines.append(f"  - {loc_desc} quadrant (importance: {imp:.2f}, y={y}, x={x})")
        lines.append("")
    
    return "\n".join(lines)

