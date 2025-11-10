import torch
import sys
from PIL import Image
import numpy as np
import cv2
from models.style_extractor_pure import PureStyleExtractor
import torch.nn as nn

class PureStyleClassifier(nn.Module):
    def __init__(self, style_dim=25, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, style_features):
        return self.net(style_features)

def extract_patches(image: np.ndarray, patch_size: int, stride: int):
    """Return list of patches from an image as numpy arrays."""
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    if not patches:
        patches.append(cv2.resize(image, (patch_size, patch_size)))  # fallback for small imgs
    return patches

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate/predict_pure_style.py <image_path>")
        return
    
    image_path = sys.argv[1]
    checkpoint_path = "checkpoints/pure_style.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    style_dim = checkpoint.get("style_dim", 25)
    
    style_extractor = PureStyleExtractor(device)
    
    model = PureStyleClassifier(style_dim=style_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"Style dimension: {style_dim} technical features")
    print(f"Architecture: 100% content-agnostic (NO CLIP)")
    
    print(f"\nAnalyzing image: {image_path}")
    print("-" * 50)
    
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Use same patch-based approach as training
    patch_size = 512
    stride = 512
    pool_method = "mean"
    
    patches = extract_patches(img_array, patch_size, stride)
    
    # Extract features with normalization (using hard-coded constants)
    patch_feats = [style_extractor(p, normalize=True) for p in patches]
    patch_feats = np.stack(patch_feats, axis=0)
    
    # Check if model expects multi-stat features based on dimension
    expected_dim = style_dim
    use_multi_stat = (expected_dim == 100)  # 25 * 4
    
    # Pool patches (same as training)
    if use_multi_stat:
        mean_vec = np.mean(patch_feats, axis=0)
        std_vec = np.std(patch_feats, axis=0)
        max_vec = np.max(patch_feats, axis=0)
        min_vec = np.min(patch_feats, axis=0)
        style_vec = np.concatenate([mean_vec, std_vec, max_vec, min_vec])
        pool_method = "multi-stat (mean+std+max+min)"
    else:
        if pool_method == "mean":
            style_vec = np.mean(patch_feats, axis=0)
        elif pool_method == "median":
            style_vec = np.median(patch_feats, axis=0)
        else:
            style_vec = np.max(patch_feats, axis=0)
    
    print(f"Processed {len(patches)} patches (size={patch_size}, stride={stride}, pool={pool_method})")
    
    style_tensor = torch.tensor(style_vec, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(style_tensor)
        prob_real = torch.sigmoid(logits).item()
    
    prob_fake = 1 - prob_real
    prediction = "REAL" if prob_real > 0.5 else "FAKE"
    confidence = max(prob_real, prob_fake)
    
    print(f"\nPrediction: {prediction}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"\nDetailed Probabilities:")
    print(f"   Real: {prob_real:.2%}")
    print(f"   Fake: {prob_fake:.2%}")
    print(f"\nDetection Method: Pure artifact analysis")
    print(f"  - Frequency domain patterns")
    print(f"  - Noise characteristics")
    print(f"  - Texture anomalies")
    print(f"  - Edge inconsistencies")
    print(f"  - Color correlations")
    print()

if __name__ == "__main__":
    main()

