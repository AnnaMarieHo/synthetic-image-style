import torch
import json
import glob
import numpy as np
import time
from PIL import Image
from collections import defaultdict
from models.mlp_classifier import PureStyleClassifier
from models.style_extractor_pure import PureStyleExtractor
from utils.patch_utils import extract_patches, aggregate_patch_features
from utils.feature_utils import build_feature_names, compute_domain_similarity
from utils.model_utils import load_classifier
from utils.config_loader import Config, get_device, ensure_multi_stat
from patches_and_gradcam.patch_importance import compute_patch_gradcam, get_important_patch_locations

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress function
    def tqdm(iterable, desc="", total=None):
        return iterable

ensure_multi_stat()

# Get configuration values
device = get_device()
checkpoint_path = Config.checkpoint()
patch_size = Config.patch_size()
stride = Config.stride()
MAX_FAKE = Config.max_fake_samples()
MAX_REAL = Config.max_real_samples()
top_features_count = Config.top_features()

print(f"Configuration:")
print(f"  Device: {device}")
print(f"  Checkpoint: {checkpoint_path}")
print(f"  Patch size: {patch_size}, Stride: {stride}")
print(f"  Max samples: {MAX_FAKE} fake, {MAX_REAL} real")
print(f"  Multi-stat pooling: ENABLED (100D only)")
print()

# Load JSON samples and extract patch-level features
files = glob.glob("feature_importance/*.json")
print("Found:", len(files), "json files.")

# Load classifier and determine style_dim
classifier, style_dim = load_classifier(checkpoint_path, device)

# Initialize style extractor
style_extractor = PureStyleExtractor(device)

# Build feature names (should always be 100 for multi-stat)
feature_names = build_feature_names(style_dim)

if style_dim != 100:
    raise ValueError(f"Expected style_dim=100 (multi-stat), got {style_dim}. Check config.yaml and checkpoint.")

X = []
y = []
image_ids = []
image_paths = []
patch_data = []  # Store patch features and locations for each image

print("\nExtracting patch-level features from images...")

total_files = len(files)
processed_count = 0
fake_count = 0
real_count = 0
start_time = time.time()

for file_idx, json_path in enumerate(tqdm(files, desc="Loading JSON files", disable=not HAS_TQDM)):
    with open(json_path, "r", encoding="utf-8") as jf:
        data = json.load(jf)

    if not isinstance(data, list):
        print(f"Skipping non-list file: {json_path}")
        continue

    total_examples = len(data)
    for ex_idx, ex in enumerate(data):
        # Check if  limits have been reached for both classes
        if fake_count >= MAX_FAKE and real_count >= MAX_REAL:
            print(f"\nReached limits: {fake_count} fake, {real_count} real. Stopping.")
            break
        
        # Determine label
        is_fake = ex.get("true_label") == "fake"
        label = 1 if is_fake else 0
        
        # Skip if limit was reached for this class
        if is_fake and fake_count >= MAX_FAKE:
            continue
        if not is_fake and real_count >= MAX_REAL:
            continue
        
        image_path = ex.get("image_path")
        if not image_path:
            # Fallback to pre-computed features if no image path
            feats = ex.get("style_features")
            if feats is None:
                continue
            # Use pre-computed features 
            if len(X) == 0:
                # Initialize feature_names from first sample
                feature_names_from_json = list(feats.keys())
            X.append([feats[k] for k in feature_names_from_json])
            y.append(label)
            image_ids.append(ex.get("image_id"))
            image_paths.append(None)
            patch_data.append(None)
            if is_fake:
                fake_count += 1
            else:
                real_count += 1
            continue

        # Load image and extract patches
        try:
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img)
            
            # Extract patches using config values
            patches, patch_locations = extract_patches(img_array, patch_size, stride)
            
            # Extract features for each patch (ALWAYS normalized)
            patch_feats = [style_extractor(p, normalize=True) for p in patches]
            patch_feats = np.stack(patch_feats, axis=0)
            
            # Aggregate to image-level features (ALWAYS multi-stat)
            style_vec = aggregate_patch_features(patch_feats, use_multi_stat=True)
            
            X.append(style_vec)
            y.append(label)
            image_ids.append(ex.get("image_id"))
            image_paths.append(image_path)
            patch_data.append({
                "patch_feats": patch_feats,
                "patch_locations": patch_locations,
                "img_shape": img_array.shape[:2]
            })
            processed_count += 1
            if is_fake:
                fake_count += 1
            else:
                real_count += 1
            
            # Progress update every 50 images
            if processed_count % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {processed_count} images ({fake_count} fake, {real_count} real) ({rate:.1f} img/s)")
            
            # Stop if limits for both classes have been reached
            if fake_count >= MAX_FAKE and real_count >= MAX_REAL:
                break
            
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {e}")
            continue
    
    # Break outer loop if limits reached
    if fake_count >= MAX_FAKE and real_count >= MAX_REAL:
        break

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print(f"Loaded {len(X)} samples with patch-level features ({fake_count} fake, {real_count} real)")
if len(X) == 0:
    raise RuntimeError("No samples loaded")

X_tensor = torch.tensor(X, device=device)

# Collect Global Frequencies

pair_freq = defaultdict(int)

print("\nGathering pair frequencies...\n")
pass1_start = time.time()

for i in tqdm(range(len(X_tensor)), desc="Frequency collection", disable=not HAS_TQDM):
    x = X_tensor[i:i+1].clone().detach()
    x.requires_grad_(True)

    logit_fake = classifier(x)
    loss = logit_fake.sum()

    classifier.zero_grad()
    loss.backward()

    grads = x.grad.detach()[0]
    importance = (grads * x[0]).abs().detach().cpu().numpy()

    # Top features (from config)
    top_idx = np.argsort(-importance)[:top_features_count]

    # Count pairs
    for a in range(len(top_idx)):
        for b in range(a + 1, len(top_idx)):
            pair = tuple(sorted((top_idx[a], top_idx[b])))
            pair_freq[pair] += 1

    # Progress update every 500 samples 
    if not HAS_TQDM and i % 500 == 0:
        elapsed = time.time() - pass1_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(X_tensor) - i - 1) / rate if rate > 0 else 0
        print(f"  Frequency pass: {i+1}/{len(X_tensor)} ({rate:.1f} samples/s, ~{remaining:.0f}s remaining)")

total_samples = len(X_tensor)

# Normalize frequencies to 0–1
pair_freq_norm = {k: v / total_samples for k, v in pair_freq.items()}

# Normalize frequencies to 0–1
pair_freq_norm_savable = {
    # Convert tuple keys to string keys
    f"{k[0]}_{k[1]}": v / total_samples for k, v in pair_freq.items()
}

freq_out_path = "feature_importance/pair_freq_norm.json"
with open(freq_out_path, "w") as f:
    json.dump(pair_freq_norm_savable, f, indent=2)

print(f"Saved global frequency map to: {freq_out_path}")

pass1_elapsed = time.time() - pass1_start
print(f"\ncomplete in {pass1_elapsed:.1f}s ({len(X_tensor)/pass1_elapsed:.1f} samples/s)\n")

# Domain similarity is now imported from utils.feature_utils

# Compute interactions with coherency score

results_fake = {}
results_real = {}

print("Computing coherency scores and patch information...\n")
pass2_start = time.time()

for i in tqdm(range(len(X_tensor)), desc="Computing interactions", disable=not HAS_TQDM):
    x = X_tensor[i:i+1].clone().detach()
    x.requires_grad_(True)

    logit_fake = classifier(x)
    loss = logit_fake.sum()

    classifier.zero_grad()
    loss.backward()

    grads = x.grad.detach()[0]
    importance = (grads * x[0]).abs().detach().cpu().numpy()

    # top features
    top_idx = np.argsort(-importance)[:10]

    # Compute pair coherency 
    pair_scores = []
    for a in range(len(top_idx)):
        for b in range(a + 1, len(top_idx)):
            i1, i2 = top_idx[a], top_idx[b]
            key = tuple(sorted((i1, i2)))

            freq_score = pair_freq_norm.get(key, 0)
            dom_score = compute_domain_similarity(feature_names[i1], feature_names[i2])

            mag = importance[i1] * importance[i2]
            mag_score = abs(mag) / (abs(mag) + 1e-6)

            coherency = freq_score * dom_score * mag_score

            pair_scores.append((coherency, i1, i2))

    # Sort by coherency
    pair_scores.sort(key=lambda x: -x[0])
    top_pairs = [
        {
            "features": [feature_names[i1], feature_names[i2]],
            "coherency": float(score),
            "values": [float(X[i, i1]), float(X[i, i2])]
        }
        for (score, i1, i2) in pair_scores[:5]
    ]


    #  Compute patch-level information 
    patch_info = None
    if patch_data[i] is not None:
        patch_feats = patch_data[i]["patch_feats"]
        patch_locations = patch_data[i]["patch_locations"]
        img_shape = patch_data[i]["img_shape"]
        
        # Compute patch importance using GradCAM
        patch_importance = compute_patch_gradcam(
            classifier, patch_feats, top_idx, importance, style_dim, device
        )
        
        # Get important patch locations
        patch_locations_info = get_important_patch_locations(
            patch_locations, patch_importance, img_shape, patch_size=patch_size
        )
        
        # Format patch locations for output
        patch_info = {
            "patch_importance": patch_importance.tolist(),
            "important_regions": {
                "high": [
                    {
                        "location": loc_desc,
                        "importance": float(imp)
                    }
                    for loc_desc, y, x, imp in patch_locations_info['high']
                ],
                "medium_high": [
                    {
                        "location": loc_desc,
                        "importance": float(imp)
                    }
                    for loc_desc, y, x, imp in patch_locations_info['medium_high']
                ],
                "medium_low": [
                    {
                        "location": loc_desc,
                        "importance": float(imp)
                    }
                    for loc_desc, y, x, imp in patch_locations_info['medium_low']
                ]
            }
        }

    result_entry = {
        "top_pairs": top_pairs,
        "patch_info": patch_info  # None if image_path was not available
    }
    
    # Save to appropriate dictionary based on label
    if y[i] == 1:  # Fake
        results_fake[image_ids[i]] = result_entry
    else:  # Real
        results_real[image_ids[i]] = result_entry

    # Progress update every 100 samples 
    if not HAS_TQDM and (i + 1) % 100 == 0:
        elapsed = time.time() - pass2_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(X_tensor) - i - 1) / rate if rate > 0 else 0
        print(f"  Processed {i+1}/{len(X_tensor)} images ({rate:.1f} img/s, ~{remaining:.0f}s remaining)")

pass2_elapsed = time.time() - pass2_start
total_elapsed = time.time() - start_time
print(f"\ncomplete in {pass2_elapsed:.1f}s ({len(X_tensor)/pass2_elapsed:.1f} samples/s)")
print(f"Total processing time: {total_elapsed:.1f}s\n")

print("Saving results...")

# Save fake samples
out_path_fake = "feature_importance/coherency/mlp_interactions_with_coherency_fake.json"
with open(out_path_fake, "w") as f:
    json.dump(results_fake, f, indent=2)
print(f"Saved fake coherency interaction map: {out_path_fake}")
print(f"  Fake images processed: {len(results_fake)}")

# Save real samples
out_path_real = "feature_importance/coherency/mlp_interactions_with_coherency_real.json"
with open(out_path_real, "w") as f:
    json.dump(results_real, f, indent=2)
print(f"Saved real coherency interaction map: {out_path_real}")
print(f"  Real images processed: {len(results_real)}")

print(f"\nTotal images processed: {len(results_fake) + len(results_real)}")
