
from PIL import Image
import os, json, numpy as np, cv2
from tqdm import tqdm
from pathlib import Path, PureWindowsPath
from models.style_extractor_pure import PureStyleExtractor
from utils.patch_utils import extract_patches, aggregate_patch_features
from utils.config_loader import Config, ensure_multi_stat
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess images for deepfake detection (100D multi-stat pooling )")
    parser.add_argument("--compute-baseline", action="store_true",
        help="If set, compute mean/std from real images and save to style_norm_baseline_real.npz")
    parser.add_argument("--data-root", "--data_root", dest="data_root", type=str, default=".",
        help="Root directory for image paths and dataset files")
    parser.add_argument("--in-json", "--in_json", dest="in_json", type=str,
        default=None,
        help="Path to input metadata JSON (defaults to config value)")
    parser.add_argument("--out-path", "--out_path", dest="out_path", type=str,
        default=None,
        help="Output .npz path (defaults to config value)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for style extractor")
    parser.add_argument("--max-per-class", type=int, default=None,
        help="Maximum samples per class (for balanced training). E.g., 9966 for balanced real/fake")
    return parser.parse_args()

def to_native_path(p: str, base: Path) -> Path:
    """Convert Windows-style or mixed separators to the current OS Path."""
    candidate = Path(*PureWindowsPath(p).parts)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate

args = parse_args()

ensure_multi_stat()

# Get config values
patch_size = Config.patch_size()
stride = Config.stride()
in_json = args.in_json or Config.metadata_path()
out_path = args.out_path or Config.embeddings_cache()

root = Path(args.data_root).resolve()
in_json = (root / Path(in_json)).resolve()
out_path = (root / Path(out_path)).resolve()

style_extractor = PureStyleExtractor(args.device)

with open(in_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Balance dataset if requested
if args.max_per_class and not args.compute_baseline:
    real_samples = [s for s in data if s["true_label"] == "real"]
    fake_samples = [s for s in data if s["true_label"] == "fake"]
    
    # Shuffle and limit
    np.random.seed(42)
    np.random.shuffle(real_samples)
    np.random.shuffle(fake_samples)
    
    real_samples = real_samples[:args.max_per_class]
    fake_samples = fake_samples[:args.max_per_class]
    
    data = real_samples + fake_samples
    np.random.shuffle(data)  # Shuffle combined
    
    print(f"Balanced dataset to {len(real_samples)} real + {len(fake_samples)} fake = {len(data)} total")

style_embs, labels, clusters, sims = [], [], [], []
raw_feats = []
style_patches = []  

print(f"  Patch size: {patch_size}, stride: {stride}")
print(f"  Multi-stat pooling: 100D features")

for sample in tqdm(data):
    if args.compute_baseline and sample["true_label"] != "real":
        continue

    p = sample["path"]
    img_path = to_native_path(p, root)
    if not img_path.exists():
        img_path = to_native_path(str(Path("openfake-annotation") / p), root)

    img = Image.open(str(img_path)).convert("RGB")

    patches, patch_locations = extract_patches(np.array(img), patch_size, stride)
    
    # Extract features (normalize for training, raw for baseline computation)
    should_normalize = not args.compute_baseline
    patch_feats = [style_extractor(p, normalize=should_normalize) for p in patches]

    # Stack patches into (n_patches, 25)
    patch_feats = np.stack(patch_feats, axis=0)

    # Store per-patch features for later visualization
    if not args.compute_baseline:
        style_patches.append(patch_feats)

    # Use multi-stat pooling (100D)
    style_vec = aggregate_patch_features(patch_feats, use_multi_stat=True)[None, :]

    if args.compute_baseline:
        raw_feats.append(style_vec)
    else:
        style_embs.append(style_vec)
        labels.append(1 if sample["true_label"] == "real" else 0)
        clusters.append(sample.get("cluster_id_style", -1))
        sims.append(sample.get("similarity", 0.0))

if args.compute_baseline:
    X = np.vstack(raw_feats)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    np.savez("style_norm_baseline_real.npz", mean=mean, std=std)
    print("Saved real-image baseline to style_norm_baseline_real.npz")
    exit()

os.makedirs(str(out_path.parent), exist_ok=True)
np.savez_compressed(
    str(out_path),
    style=np.vstack(style_embs),
    label=np.array(labels),
    cluster=np.array(clusters),
    similarity=np.array(sims),
    style_patches=np.array(style_patches, dtype=object) 
)

print(f"\nSaved pure style embeddings to {out_path}")
print(f"  Style features: {style_embs[0].shape[1]}")
print(f"  Total samples: {len(labels)}")
print(f"  Real: {sum(labels)}")
print(f"  Fake: {len(labels) - sum(labels)}")
print(f"\nFeatures:")
for i, name in enumerate(style_extractor.get_feature_names(), 1):
    print(f"  {i:2d}. {name}")
