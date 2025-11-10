
from PIL import Image
import os, json, numpy as np, cv2
from tqdm import tqdm
from pathlib import Path, PureWindowsPath
from models.style_extractor_pure import PureStyleExtractor
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-baseline", action="store_true",
        help="If set, compute mean/std from real images and save to style_norm_baseline_real.npz")
    parser.add_argument("--data-root", "--data_root", dest="data_root", type=str, default=".",
        help="Root directory for image paths and dataset files")
    parser.add_argument("--in-json", "--in_json", dest="in_json", type=str,
        default="openfake-annotation/datasets/combined/metadata.json",
        help="Path to input metadata JSON (relative to data root or absolute)")
    parser.add_argument("--out-path", "--out_path", dest="out_path", type=str,
        default="openfake-annotation/datasets/combined/cache/pure_style_embeddings.npz",
        help="Output .npz path (relative to data root or absolute)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for style extractor")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size in pixels")
    parser.add_argument("--stride", type=int, default=256, help="Stride for patch extraction")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "median", "max"],
        help="Pooling method for patch-level features")
    parser.add_argument("--multi-stat", action="store_true",
        help="Use multi-statistic pooling (mean+std+max+min) for 100 features instead of 25")
    parser.add_argument("--max-per-class", type=int, default=None,
        help="Maximum samples per class (for balanced training). E.g., 9966 for balanced real/fake")
    return parser.parse_args()

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

def to_native_path(p: str, base: Path) -> Path:
    """Convert Windows-style or mixed separators to the current OS Path."""
    candidate = Path(*PureWindowsPath(p).parts)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate

args = parse_args()

root = Path(args.data_root).resolve()
in_json = (root / Path(args.in_json)).resolve()
out_path = (root / Path(args.out_path)).resolve()

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

print(f"  Patch size: {args.patch_size}, stride: {args.stride}, pooling: {args.pool}")
if args.multi_stat:
    print(f"  Multi-stat pooling enabled: 100 features (mean+std+max+min)")

for sample in tqdm(data):
    if args.compute_baseline and sample["true_label"] != "real":
        continue

    p = sample["path"]
    img_path = to_native_path(p, root)
    if not img_path.exists():
        img_path = to_native_path(str(Path("openfake-annotation") / p), root)

    img = Image.open(str(img_path)).convert("RGB")

    patches = extract_patches(np.array(img), args.patch_size, args.stride)
    
    # Extract features (normalize for training, raw for baseline computation)
    should_normalize = not args.compute_baseline
    patch_feats = [style_extractor(p, normalize=should_normalize) for p in patches]

    # Stack patches into (n_patches, 25)
    patch_feats = np.stack(patch_feats, axis=0)

    # Store per-patch features for later visualization
    if not args.compute_baseline:
        style_patches.append(patch_feats)

    # Pool into per-image vector
    if args.multi_stat:
        # Multi-statistic pooling: captures patch varianc
        mean_vec = np.mean(patch_feats, axis=0)
        std_vec = np.std(patch_feats, axis=0)
        max_vec = np.max(patch_feats, axis=0)
        min_vec = np.min(patch_feats, axis=0)
        style_vec = np.concatenate([mean_vec, std_vec, max_vec, min_vec])[None, :]
    else:
        # Single-statistic pooling
        if args.pool == "mean":
            style_vec = np.mean(patch_feats, axis=0, keepdims=True)
        elif args.pool == "median":
            style_vec = np.median(patch_feats, axis=0, keepdims=True)
        else:
            style_vec = np.max(patch_feats, axis=0, keepdims=True)

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
