from PIL import Image
import os, json, numpy as np
from tqdm import tqdm
from pathlib import Path, PureWindowsPath
from models.style_extractor_pure import PureStyleExtractor
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def to_native_path(p: str, base: Path) -> Path:
    """Convert Windows-style or mixed separators to the current OS Path.

    If the resulting path is relative, it will be interpreted relative to `base`.
    """
    # Parse as Windows path to split on backslashes and handle drive letters safely
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

style_embs, labels, clusters, sims = [], [], [], []

print(f"Processing {len(data)} samples...")
for sample in tqdm(data):
    # Normalize metadata path for cross-platform usage (Windows -> macOS/Linux)
    p = sample["path"]
    img_path = to_native_path(p, root)
    if not img_path.exists():
        # Fallback: some metadata may be relative to 'openfake-annotation/'
        img_path = to_native_path(str(Path("openfake-annotation") / p), root)

    # print(str(img_path))

    img = Image.open(str(img_path)).convert("RGB")
    
    style_vec = style_extractor(np.array(img))[None, :]
    
    style_embs.append(style_vec)
    labels.append(1 if sample["true_label"] == "real" else 0)
    clusters.append(sample.get("cluster_id_style", -1))
    sims.append(sample.get("similarity", 0.0))

os.makedirs(str(out_path.parent), exist_ok=True)
np.savez_compressed(
    str(out_path),
    style=np.vstack(style_embs),
    label=np.array(labels),
    cluster=np.array(clusters),
    similarity=np.array(sims),
)

print(f"\nSaved pure style embeddings to {out_path}")
print(f"  Style features: {style_embs[0].shape[1]} (25 technical features)")
print(f"  Total samples: {len(labels)}")
print(f"  Real: {sum(labels)}")
print(f"  Fake: {len(labels) - sum(labels)}")
print(f"\nFeatures:")
for i, name in enumerate(style_extractor.get_feature_names(), 1):
    print(f"  {i:2d}. {name}")

