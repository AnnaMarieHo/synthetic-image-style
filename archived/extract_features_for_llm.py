"""
Extract style features + predictions for LLM training data generation.

Uses:
1. PureStyleExtractor to extract 25 deterministic style features
2. pure_style.pt (trained on 9966 fake + 9966 real unfiltered) and predicts fake/real

Output: JSON with features + predictions for all fake_balanced_filtered images
"""

import torch
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys
import os
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.style_extractor_pure import PureStyleExtractor
import torch.nn as nn


def extract_patches(image: np.ndarray, patch_size: int, stride: int):
    """Extract patches from an image. Returns list of patches as numpy arrays."""
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    if not patches:
        # Fallback for small images
        patches.append(cv2.resize(image, (patch_size, patch_size)))
    return patches


class PureStyleClassifier(nn.Module):
    """Same architecture as training script"""
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


def main():
    # Configuration
    # metadata_path = "openfake-annotation/datasets/fake_balanced_filtered/metadata.json"
    metadata_path = "openfake-annotation/datasets/combined/metadata.json"
    checkpoint_path = "checkpoints/pure_style.pt"
    output_path = "openfake-annotation/datasets/combined/llm_training_data.jsonl"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("="*60)
    print(f"Device: {device}")
    print(f"Metadata: {metadata_path}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Output: {output_path} (JSONL - incremental saving)")
    print()
    
    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} images from metadata")
    
    # Load style extractor (deterministic, no training)
    style_extractor = PureStyleExtractor(device)
    feature_names = style_extractor.get_feature_names()
    print(f"Initialized PureStyleExtractor ({len(feature_names)} features)")
    
    # Load trained model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    style_dim = checkpoint.get("style_dim", 25)
    
    model = PureStyleClassifier(style_dim=style_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded trained model")
    print()
    
    # Prepare output file (overwrite if exists)
    os.makedirs(Path(output_path).parent, exist_ok=True)
    output_file = open(output_path, "w", encoding="utf-8")
    
    # Track statistics
    num_processed = 0
    errors = []
    predictions_count = {"real": 0, "fake": 0}
    true_labels_count = {"real": 0, "fake": 0}
    
    print("Processing images (saving incrementally)...")
    print(f"Progress is saved after each image to: {output_path}")
    print()
    
    try:
        for idx, item in enumerate(tqdm(metadata)):
            try:
                # Resolve image path
                image_rel_path = item["path"]
                image_path = Path(image_rel_path)
                
                # Try multiple resolution strategies
                if not image_path.exists():
                    # Strategy 1: Prepend openfake-annotation to the path
                    image_path = Path("openfake-annotation") / image_rel_path
                
                if not image_path.exists():
                    # Strategy 2: Replace 'datasets' prefix with 'openfake-annotation/datasets'
                    if image_rel_path.startswith("datasets"):
                        image_path = Path("openfake-annotation") / image_rel_path
                
                if not image_path.exists():
                    # Strategy 3: Look in metadata parent's images folder
                    image_path = Path(metadata_path).parent / "images" / Path(image_rel_path).name
                
                if not image_path.exists():
                    errors.append(f"Not found: {image_rel_path}")
                    continue
                
                # Load image
                img = Image.open(image_path).convert("RGB")
                img_array = np.array(img)
                
                # Extract patches and compute 100 features (multi-stat pooling)
                # This matches the training procedure in preprocess_pure_style.py with --multi-stat
                patch_size = 512
                stride = 512
                patches = extract_patches(img_array, patch_size, stride)
                
                # Extract 25 features from each patch
                patch_feats = [style_extractor(p, normalize=True) for p in patches]
                patch_feats = np.stack(patch_feats, axis=0)  # Shape: (n_patches, 25)
                
                # Multi-stat pooling: mean+std+max+min across patches results in 100 features
                mean_vec = np.mean(patch_feats, axis=0)  # (25,)
                std_vec = np.std(patch_feats, axis=0)    # (25,)
                max_vec = np.max(patch_feats, axis=0)    # (25,)
                min_vec = np.min(patch_feats, axis=0)    # (25,)
                style_vec = np.concatenate([mean_vec, std_vec, max_vec, min_vec])  # (100,)
                
                # Run through trained model to get prediction
                style_tensor = torch.tensor(style_vec, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(style_tensor)
                    prob_real = torch.sigmoid(logits).item()
                
                prob_fake = 1 - prob_real
                prediction = "real" if prob_real > 0.5 else "fake"
                confidence = max(prob_real, prob_fake)
                
                # Create feature dictionary with multi-stat labels
                # 100 features = 25 base features × 4 statistics (mean, std, max, min)
                features_dict = {}
                for i, name in enumerate(feature_names):
                    features_dict[f"{name}_mean"] = float(mean_vec[i])
                    features_dict[f"{name}_std"] = float(std_vec[i])
                    features_dict[f"{name}_max"] = float(max_vec[i])
                    features_dict[f"{name}_min"] = float(min_vec[i])
                
                # Compile training example
                training_example = {
                    "image_id": Path(image_path).stem,
                    "image_path": str(image_path),
                    "caption": item["caption"],
                    "true_label": item["true_label"],
                    "similarity": item.get("similarity", 0.0),
                    
                    # Model predictions (from pure_style.pt)
                    "prediction": prediction,
                    "confidence": confidence,
                    "prob_real": prob_real,
                    "prob_fake": prob_fake,
                    "logit": float(logits.item()),
                    
                    # Style features (from PureStyleExtractor)
                    "style_features": features_dict,
                    "style_features_vector": style_vec.tolist(),
                }
                
                # Write to file immediately (JSONL format - one JSON object per line)
                output_file.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                output_file.flush()  # Ensure it's written to disk
                
                # Update statistics
                num_processed += 1
                predictions_count[prediction] += 1
                true_labels_count[item["true_label"]] += 1
                
            except Exception as e:
                errors.append(f"{item.get('path', 'unknown')}: {str(e)}")
                continue
    
    finally:
        # Always close the file, even if there's an error
        output_file.close()
    
    # Summary
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Successfully processed: {num_processed}/{len(metadata)} images")
    
    if errors:
        print(f"Errors: {len(errors)}")
        if len(errors) <= 5:
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"  (showing first 5 of {len(errors)})")
            for err in errors[:5]:
                print(f"  - {err}")
    
    print(f"\nSaved to: {output_path} (JSONL format)")
    
    # Show example by reading first line
    if num_processed > 0:
        print()
        print("="*60)
        print("EXAMPLE ENTRY (first processed image)")
        print("="*60)
        with open(output_path, "r", encoding="utf-8") as f:
            example = json.loads(f.readline())
        print(f"Image: {example['image_id']}")
        print(f"Caption: {example['caption'][:80]}...")
        print(f"\nTrue label: {example['true_label']}")
        print(f"Predicted: {example['prediction']} ({example['confidence']:.1%} confidence)")
        print(f"\nStyle features (first 5 of 100):")
        for i, (name, value) in enumerate(list(example['style_features'].items())[:5]):
            print(f"  {i+1}. {name}: {value:.4f}")
        print(f"  ... ({len(example['style_features']) - 5} more features)")
        print("="*60)
    
    # Stats
    print()
    print("PREDICTION STATISTICS")
    print("="*60)
    print(f"Predicted as real: {predictions_count['real']}")
    print(f"Predicted as fake: {predictions_count['fake']}")
    print(f"\nTrue labels:")
    print(f"  Real: {true_labels_count['real']}")
    print(f"  Fake: {true_labels_count['fake']}")
    
    # Accuracy on this subset
    correct = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry['prediction'] == entry['true_label']:
                correct += 1
    
    accuracy = correct / num_processed if num_processed > 0 else 0
    print(f"\nAccuracy on this subset: {accuracy:.1%}")
    
    print()
    print("="*60)
    print("READY FOR LLM TRAINING")
    print("="*60)
    print(f"Data format: JSONL (one JSON object per line)")
    print(f"To convert to standard JSON array, run:")
    print(f"  python -c \"import json; data=[json.loads(l) for l in open('{output_path}')]; json.dump(data, open('{output_path.replace('.jsonl', '.json')}', 'w'), indent=2)\"")
    print()


if __name__ == "__main__":
    main()
