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

from models.mlp_classifier import PureStyleClassifier
from models.style_extractor_pure import PureStyleExtractor
from utils.patch_utils import extract_patches, aggregate_patch_features
from utils.model_utils import load_classifier
from utils.feature_utils import build_feature_names
from utils.config_loader import Config, get_device, ensure_multi_stat


def main():
    """
    Extract style features + predictions for LLM training data generation.
    
    Uses:
    1. PureStyleExtractor to extract 25 base style features per patch
    2. Multi-stat pooling to create 100D feature vectors
    3. Trained classifier to predict fake/real
    
    Output: JSONL with features + predictions
    """
    # Ensure multi-stat is configured
    ensure_multi_stat()
    
    # Configuration from config.yaml
    metadata_path = Config.metadata_path()
    checkpoint_path = Config.checkpoint()
    output_path = "openfake-annotation/datasets/combined/llm_training_data_real.jsonl"
    patch_size = Config.patch_size()
    stride = Config.stride()
    
    device = get_device()

    print(f"Device: {device}")
    print(f"Metadata: {metadata_path}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Output: {output_path} (JSONL - incremental saving)")
    print(f"Patch size: {patch_size}, Stride: {stride}")
    print(f"Pooling: Multi-stat (100D ONLY)")
    print()
    
    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} images from metadata")
    
    style_extractor = PureStyleExtractor(device)
    feature_names = style_extractor.get_feature_names()
    print(f"Initialized PureStyleExtractor ({len(feature_names)} features)")
    
    # Load trained model using utility function
    model, style_dim = load_classifier(checkpoint_path, device)
    print(f"Loaded trained model (style_dim={style_dim})")
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
                    image_path = Path("openfake-annotation") / image_rel_path
                
                if not image_path.exists():
                    if image_rel_path.startswith("datasets"):
                        image_path = Path("openfake-annotation") / image_rel_path
                
                if not image_path.exists():
                    image_path = Path(metadata_path).parent / "images" / Path(image_rel_path).name
                
                if not image_path.exists():
                    errors.append(f"Not found: {image_rel_path}")
                    continue
                
                # Load image
                img = Image.open(image_path).convert("RGB")
                img_array = np.array(img)
                
                # Extract patches using config values
                patches, patch_locations = extract_patches(img_array, patch_size, stride)
                
                # Extract 25 base features from each patch (ALWAYS normalized)
                patch_feats = [style_extractor(p, normalize=True) for p in patches]
                patch_feats = np.stack(patch_feats, axis=0)  # Shape: (n_patches, 25)
                
                #  use multi-stat pooling (100D)
                style_vec = aggregate_patch_features(patch_feats, use_multi_stat=True)
                
                # Run through trained model to get prediction
                style_tensor = torch.tensor(style_vec, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(style_tensor)
                    prob_real = torch.sigmoid(logits).item()
                
                prob_fake = 1 - prob_real
                prediction = "real" if prob_real > 0.5 else "fake"
                confidence = max(prob_real, prob_fake)
                
                # Create feature dictionary - always 100D multi-stat
                full_feature_names = build_feature_names(style_dim)
                features_dict = {name: float(val) for name, val in zip(full_feature_names, style_vec)}
                
                # Compile training example
                training_example = {
                    "image_id": Path(image_path).stem,
                    "image_path": str(image_path),
                    "caption": item["caption"],
                    "true_label": item["true_label"],
                    "similarity": item.get("similarity", 0.0),
                    
                    # Model predictions
                    "prediction": prediction,
                    "confidence": confidence,
                    "prob_real": prob_real,
                    "prob_fake": prob_fake,
                    "logit": float(logits.item()),
                    
                    # Style features (from PureStyleExtractor)
                    "style_features": features_dict,
                    "style_features_vector": style_vec.tolist(),
                }
                
                # Write to file immediately 
                output_file.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                output_file.flush()  # Ensure it's written
                
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
    print("RESULTS")
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
        print("EXAMPLE ENTRY (first processed image)")
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
      
    # Stats
    print()
    print("PREDICTION STATISTICS")
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
    print("READY FOR LLM TRAINING")
    print(f"Data format: JSONL (one JSON object per line)")
    print()


if __name__ == "__main__":
    main()
