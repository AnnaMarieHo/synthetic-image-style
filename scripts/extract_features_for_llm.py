"""
Extract style features + predictions for LLM training data generation.

Uses:
1. PureStyleExtractor → extract 25 deterministic style features
2. pure_style.pt (trained on 15k fake + 9966 real unfiltered) → predict fake/real

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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.style_extractor_pure import PureStyleExtractor
import torch.nn as nn


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
        """Takes 25 features as input, outputs 1 logit"""
        return self.net(style_features)


def main():
    # Configuration
    metadata_path = "openfake-annotation/datasets/fake_balanced_filtered/metadata.json"
    checkpoint_path = "checkpoints/pure_style.pt"
    output_path = "openfake-annotation/datasets/fake_balanced_filtered/llm_training_data.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("="*60)
    print(f"Device: {device}")
    print(f"Metadata: {metadata_path}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} images from metadata")
    
    # Load style extractor
    style_extractor = PureStyleExtractor(device)
    feature_names = style_extractor.get_feature_names()
    print(f"Initialized PureStyleExtractor ({len(feature_names)} features)")
    
    # Load trained model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    style_dim = checkpoint.get("style_dim", 25)
    
    model = PureStyleClassifier(style_dim=style_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded trained model (trained on 15k unfiltered fakes & 9966 real samples)")
    print()
    
    # Process all images
    training_data = []
    errors = []
    
    for idx, item in enumerate(tqdm(metadata)):
        try:
            # Resolve image path
            image_rel_path = item["path"]
            image_path = Path(image_rel_path)
            if not image_path.exists():
                image_path = Path("openfake-annotation") / Path(image_rel_path).relative_to(Path(image_rel_path).parts[0])
            if not image_path.exists():
                image_path = Path(metadata_path).parent / "images" / Path(image_rel_path).name
            
            if not image_path.exists():
                errors.append(f"Not found: {image_rel_path}")
                continue
            
            # Load image
            img = Image.open(image_path).convert("RGB")
            
            # Extract 25 style features 
            style_vec = style_extractor(np.array(img))
            
            # Run through trained model to get prediction
            style_tensor = torch.tensor(style_vec, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(style_tensor)
                prob_real = torch.sigmoid(logits).item()
            
            prob_fake = 1 - prob_real
            prediction = "real" if prob_real > 0.5 else "fake"
            confidence = max(prob_real, prob_fake)
            
            # Create feature dictionary
            features_dict = {
                name: float(value) 
                for name, value in zip(feature_names, style_vec)
            }
            
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
            
            training_data.append(training_example)
            
        except Exception as e:
            errors.append(f"{item.get('path', 'unknown')}: {str(e)}")
            continue
    
    # Summary
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Successfully processed: {len(training_data)}/{len(metadata)} images")
    
    if errors:
        print(f"Errors: {len(errors)}")
        if len(errors) <= 5:
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"  (showing first 5 of {len(errors)})")
            for err in errors[:5]:
                print(f"  - {err}")
    
    # Save
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {output_path}")
    
    # Show example
    if training_data:
        print()
        print("="*60)
        print("EXAMPLE ENTRY")
        print("="*60)
        example = training_data[0]
        print(f"Image: {example['image_id']}")
        print(f"Caption: {example['caption'][:80]}...")
        print(f"\nTrue label: {example['true_label']}")
        print(f"Predicted: {example['prediction']} ({example['confidence']:.1%} confidence)")
        print(f"\nStyle features (first 5 of 25):")
        for i, (name, value) in enumerate(list(example['style_features'].items())[:5]):
            print(f"  {i+1}. {name}: {value:.4f}")
        print(f"  ... ({len(example['style_features']) - 5} more features)")
        print("="*60)
    
    # Stats
    predictions = [x['prediction'] for x in training_data]
    true_labels = [x['true_label'] for x in training_data]
    
    print()
    print("PREDICTION STATISTICS")
    print("="*60)
    print(f"Predicted as real: {predictions.count('real')}")
    print(f"Predicted as fake: {predictions.count('fake')}")
    print(f"\nTrue labels:")
    print(f"  Real: {true_labels.count('real')}")
    print(f"  Fake: {true_labels.count('fake')}")
    
    # Accuracy on this subset
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    accuracy = correct / len(predictions) if predictions else 0
    print(f"\nAccuracy on this subset: {accuracy:.1%}")
    


if __name__ == "__main__":
    main()
