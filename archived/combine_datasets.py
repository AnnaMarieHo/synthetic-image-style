import json
import os

# Load both metadata files
with open("openfake-annotation/datasets/openfake_raw/metadata.json", "r") as f:
    real_data = json.load(f)
    
with open("openfake-annotation/datasets/fake_balanced_filtered/metadata.json", "r") as f:
    fake_data = json.load(f)

print(f"Loaded {len(real_data)} real samples and {len(fake_data)} fake samples")

# Combine
combined = real_data + fake_data

output_dir = "openfake-annotation/datasets/combined"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "cache"), exist_ok=True)


output_path = os.path.join(output_dir, "metadata.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2)

print(f"  Saved combined metadata to {output_path}")
print(f"  Total samples: {len(combined)}")
print(f"  Real: {sum(1 for s in combined if s['true_label'] == 'real')}")
print(f"  Fake: {sum(1 for s in combined if s['true_label'] == 'fake')}")
print(f"\nNext steps:")
print(f"  1. Update preprocess_embeddings.py to use: {output_path}")
print(f"  2. Run: python scripts/preprocess_embeddings.py")

