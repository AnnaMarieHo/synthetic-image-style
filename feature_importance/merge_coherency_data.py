"""
Merge coherency data with LLM training annotations.

This script combines the main annotation dataset with feature interaction data,
keeping only samples that have both annotations and interactions.
"""
import json
import os

# Configuration
DATA_DIR = "feature_importance"
LLM_DATA_PATH = os.path.join(DATA_DIR, "llm_training_data/llm_training_data_fakes.json")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "coherency/mlp_interactions_with_coherency_fake.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "llm_training_interactions_fakes.json")

print(f"Loading annotations from: {LLM_DATA_PATH}")
print(f"Loading interactions from: {INTERACTIONS_PATH}")

# Load main annotation dataset
with open(LLM_DATA_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)

# Load feature interactions
with open(INTERACTIONS_PATH, "r", encoding="utf-8") as f:
    interactions = json.load(f)

# Merge step: only keep samples that have interactions
merged = []

for ex in samples:
    img_id = ex.get("image_id")
    
    # Only process samples that have interactions
    if img_id in interactions:
        ex["feature_interactions"] = interactions[img_id]
        merged.append(ex)

print(f"Total samples in dataset: {len(samples)}")
print(f"Total interactions available: {len(interactions)}")
print(f"Merged samples: {len(merged)}")

# Save merged output
print(f"\nSaving to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print("Merge complete")

