import json

# Load main annotation dataset
with open("llm_training_data_fakes.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# Load feature interactions
with open("mlp_interactions_with_coherency_fake.json", "r", encoding="utf-8") as f:
    interactions = json.load(f)

# Merge step - only keep samples that have interactions
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
with open("llm_training_interactions_fakes.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

