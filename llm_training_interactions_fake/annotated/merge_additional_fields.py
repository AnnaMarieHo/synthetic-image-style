"""
Merge additional fields from llm_training_data_fakes.json and 
mlp_interactions_with_coherency_fake.json into merged_metadata_fake.json
"""
import json
import os
from pathlib import Path

# Paths
base_dir = Path(__file__).parent
annotated_dir = base_dir
feature_dir = base_dir.parent.parent / "feature_importance"

merged_metadata_path = annotated_dir / "merged_metadata_fake_cleaned.json"
llm_training_data_path = feature_dir / "llm_training_data_fakes.json"
mlp_interactions_path = feature_dir / "mlp_interactions_with_coherency_fake.json"
output_path = annotated_dir / "merged_metadata_fake.json"

print("Loading files...")

# Load merged metadata (dictionary with image_id keys)
with open(merged_metadata_path, "r", encoding="utf-8") as f:
    merged_data = json.load(f)
    # Handle case where it's a list containing a dict
    if isinstance(merged_data, list) and len(merged_data) > 0:
        if isinstance(merged_data[0], dict):
            merged_data = merged_data[0]
        else:
            print("Error: merged_metadata_fake.json has unexpected structure")
            exit(1)

print(f"Loaded {len(merged_data)} entries from merged_metadata_fake.json")

# Load LLM training data (list of objects with image_id field)
with open(llm_training_data_path, "r", encoding="utf-8") as f:
    llm_data_list = json.load(f)

# Convert to dictionary keyed by image_id
llm_data_dict = {}
for item in llm_data_list:
    if "image_id" in item:
        llm_data_dict[item["image_id"]] = item

print(f"Loaded {len(llm_data_dict)} entries from llm_training_data_fakes.json")

# Load MLP interactions (dictionary with image_id keys)
with open(mlp_interactions_path, "r", encoding="utf-8") as f:
    mlp_data = json.load(f)

print(f"Loaded {len(mlp_data)} entries from mlp_interactions_with_coherency_fake.json")

# Fields to merge from llm_training_data_fakes.json
llm_fields_to_merge = [
    "true_label",
    "similarity",
    "prediction",
    "confidence",
    "prob_real",
    "prob_fake",
    "logit"
]

# Fields to merge from mlp_interactions_with_coherency_fake.json
mlp_fields_to_merge = [
    "top_pairs",
    "patch_info"
]

# Merge the data
merged_count = 0
llm_missing = 0
mlp_missing = 0

for image_id, caption_text in merged_data.items():
    # Convert caption from string to dict structure
    if isinstance(caption_text, str):
        merged_data[image_id] = {
            "caption": caption_text
        }
    elif not isinstance(merged_data[image_id], dict):
        merged_data[image_id] = {
            "caption": str(caption_text)
        }
    
    # Merge LLM training data fields
    if image_id in llm_data_dict:
        for field in llm_fields_to_merge:
            if field in llm_data_dict[image_id]:
                merged_data[image_id][field] = llm_data_dict[image_id][field]
        merged_count += 1
    else:
        llm_missing += 1
    
    # Merge MLP interactions fields
    if image_id in mlp_data:
        for field in mlp_fields_to_merge:
            if field in mlp_data[image_id]:
                merged_data[image_id][field] = mlp_data[image_id][field]
    else:
        mlp_missing += 1

print(f"\nMerge complete!")
print(f"  Entries with LLM data: {merged_count}")
print(f"  Entries missing LLM data: {llm_missing}")
print(f"  Entries missing MLP data: {mlp_missing}")

# Save merged data
print(f"\nSaving to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(merged_data)} entries to {output_path}")

