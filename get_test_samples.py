import json
import os

def get_unused_samples(json_path, max_samples_used):
    """
    Loads data, mimics the filtering/sampling logic of the training code,
    and returns the metadata for all samples that were NOT used for training.
    """
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use data.items() to get the key (sample ID) along with the item (metadata)
    items = list(data.items())
    
    validated_items = []
    
    for key, item in items:
        
        # Skip if no top_pairs or no caption
        if "top_pairs" not in item or not item.get("top_pairs") or "caption" not in item or not item.get("caption"):
            continue
            
        # The training code limits to the top 5 pairs for processing
        raw_top_pairs = item["top_pairs"][:5]
        top_pairs = []
        
        for pair in raw_top_pairs:
            # Ensure all required fields are present and correctly typed
            if "features" not in pair or len(pair.get("features", [])) != 2:
                continue
            if "coherency" not in pair or "values" not in pair or len(pair.get("values", [])) != 2:
                continue
            
            try:
                # Ensure coherency and values are numeric
                float(pair["coherency"])
                [float(v) for v in pair["values"]]
            except (ValueError, TypeError):
                continue
            
            top_pairs.append(pair) # Keep the validated pair
        
        # Skip if no valid pairs after validation
        if not top_pairs:
            continue
        
        
        # If the item passed validation, we track it
        validated_items.append((key, item))


    # The training code used the first `max_samples_used` VALIDATED items.
    # The remaining items are the unused test set.
    unused_items = validated_items[max_samples_used:]
    
    # Convert unused items back to a dictionary format {ID: metadata}
    unused_data = {key: item for key, item in unused_items}
    
    return unused_data

MAX_SAMPLES_USED_IN_TRAINING = 900

# File names
FAKE_TRAIN_FILE = "merged_metadata_fake.json"
REAL_TRAIN_FILE = "merged_metadata_real.json"

output_fake_file = "test_metadata_fake_unused.json"
output_real_file = "test_metadata_real_unused.json"

# Get unused fake samples
unused_fake_data = get_unused_samples(FAKE_TRAIN_FILE, MAX_SAMPLES_USED_IN_TRAINING)
unused_fake_count = len(unused_fake_data)

# Get unused real samples
unused_real_data = get_unused_samples(REAL_TRAIN_FILE, MAX_SAMPLES_USED_IN_TRAINING)
unused_real_count = len(unused_real_data)

total_unused_count = unused_fake_count + unused_real_count

# Save the unused data to new JSON files for the user to use as a test set
with open(output_fake_file, 'w') as f:
    json.dump(unused_fake_data, f, indent=2)

with open(output_real_file, 'w') as f:
    json.dump(unused_real_data, f, indent=2)

print(f"Unused Fake Samples: {unused_fake_count}")
print(f"Unused Real Samples: {unused_real_count}")
print(f"Total Unused Samples (Test Set): {total_unused_count}")
print(f"Unused fake metadata saved to: {output_fake_file}")
print(f"Unused real metadata saved to: {output_real_file}")