import json
import os

directory = "./"

# Collect all JSON file paths (sorted for consistency)
json_files = sorted(
    [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
)

print(f"Found {len(json_files)} JSON files to merge")

merged_data = []

for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            # Handle both cases: file contains a list or a single dict
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)
        except json.JSONDecodeError as e:
            print(f" Skipping {file_path} due to JSON error: {e}")

# Output file
output_path = os.path.join(directory, "merged_metadata_real.json")

# Save combined data
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"\nMerged {len(json_files)} files")
print(f"Total samples: {len(merged_data)}")
print(f"Saved to: {output_path}")
