import json
import os

directory = "./"

# Collect all JSON file paths (sorted for consistency)
json_files = sorted(
    [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
)

print(f"Found {len(json_files)} JSON files to merge")

merged_data = {}
total_keys = 0
duplicate_keys = []

for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            # Handle both cases: file contains a list or a single dict
            if isinstance(data, list):
                # If list, merge all dictionaries in the list
                for item in data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if key in merged_data:
                                duplicate_keys.append((file_path, key))
                            merged_data[key] = value
                            total_keys += 1
            elif isinstance(data, dict):
                # If dict, merge it directly
                for key, value in data.items():
                    if key in merged_data:
                        duplicate_keys.append((file_path, key))
                    merged_data[key] = value
                    total_keys += 1
            else:
                print(f"Warning: {file_path} contains neither a dict nor a list, skipping...")
        except json.JSONDecodeError as e:
            print(f"Error: Skipping {file_path} due to JSON error: {e}")
        except Exception as e:
            print(f"Error: Failed to process {file_path}: {e}")

# Output file
output_path = os.path.join(directory, "merged_metadata_real.json")

# Save combined data
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"\nMerged {len(json_files)} files")
print(f"Total unique keys: {len(merged_data)}")
print(f"Total keys processed: {total_keys}")
if duplicate_keys:
    print(f"Warning: Found {len(duplicate_keys)} duplicate keys (last value kept)")
    if len(duplicate_keys) <= 10:
        for file_path, key in duplicate_keys:
            print(f"  - {key} from {os.path.basename(file_path)}")
    else:
        print(f"  (showing first 10)")
        for file_path, key in duplicate_keys[:10]:
            print(f"  - {key} from {os.path.basename(file_path)}")
print(f"Saved to: {output_path}")
