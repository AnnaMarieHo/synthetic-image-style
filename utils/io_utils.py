"""
I/O utilities for JSON merging and data handling.

This module provides functions for:
- Merging multiple JSON files into a single consolidated file
- Handling both list and dict JSON structures
"""

import json
import os
from typing import List, Tuple


def merge_json_files(directory: str, output_filename: str, exclude_files: List[str] = None) -> dict:
    """
    Merge all JSON files in a directory into a single file.
    
    Handles both list and dict JSON structures. For dicts, keys are merged
    (with last value winning for duplicates). For lists, items are merged
    if they're dicts.
    
    Args:
        directory: Directory containing JSON files to merge
        output_filename: Name of output file (will be placed in directory)
        exclude_files: List of filenames to exclude from merging
    
    Returns:
        Dictionary with merged data
    """
    if exclude_files is None:
        exclude_files = []
    
    # Add output file to exclusions to avoid circular merge
    exclude_files.append(output_filename)
    
    # Collect all JSON file paths (sorted for consistency)
    json_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".json") and f not in exclude_files
    ])
    
    print(f"Found {len(json_files)} JSON files to merge in {directory}")
    
    merged_data = {}
    total_keys = 0
    duplicate_keys = []
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                
                # Handle list structure
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                if key in merged_data:
                                    duplicate_keys.append((file_path, key))
                                merged_data[key] = value
                                total_keys += 1
                
                # Handle dict structure
                elif isinstance(data, dict):
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
    
    # Save merged data
    output_path = os.path.join(directory, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nMerged {len(json_files)} files")
    print(f"Total unique keys: {len(merged_data)}")
    print(f"Total keys processed: {total_keys}")
    
    if duplicate_keys:
        print(f"Warning: Found {len(duplicate_keys)} duplicate keys (last value kept)")
        if len(duplicate_keys) <= 10:
            for file_path, key in duplicate_keys:
                print(f"  - {key} from {os.path.basename(file_path)}")
        else:
            print(f"  (showing first 10 of {len(duplicate_keys)})")
            for file_path, key in duplicate_keys[:10]:
                print(f"  - {key} from {os.path.basename(file_path)}")
    
    print(f"Saved to: {output_path}")
    
    return merged_data


def main():
    """CLI interface for merging JSON files."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python io_utils.py <directory> <output_filename> [exclude_file1 exclude_file2 ...]")
        print("\nExample:")
        print("  python io_utils.py ./data merged_output.json")
        print("  python io_utils.py ./data merged.json checkpoint.json temp.json")
        sys.exit(1)
    
    directory = sys.argv[1]
    output_filename = sys.argv[2]
    exclude_files = sys.argv[3:] if len(sys.argv) > 3 else None
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    merge_json_files(directory, output_filename, exclude_files)


if __name__ == "__main__":
    main()

