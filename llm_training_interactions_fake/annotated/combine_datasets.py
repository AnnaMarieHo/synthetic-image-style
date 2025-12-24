"""
Merge JSON files for fake image annotations.

This script uses the consolidated merge utility to combine all JSON annotation
files in the current directory into a single merged_metadata_fake.json file.
"""
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.io_utils import merge_json_files

if __name__ == "__main__":
    directory = os.path.dirname(__file__)
    merge_json_files(directory, "merged_metadata_fake.json")
