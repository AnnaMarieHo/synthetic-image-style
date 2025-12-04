"""
Extract captions between <START></START> tags from JSON files.

Reads caption JSON files and extracts only the text between <START> and </START> tags,
saving cleaned captions to a new JSON file.
"""
import json
import re
import os
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_start_content(text):
    """
    Extract text between <START> and </START> tags.
    
    Args:
        text: Input text that may contain <START></START> tags
    
    Returns:
        Extracted text between tags, or original text if no tags found
    """
    if not isinstance(text, str):
        return text
    


    if re.search(r'</think>', text, re.IGNORECASE):       
        # Take everything after <START>
        print("there is a think tag")
        parts = re.split(r'</think>', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[-1].strip()
    
        return text.strip()

    else:
        pattern = r'<START>(.*?)</START>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Take the first match (or concatenate all if multiple)
            extracted = matches[0].strip()
            return extracted
    
    # No <START> tags found, return original text
    return text.strip()


def process_json_file(input_path, output_path=None):
    """
    Process a JSON file and extract <START> tag content from all captions.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (default: adds '_cleaned' before extension)
    
    Returns:
        Dictionary with cleaned captions
    """
    logger.info(f"Processing {input_path}...")
    
    # Generate output path if not provided
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_cleaned{input_path_obj.suffix}")
    
    # Load JSON file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {input_path}: {e}")
        return None
    
    logger.info(f"Loaded {len(data)} entries from {input_path}")
    
    # Process each entry
    cleaned_data = {}
    extracted_count = 0
    no_tags_count = 0
    total_chars_before = 0
    total_chars_after = 0
    
    for key, value in data.items():
        if not isinstance(value, str):
            # Keep non-string values as-is
            cleaned_data[key] = value
            continue
        
        total_chars_before += len(value)
        
        # Extract content between <START> tags
        cleaned_value = extract_start_content(value)
        
        # Handle None case (shouldn't happen with fixed function, but safety check)
        if cleaned_value is None:
            cleaned_value = value
            logger.warning(f"extract_start_content returned None for key {key}, using original value")
        
        total_chars_after += len(cleaned_value)
        
        # Track statistics
        if "<START>" in value.upper() or "</START>" in value.upper():
            extracted_count += 1
        else:
            no_tags_count += 1
        
        cleaned_data[key] = cleaned_value
    
    # Save cleaned data
    logger.info(f"Saving cleaned captions to {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(cleaned_data)} cleaned captions to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {e}")
        return None
    

    logger.info(f"Total entries processed: {len(data)}")
    logger.info(f"Entries with tags: {extracted_count}")
    logger.info(f"Entries without  tags: {no_tags_count}")
    logger.info(f"Total characters before: {total_chars_before:,}")
    logger.info(f"Total characters after: {total_chars_after:,}")
    logger.info(f"Characters removed: {total_chars_before - total_chars_after:,}")
    logger.info(f"Average chars per entry (before): {total_chars_before // max(len(data), 1):,}")
    logger.info(f"Average chars per entry (after): {total_chars_after // max(len(data), 1):,}")
    
    return cleaned_data


def main():
    """Main function to process JSON files"""
    import sys

    # Default files to process
    default_files = [
        "annotated/merged_metadata_fake.json",
    ]
    
    # Get input files from command line or use defaults
    if len(sys.argv) > 1:
        input_files = sys.argv[1:]
    else:
        # Check which default files exist
        input_files = [f for f in default_files if os.path.exists(f)]
        if not input_files:
            logger.warning("No input files found. Usage: python extract_start_tags.py <file1.json> [file2.json] ...")
            logger.info("Or place one of these files in the current directory:")
            for f in default_files:
                logger.info(f"  - {f}")
            return
    
    # Process each file
    for input_file in input_files:
        if not os.path.exists(input_file):
            logger.warning(f"File not found: {input_file}, skipping...")
            continue
        
        logger.info("")
        process_json_file(input_file)
        logger.info("")
    


if __name__ == "__main__":
    main()

