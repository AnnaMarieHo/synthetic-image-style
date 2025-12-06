import os, argparse, pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="ComplexDataLab/OpenFake")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--num_fakes", type=int, default=15000, help="Number of fake samples to download")
parser.add_argument("--out_dir", type=str, default="datasets/unfiltered_fakes")
args = parser.parse_args()

# Create output directory
images_dir = os.path.join(args.out_dir, "images")
os.makedirs(images_dir, exist_ok=True)

print(f"Streaming {args.dataset_name} ({args.split}) ...")
print(f"Target: {args.num_fakes} FAKE samples (no filtering)")
dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)

# Check structure
first = next(iter(dataset))
print("Example keys:", list(first.keys()))

img_col = "image"
cap_col = "prompt" if "prompt" in first else "caption"
label_col = "label" if "label" in first else None

# Restart iterator after inspection
dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)

records_fake = []
fake_count = 0


for i, sample in enumerate(tqdm(dataset)):
    # Stop if there arent enough fakes
    if fake_count >= args.num_fakes:
        break
    
    try:
        # Get label
        label = str(sample[label_col]) if label_col else "unknown"
        
        # Skip if real
        if label.lower() == "real":
            continue
        
        # Process fake sample
        img = sample[img_col]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        
        caption = sample.get(cap_col, "")
        
        # Save with sequential numbering
        filename = f"{fake_count:05d}.jpg"
        path = os.path.join(images_dir, filename)
        img.save(path)
        
        # Store relative path for consistency
        relative_path = os.path.join(args.out_dir, "images", filename)
        record = {
            "path": relative_path,
            "caption": caption,
            "true_label": "fake"
        }
        records_fake.append(record)
        fake_count += 1
        
    except Exception as e:
        print(f"Skipping sample {i}: {e}")
        continue

# Save metadata
metadata_path = os.path.join(args.out_dir, "metadata.json")
pd.DataFrame(records_fake).to_json(metadata_path, orient="records", indent=2)

print(f"Successfully downloaded {len(records_fake)} fake samples")
print(f"Output directory: {args.out_dir}")
print(f"Metadata: {metadata_path}")

