
import os, argparse, pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from itertools import islice

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="ComplexDataLab/OpenFake")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--num_samples", type=int, default=20000)
parser.add_argument("--out_dir", type=str, default="datasets/test")
args = parser.parse_args()

real_dir = os.path.join(args.out_dir, "real")
fake_dir = os.path.join(args.out_dir, "fake")

for d in [real_dir, fake_dir]:
    os.makedirs(os.path.join(d, "images"), exist_ok=True)

print(f"Streaming {args.dataset_name} ({args.split}) ...")
dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)

first = next(iter(dataset))
print("Example keys:", list(first.keys()))

img_col = "image"
cap_col = "prompt" if "prompt" in first else "caption"
label_col = "label" if "label" in first else None

dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)

records_real, records_fake = [], []
subset_iter = islice(dataset, args.num_samples)
print(f"Collecting and saving up to {args.num_samples} samples...")

for i, sample in enumerate(tqdm(subset_iter)):
    try:
        img = sample[img_col]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        label = str(sample[label_col]) if label_col else "unknown"
        caption = sample.get(cap_col, "")

        subdir = real_dir if label.lower() == "real" else fake_dir
        filename = f"{i:05d}.jpg"
        path = os.path.join(subdir, "images", filename)
        img.save(path)

        record = {"path": path, "caption": caption, "true_label": label}
        if label.lower() == "real":
            records_real.append(record)
        else:
            records_fake.append(record)

    except Exception as e:
        print(f"Skipping {i}: {e}")

pd.DataFrame(records_real).to_json(os.path.join(real_dir, "metadata.json"), orient="records", indent=2)
pd.DataFrame(records_fake).to_json(os.path.join(fake_dir, "metadata.json"), orient="records", indent=2)

print(f"Saved {len(records_real)} real and {len(records_fake)} fake samples.")
print(f"Output directory: {args.out_dir}")
