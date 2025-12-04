import json
import pandas as pd

# Load JSONL into DataFrame
path = "openfake-annotation/datasets/combined/llm_training_data.jsonl"
records = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
df = pd.DataFrame(records)

# Split by true_label
real_df = df[df["true_label"] == "real"]
fake_df = df[df["true_label"] == "fake"][:9966]

# Save to separate JSONL files
real_df.to_json("real_samples.jsonl", orient="records", lines=True, force_ascii=False)
fake_df.to_json("fake_samples.jsonl", orient="records", lines=True, force_ascii=False)

print(f"Saved {len(real_df)} real samples and {len(fake_df)} fake samples.")
