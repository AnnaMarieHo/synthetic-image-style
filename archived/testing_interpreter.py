import json, random
import pandas as pd

# with open("openfake-annotation/merged_metadata_fake.json", "r",encoding="utf-8") as f:
#     real = json.load(f)

# with open("openfake-annotation/merged_metadata_real.json", "r", encoding="utf-8") as f:
#     fake = json.load(f)

# # Shuffle + sample
# random.shuffle(real)
# random.shuffle(fake)

# real_subset = real[:5000]
# fake_subset = fake[:5000]

# train_data = real_subset + fake_subset
# random.shuffle(train_data)

# with open("train_balanced_10k.json", "w") as f:
#     json.dump(train_data, f, indent=2)


# Load the JSON file
with open("train_balanced_10k.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert list of dicts -> DataFrame
df = pd.DataFrame(data)
df["llm_reasoning"] = df["llm_reasoning"].str.split("ASSISTANT:").str[-1]

cleaned_data = df.to_dict(orient="records")

print(df.shape)
print(df.head())

with open("train_balanced_10k_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)