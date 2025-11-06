import os
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from tqdm import tqdm

print("Loading model...")
local_path = "./llava"

processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    local_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

file_in = "fake_balanced_filtered/llm_training_data.json"
if len(os.sys.argv) > 1:
    file_in = os.sys.argv[1]

print(f"Loading {file_in}...")
with open(file_in, "r") as f:
    data = json.load(f)

unprocessed = [i for i, ex in enumerate(data) if "llm_reasoning" not in ex]
print(f"Total examples: {len(data)}, unprocessed: {len(unprocessed)}")

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
subset = unprocessed[:5]
print(f"Processing chunk {task_id}: {len(subset)} examples")


PROMPT_TEMPLATE = """

This image shows: {caption}

Most significant style anomalies:
{feature_lines}

Model classification: {prediction} ({confidence:.1f}% confidence)

Generate a brief, technical explanation (~200–300 words) that:
1. References visible visual elements in the image
2. Connects them to the anomalies listed above
3. Explains why these suggest a {prediction_lower} image
4. Focuses only on the features most relevant to authenticity assessment
"""

for idx in tqdm(subset, desc=f"Task {task_id}"):
    ex = data[idx]


# --- Select most statistically significant features ---
    style_features = ex.get("style_features", {})
    if style_features:
        # Sort features by absolute deviation (largest anomalies first)
        sorted_feats = sorted(style_features.items(), key=lambda x: abs(x[1]), reverse=True)
        top_feats = sorted_feats[:6]  # top 6 most anomalous

        def describe_feature(k, v):
            if v > 1.5:
                status = "HIGH"
            elif v < -1.5:
                status = "LOW"
            else:
                status = "NORMAL"
            return f"- {k}: {v:.3f} ({status})"

            feature_lines = "\n".join([describe_feature(k, v) for k, v in top_feats])
    else:
        feature_lines = "- No style features available"




    # Normalize and fix image path
    raw_path = ex["image_path"].replace("\\", "/")

    # Remove any unwanted prefix from earlier dataset structure
    if "openfake-annotation/datasets/" in raw_path:
        raw_path = raw_path.split("openfake-annotation/datasets/")[-1]

    # Ensure we reference the actual cluster folder
    if not raw_path.startswith("fake_balanced_filtered/"):
        raw_path = os.path.join("fake_balanced_filtered", raw_path)

    # Absolute path within your working directory
    img_path = os.path.join(os.getcwd(), raw_path)

    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(file_in), img_path)

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {img_path}: {e}")
        continue

    # --- Extract values ---
    caption = ex.get("caption", "")
    prediction = ex.get("prediction", "unknown").upper()
    confidence = ex.get("confidence", 0) * 100
    prob_real = ex.get("prob_real", 0.0)
    prob_fake = ex.get("prob_fake", 0.0)
    style_features = ex.get("style_features", {})

    # --- Format feature section nicely ---
    # Sort by feature name for readability
    feature_lines = "\n".join(
        [f"- {k}: {v:.3f}" for k, v in sorted(style_features.items())]
    ) if style_features else "- No style features available"

    # --- Construct the final prompt ---
    user_prompt = PROMPT_TEMPLATE.format(
        caption=caption.strip(),
        feature_lines=feature_lines,
        prediction=prediction,
        confidence=confidence,
        prob_real=prob_real,
        prob_fake=prob_fake,
        prediction_lower=prediction.lower()
    )

    # --- Prepare multimodal input ---
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=img, return_tensors="pt").to("cuda")

    # --- Generate reasoning ---
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=700)

    reasoning = processor.decode(output[0], skip_special_tokens=True)
    ex["llm_reasoning"] = reasoning

    print(f"Processed {os.path.basename(img_path)}")

out_path = file_in.replace(".json", f"_annotated_{task_id}.json")
with open(out_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nSaved technical explanations for {len(subset)} images → {out_path}\n")
