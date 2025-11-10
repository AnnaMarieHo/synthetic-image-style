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


# --- Template (no numeric anchors, encourages visual reasoning) ---
PROMPT_TEMPLATE = """
Analyze this image for authenticity assessment.

The model has detected several style anomalies in this image.
These irregularities are derived from internal measurements that contributed to the model’s decision.

Model classification: {prediction} ({confidence:.1f}% confidence)

Write a concise, technical explanation (200–300 words) that:
1. References visible visual elements in the image.
2. Conceptually connects these elements to the detected anomalies.
3. Explains why the observed cues support a {prediction} classification. ({domain_hint}).
4. Explains why these visible cues align with the stated {prediction} classification.
5. Illustrate your explanations using **image specific** visual elements.
4. write 3 concluding sentences describing the visual elements that illustrate the detected style anomalies.
"""


for idx in tqdm(subset, desc=f"Task {task_id}"):
    ex = data[idx]

    # --- Select most statistically significant features ---
    style_features = ex.get("style_features", {})
    if style_features:
        # Sort by magnitude of deviation
        sorted_feats = sorted(style_features.items(), key=lambda x: abs(x[1]), reverse=True)
        top_feats = sorted_feats[:6]

        domain_directions = []

        for k, v in top_feats:
            val = float(v)
            if val > 1.5:
                direction = "high"
            elif val < -1.5:
                direction = "low"
            else:
                direction = "normal"

            # map feature to conceptual domain
            name = k.lower()
            if "color_correlation" in name or "color_saturation" in name:
                domain = "color balance and saturation"
            elif "lab_a" in name or "lab_b" in name:
                domain = "hue asymmetry and color bias"
            elif "high_freq" in name or "mid_freq" in name or "freq_falloff" in name:
                domain = "frequency detail and smoothness"
            elif "spectral_entropy" in name:
                domain = "spectral diversity and fine detail distribution"
            elif "glcm_contrast" in name:
                domain = "local contrast structure"
            elif "glcm_energy" in name or "glcm_homogeneity" in name or "lbp_entropy" in name:
                domain = "microtexture regularity and surface uniformity"
            elif "edge_density" in name or "edge_coherence" in name:
                domain = "edge coherence and outline sharpness"
            elif "gradient_skewness" in name or "gradient_std" in name or "gradient_mean" in name:
                domain = "gradient distribution and shading balance"
            elif "noise" in name:
                domain = "noise texture and sensor realism"
            else:
                domain = "general visual cue"

            # Combine with direction
            if direction != "normal":
                domain_directions.append(f"{direction} {domain}")
            else:
                domain_directions.append(domain)

        # Deduplicate while preserving order
        seen = set()
        domain_directions = [d for d in domain_directions if not (d in seen or seen.add(d))]
        domain_hint = ", ".join(domain_directions[:4]) if domain_directions else "various visual cues"
    else:
        domain_hint = "general visual cues"

    # --- Normalize and fix image path ---
    raw_path = ex["image_path"].replace("\\", "/")

    if "openfake-annotation/datasets/" in raw_path:
        raw_path = raw_path.split("openfake-annotation/datasets/")[-1]

    if not raw_path.startswith("fake_balanced_filtered/"):
        raw_path = os.path.join("fake_balanced_filtered", raw_path)

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

    # --- Construct the final prompt ---
    user_prompt = PROMPT_TEMPLATE.format(
        caption=caption.strip(),
        prediction=prediction,
        confidence=confidence,
        domain_hint=domain_hint
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
        output = model.generate(
            **inputs,
            max_new_tokens=700,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    reasoning = processor.decode(output[0], skip_special_tokens=True)
    ex["llm_reasoning"] = reasoning

    print(f"Processed {os.path.basename(img_path)}")

out_path = file_in.replace(".json", f"_annotated_{task_id}.json")
with open(out_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nSaved technical explanations for {len(subset)} images → {out_path}\n")


