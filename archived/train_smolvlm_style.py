"""
SmolVLM + Style Conditioning

SmolVLM is a small (2B) vision-language model from HuggingFace
that was actually trained to understand images!

Perfect for your 6GB GPU + image-specific explanations.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")


# ============================================================
# 1. Load SmolVLM
# ============================================================
print("Loading SmolVLM-Instruct...")
model_id = "HuggingFaceTB/SmolVLM-Instruct"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# Prepare model for k-bit training (QLoRA)
print("  Preparing model for k-bit training...")
base_model = prepare_model_for_kbit_training(base_model)

# Freeze vision encoder explicitly
print("  Freezing vision encoder...")
for name, param in base_model.named_parameters():
    if "vision" in name.lower():
        param.requires_grad = False

# Add LoRA to language model
print("  Adding LoRA to language model...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

base_model = get_peft_model(base_model, lora_config)
base_model.print_trainable_parameters()

model_device = next(base_model.parameters()).device
print(f"[OK] SmolVLM loaded on device: {model_device}")


# ============================================================
# 2. Dataset
# ============================================================
class SmolVLMStyleDataset(Dataset):
    def __init__(self, json_path, max_samples=None):
        print(f"\nLoading dataset from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data if max_samples is None else data[:max_samples]
        print(f"[OK] Loaded {len(self.data)} samples\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img = Image.open(ex["image_path"]).convert("RGB")
        
        # Build instruction with style features
        prob_fake = ex["prob_fake"]
        prediction = "FAKE" if prob_fake > 0.5 else "REAL"
        
        # Get the 25D feature vector (not the dict)
        # style_features_vector is 100D (mean+std+max+min), take first 25 (mean)
        features = ex["style_features_vector"][:25]  # 25D mean feature vector
        if isinstance(features, str):
            features = [float(x) for x in features.strip('[]').split(',')][:25]
        elif isinstance(features[0], str):
            features = [float(f) for f in features][:25]
        
        # Format features for readability
        feature_names = [
            "Texture Consistency", "Edge Quality", "Noise Pattern", "Color Correlation",
            "Frequency Content", "Gradient Smoothness", "Local Variance", "Contrast Distribution",
            "Sharpness", "Blur Artifacts", "Compression Artifacts", "Color Balance",
            "Shadow Consistency", "Highlight Behavior", "Skin Texture", "Face Symmetry",
            "Boundary Artifacts", "Spatial Coherence", "Temporal Stability", "Detail Preservation",
            "Natural Lighting", "Reflection Consistency", "Depth Perception", "Material Properties",
            "Overall Realism"
        ]
        
        # Create feature summary (show top signals)
        feature_str = "\n".join([f"  • {name}: {val:.3f}" for name, val in zip(feature_names, features)])
        
        # SmolVLM format: User message with image + features
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""Analyze this image for deepfake artifacts using the technical measurements below.

TECHNICAL MEASUREMENTS:
{feature_str}

CLASSIFIER OUTPUT: {prob_fake:.1%} probability of being fake (predicted: {prediction})

TASK: Interpret these technical measurements in the context of the image. Explain which features indicate whether this image is real or fake, and what visual artifacts or qualities they correspond to in the actual image."""}
                ]
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": ex["llm_reasoning"]}]
            }
        ]
        
        return img, messages


# ============================================================
# 3. Custom data collator (processes data for SmolVLM)
# ============================================================
class SmolVLMDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
    
    def __call__(self, batch):
        images, messages_list = zip(*batch)
        
        # Apply chat template to get formatted text
        full_texts = []
        assistant_responses = []
        
        for msg in messages_list:
            # Full conversation
            full_text = self.processor.apply_chat_template(msg, add_generation_prompt=False)
            full_texts.append(full_text)
            
            # Extract just assistant's response text
            assistant_responses.append(msg[1]["content"][0]["text"])
        
        # Process images + texts
        inputs = self.processor(
            images=list(images),
            text=full_texts,
            return_tensors="pt",
            padding=True,
        )
        
        # Create labels by masking everything except assistant response
        labels = inputs["input_ids"].clone()
        
        for i in range(len(batch)):
            # Tokenize the assistant's response to find it in the full sequence
            assistant_text = assistant_responses[i]
            assistant_tokens = self.tokenizer.encode(assistant_text, add_special_tokens=False)
            
            # Find where assistant response starts in the full token sequence
            full_tokens = inputs["input_ids"][i].tolist()
            
            # Search for the assistant tokens in the full sequence
            found = False
            for start_idx in range(len(full_tokens) - len(assistant_tokens) + 1):
                if full_tokens[start_idx:start_idx + len(assistant_tokens)] == assistant_tokens:
                    # Mask everything before assistant response
                    labels[i, :start_idx] = -100
                    found = True
                    break
            
            # If not found by exact match, mask first 80% as a fallback
            if not found:
                mask_len = int(len(full_tokens) * 0.8)
                labels[i, :mask_len] = -100
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        inputs["labels"] = labels
        
        return inputs


data_collator = SmolVLMDataCollator(processor)

# Use base_model directly (no wrapper needed!)
model = base_model


# ============================================================
# 4. Training
# ============================================================
class MemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            torch.cuda.empty_cache()
        return control


train_json = "train_balanced_10k_cleaned.json"
dataset = SmolVLMStyleDataset(train_json, max_samples=1000)

args = TrainingArguments(
    output_dir="smolvlm_style_interpreter",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    save_total_limit=2,
    warmup_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
    callbacks=[MemoryCallback()],
)

print("\n" + "="*70)
print("SmolVLM + Style Conditioning")
print("="*70)
print("Why SmolVLM:")
print("  - Small (2B params) - perfect for 6GB GPU")
print("  - Vision-native - actually trained on images!")
print("  - Modern architecture from HuggingFace")
print("  - Understands image content natively")
print("\nTraining:")
print("  [+] LoRA in language model: TRAINABLE")
print("  [-] Vision encoder: FROZEN")
print("\nKey Advantage:")
print("  SmolVLM already knows how to interpret images,")
print("  unlike Phi-2 which is text-only!")
print("="*70 + "\n")

print("Starting training...")
trainer.train()

# Save
output_dir = "trained_smolvlm_interpreter"
os.makedirs(output_dir, exist_ok=True)

base_model.save_pretrained(f"{output_dir}/model")
processor.save_pretrained(f"{output_dir}/processor")

config = {
    "model_id": model_id,
    "style_conditioned": True,
}
with open(f"{output_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "="*70)
print("[OK] Training complete!")
print(f"[OK] Model saved to: {output_dir}/model")
print(f"[OK] Processor saved to: {output_dir}/processor")
print("\nThis model should generate image-specific explanations")
print("because SmolVLM actually understands visual content!")
print("="*70)

