"""
Compare Base SmolVLM vs Fine-tuned SmolVLM

Tests the same image with both models to see what the training added.
"""
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from peft import PeftModel
from models.style_extractor_pure import PureStyleExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_patches(image: np.ndarray, patch_size: int, stride: int):
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    if not patches:
        patches.append(cv2.resize(image, (patch_size, patch_size)))
    return patches


def extract_style_features(image_path):
    print("Extracting style features...")
    
    checkpoint = torch.load("checkpoints/pure_style.pt", map_location=device)
    style_dim = checkpoint.get("style_dim", 25)
    
    style_extractor = PureStyleExtractor(device)
    
    from models.mlp_classifier import PureStyleClassifier
    classifier = PureStyleClassifier(style_dim=style_dim).to(device)
    classifier.load_state_dict(checkpoint["model"])
    classifier.eval()
    
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    patches = extract_patches(img_array, 512, 512)
    patch_feats = [style_extractor(p, normalize=True) for p in patches]
    patch_feats = np.stack(patch_feats, axis=0)
    
    use_multi_stat = (style_dim == 100)
    if use_multi_stat:
        mean_vec = np.mean(patch_feats, axis=0)
        std_vec = np.std(patch_feats, axis=0)
        max_vec = np.max(patch_feats, axis=0)
        min_vec = np.min(patch_feats, axis=0)
        style_vec = np.concatenate([mean_vec, std_vec, max_vec, min_vec])
    else:
        style_vec = np.mean(patch_feats, axis=0)
    
    style_tensor = torch.tensor(style_vec, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = classifier(style_tensor)
        prob_real = torch.sigmoid(logits).item()
    
    prob_fake = 1 - prob_real
    
    print(f"  Prob Fake: {prob_fake:.3f} ({'FAKE' if prob_fake > 0.5 else 'REAL'})")
    
    return style_vec[:25], prob_fake


def load_base_model():
    """Load the base SmolVLM without fine-tuning"""
    print("\nLoading BASE SmolVLM (no fine-tuning)...")
    
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
        trust_remote_code=True,
    )
    
    print("  Base model loaded!")
    return base_model, processor


def load_finetuned_model():
    """Load the fine-tuned SmolVLM with LoRA"""
    print("\nLoading FINE-TUNED SmolVLM (with LoRA)...")
    
    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    checkpoint_dir = "trained_smolvlm_interpreter"
    
    processor = AutoProcessor.from_pretrained(
        f"{checkpoint_dir}/processor",
        trust_remote_code=True
    )
    
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
        trust_remote_code=True,
    )
    
    # Load LoRA weights
    finetuned_model = PeftModel.from_pretrained(
        base_model,
        f"{checkpoint_dir}/model"
    )
    
    print("  Fine-tuned model loaded!")
    return finetuned_model, processor


@torch.no_grad()
def generate_explanation(image_path, style_features, prob_fake, model, processor, model_name):
    print(f"\nGenerating explanation with {model_name}...")
    
    img = Image.open(image_path).convert("RGB")
    prediction = "FAKE" if prob_fake > 0.5 else "REAL"
    
    feature_names = [
        "color_correlation_gb", "color_correlation_rb", "color_correlation_rg",
        "color_saturation_var", "edge_coherence", "edge_density", "freq_falloff",
        "glcm_contrast_1", "glcm_contrast_3", "glcm_contrast_5", "glcm_energy_1",
        "glcm_homogeneity_1", "gradient_mean", "gradient_skewness", "gradient_std",
        "high_freq_energy", "lab_a_skewness", "lab_b_skewness", "lbp_entropy",
        "mid_freq_energy", "noise_kurtosis", "noise_local_var", "noise_skewness",
        "noise_variance", "spectral_entropy"
    ]
    
    # feature_str = "\n".join([f"  • {name}: {val:.3f}" for name, val in zip(feature_names, style_features)])
    feature_str = "\n".join([f"  • {name}: {val:.3f}" for name, val in zip(feature_names, style_features)])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"""Explain why this image might be a deep fake using the technical measurements below.

TECHNICAL MEASUREMENTS:
{feature_str}

CLASSIFIER OUTPUT: {prob_fake:.1%} probability of being fake (predicted: {prediction})"""}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(
        images=[img],
        text=text,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.1,      # Low temp for more deterministic output
        do_sample=False,      # Greedy decoding for consistency
    )
    
    generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
    
    if "Assistant:" in generated_text or "ASSISTANT:" in generated_text:
        explanation = generated_text.split("ssistant:")[-1].strip()
    else:
        explanation = generated_text.split(text)[-1].strip()
    
    return explanation


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_smolvlm_base_vs_finetuned.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    print("="*70)
    print("COMPARISON: Base SmolVLM vs Fine-tuned SmolVLM")
    print("="*70)
    print(f"Image: {image_path}\n")
    
    # Extract features once
    style_features, prob_fake = extract_style_features(image_path)
    
    # Load both models
    base_model, base_processor = load_base_model()
    finetuned_model, ft_processor = load_finetuned_model()
    
    # Generate with base model
    print("\n" + "="*70)
    base_explanation = generate_explanation(
        image_path, style_features, prob_fake, 
        base_model, base_processor, "BASE MODEL"
    )
    
    # Generate with fine-tuned model
    print("\n" + "="*70)
    ft_explanation = generate_explanation(
        image_path, style_features, prob_fake,
        finetuned_model, ft_processor, "FINE-TUNED MODEL"
    )
    
    # Display comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\nClassifier: {prob_fake:.1%} probability FAKE")
    
    print("\n" + "-"*70)
    print("BASE SmolVLM (no training):")
    print("-"*70)
    print(base_explanation)
    
    print("\n" + "-"*70)
    print("FINE-TUNED SmolVLM (your 24-hour training):")
    print("-"*70)
    print(ft_explanation)
    
    print("\n" + "="*70)
    print("ANALYSIS:")
    print("="*70)
    print("Look for differences in:")
    print("  • Does fine-tuned mention specific artifacts more?")
    print("  • Does fine-tuned reference texture/noise/edges differently?")
    print("  • Is fine-tuned more aligned with the features?")
    print("="*70)


if __name__ == "__main__":
    main()

