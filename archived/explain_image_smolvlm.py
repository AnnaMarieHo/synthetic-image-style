"""
Inference using SmolVLM + Style Conditioning

Usage:
    python explain_image_smolvlm.py <image_path>
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


# ============================================================
# Extract style features
# ============================================================
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


def extract_style_features_with_importance(image_path):
    """Extract features and compute gradient-based importance (same as training script)"""
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
    
    print(f"  Processed {len(patches)} patches")
    print(f"  Prob Fake: {prob_fake:.3f} ({'FAKE' if prob_fake > 0.5 else 'REAL'})")
    
    # ============================================================
    # GRADIENT-BASED FEATURE IMPORTANCE
    # ============================================================
    print("  Computing feature importance using gradients...")
    
    x = torch.tensor(style_vec, dtype=torch.float32, device=device).unsqueeze(0)
    x.requires_grad_(True)
    
    logit_fake = classifier(x)
    loss = logit_fake.sum()
    
    classifier.zero_grad()
    loss.backward()
    
    grads = x.grad.detach()[0]
    importance = (grads * x[0]).abs().detach().cpu().numpy()
    
    # Base feature names
    base_feature_names = [
        "color_correlation_gb", "color_correlation_rb", "color_correlation_rg",
        "color_saturation_var", "edge_coherence", "edge_density", "freq_falloff",
        "glcm_contrast_1", "glcm_contrast_3", "glcm_contrast_5", "glcm_energy_1",
        "glcm_homogeneity_1", "gradient_mean", "gradient_skewness", "gradient_std",
        "high_freq_energy", "lab_a_skewness", "lab_b_skewness", "lbp_entropy",
        "mid_freq_energy", "noise_kurtosis", "noise_local_var", "noise_skewness",
        "noise_variance", "spectral_entropy"
    ]
    
    # Build full feature names
    if style_dim == 100:
        feature_names = []
        for base in base_feature_names:
            feature_names.extend([f"{base}_mean", f"{base}_std", f"{base}_max", f"{base}_min"])
    else:
        feature_names = base_feature_names
    
    # Get top features by gradient importance
    top_idx = np.argsort(-importance)[:10]
    
    top_features = []
    for i in top_idx[:5]:
        if i < 25:
            top_features.append((feature_names[i], style_vec[i], importance[i]))
        else:
            base_idx = (i - 25) % 25
            stat_type = ["std", "max", "min"][(i - 25) // 25]
            top_features.append((f"{base_feature_names[base_idx]}_{stat_type}", style_vec[i], importance[i]))
    
    # Compute top pairs
    top_pairs = []
    for a in range(min(5, len(top_idx))):
        for b in range(a + 1, min(5, len(top_idx))):
            i1, i2 = top_idx[a], top_idx[b]
            coherency = importance[i1] * importance[i2]
            
            if i1 < len(feature_names) and i2 < len(feature_names):
                name1 = feature_names[i1]
                name2 = feature_names[i2]
            else:
                continue
            
            top_pairs.append({
                "features": [name1, name2],
                "coherency": float(coherency)
            })
    
    top_pairs.sort(key=lambda x: x["coherency"], reverse=True)
    top_pairs = top_pairs[:3]
    
    print(f"  [OK] Identified top {len(top_features)} important features\n")
    
    return style_vec, prob_fake, top_features, top_pairs


# ============================================================
# Load model
# ============================================================
def load_model():
    print("\nLoading SmolVLM...")
    
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
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Load fine-tuned LoRA
    print("  Loading fine-tuned LoRA weights...")
    model = PeftModel.from_pretrained(base_model, f"{checkpoint_dir}/model")
    model.eval()
    
    print("  Model loaded!\n")
    
    return model, processor


# ============================================================
# Generate explanation
# ============================================================
@torch.no_grad()
def generate_explanation(image_path, prob_fake, top_features, top_pairs, model, processor):
    """Generate explanation using gradient-identified important features"""
    print("Generating explanation...")
    
    img = Image.open(image_path).convert("RGB")
    
    prediction = "FAKE" if prob_fake > 0.5 else "REAL"
    
    # Format top contributing features (by gradient importance)
    top_features_str = "\n".join([
        f"  • {name}: {val:.3f} (importance: {importance:.3f})"
        for name, val, importance in top_features
    ])
    
    # Format top feature interactions
    pairs_str = "\n".join([
        f"  • {pair['features'][0]} + {pair['features'][1]}"
        for pair in top_pairs
    ])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"""Look at this image. The classifier says it's {prob_fake:.1%} likely to be {prediction}.

The classifier identified these specific features as MOST IMPORTANT for this decision:

KEY FEATURES:
{top_features_str}

FEATURE INTERACTIONS:
{pairs_str}

Based on what you SEE in the image, explain in 2-3 sentences how these feature irregularities might manifest visually."""}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process
    inputs = processor(
        images=[img],
        text=text,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Generate (deterministic)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,  # Deterministic output
    )
    
    # Decode
    generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "Assistant:" in generated_text or "ASSISTANT:" in generated_text:
        explanation = generated_text.split("ssistant:")[-1].strip()
    else:
        # Fallback: take everything after the prompt
        explanation = generated_text.split(text)[-1].strip()
    
    return explanation


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python explain_image_smolvlm.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    print("="*70)
    print("IMAGE ANALYSIS (SmolVLM + MLP Feature Importance)")
    print("="*70)
    print(f"Image: {image_path}\n")
    
    # Extract style features and compute gradient-based importance
    style_features, prob_fake, top_features, top_pairs = extract_style_features_with_importance(image_path)
    
    # Load model
    model, processor = load_model()
    
    # Generate explanation
    print("="*70)
    explanation = generate_explanation(image_path, prob_fake, top_features, top_pairs, model, processor)
    
    # Display
    print("\n" + "="*70)
    print("ANALYSIS RESULT (SmolVLM + Gradient-Based Features)")
    print("="*70)
    print(f"\nPrediction: {'FAKE' if prob_fake > 0.5 else 'REAL'}")
    print(f"Confidence: {max(prob_fake, 1-prob_fake):.1%}")
    print(f"Prob Fake: {prob_fake:.3f}")
    
    print("\n" + "-"*70)
    print("TOP FEATURES (by gradient importance):")
    print("-"*70)
    for name, val, importance in top_features:
        print(f"  {name}: {val:.3f} (importance: {importance:.3f})")
    
    print("\n" + "-"*70)
    print("TOP FEATURE PAIRS:")
    print("-"*70)
    for pair in top_pairs:
        print(f"  [{pair['features'][0]}, {pair['features'][1]}]")
    
    print("\n" + "-"*70)
    print("EXPLANATION (Vision + Feature-Guided):")
    print("-"*70)
    print(explanation)
    print("\n" + "="*70)
    print("SmolVLM SEES the image AND is guided by the MLP's")
    print("gradient-identified important features!")
    print("="*70)
   

if __name__ == "__main__":
    main()


   

if __name__ == "__main__":
    main()

