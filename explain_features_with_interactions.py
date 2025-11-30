"""
Feature Interpreter with Interaction Analysis

Uses the feature interactions (pairs/triplets) that contributed to the MLP's decision
to generate targeted explanations.

Can work with 450 samples via few-shot prompting or fine-tuning.
"""
import sys
import re
import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from peft import PeftModel
import matplotlib.pyplot as plt
from transformers import BitsAndBytesConfig        
from models.mlp_classifier import PureStyleClassifier
from patches_and_gradcam.prompts import get_inference_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from patches_and_gradcam.extract_features import extract_style_features_and_interactions, load_pair_frequencies
from patches_and_gradcam.patch_importance import (
    compute_patch_gradcam,
    save_gradcam_visualization,
    create_gradcam_heatmap,
    format_patch_locations, 
    get_important_patch_locations
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_text_llm(use_finetuned=True):
    """Load a text-only LLM (Qwen 2.5 1.5B Instruct or fine-tuned Qwen)"""
    if use_finetuned:
        print("Loading FINE-TUNED Qwen 2.5 1.5B Instruct (Feature Interpreter)...\n")
        # Load with PEFT for LoRA weights

        model_path = "trained_qwen2.5_1.5b_feature_interpreter/model"
        tokenizer_path = "trained_qwen2.5_1.5b_feature_interpreter/tokenizer"
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model with quantization (matching training script)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        
    else:
        print("Loading base Qwen 2.5 1.5B Instruct (with few-shot prompting)...\n")
        
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load with quantization for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    model.eval()
    print("Model loaded!\n")
    
    return model, tokenizer



def interpret_features_with_context(features, prob_fake, top_pairs, model, tokenizer, few_shot_examples, patch_locations_info=None):
    """Generate explanation using gradient-computed top contributing features"""
    prediction = "FAKE" if prob_fake > 0.5 else "REAL"
    
    # Format feature interactions as JSON string (matching training format exactly)
    # Use top 2 pairs
    # Each pair must have: features, coherency, and values
    validated_pairs = []
    for pair in top_pairs[:2]:
        # Ensure all required fields are present (matching training validation)
        if "features" not in pair or len(pair.get("features", [])) != 2:
            continue
        if "coherency" not in pair:
            continue
        if "values" not in pair or len(pair.get("values", [])) != 2:
            continue
        
        # Ensure coherency and values are numeric
        try:
            validated_pairs.append({
                "features": pair["features"],
                "coherency": float(pair["coherency"]),
                "values": [float(v) for v in pair["values"]]
            })
        except (ValueError, TypeError):
            continue
    
    if not validated_pairs:
        raise ValueError("No valid feature pairs found with features, coherency, and values")
    
    # Format interactions as JSON string (matching training format)
    interactions_json = json.dumps({"top_pairs": validated_pairs}, indent=2, ensure_ascii=False)
    
    # Extract individual features from pairs for top values (matching training script format)
    feature_scores = {}
    for pair in validated_pairs:
        for feat in pair['features']:
            if feat not in feature_scores:
                # Get actual value from pair
                if 'values' in pair and len(pair['values']) == 2:
                    feat_idx = pair['features'].index(feat)
                    feature_scores[feat] = pair['values'][feat_idx]
                else:
                    # Fallback to coherency if values not available
                    feature_scores[feat] = pair.get('coherency', 0.0)
    

    prompt = get_inference_prompt(prob_fake, interactions_json)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,  # Deterministic
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Only decode the newly generated tokens (exclude the input prompt)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text.strip()



def main():

    GLOBAL_PAIR_FREQ = load_pair_frequencies()

    if len(sys.argv) < 2:
        print("Usage: python explain_features_with_interactions.py <image_path> [--finetuned]")
        print("  --finetuned: Use fine-tuned model (no few-shot examples needed)")
        return
    
    image_path = sys.argv[1]

    # Use base model instead of fine-tuned
    use_finetuned = True #"--finetuned" in sys.argv
    
    # Extract features and compute gradient-based importance
    result = extract_style_features_and_interactions(image_path, device)
    features, prob_fake, top_pairs = result[:3]
    
    # Unpack additional data for GradCAM if available
    patch_locations_info = None
    if len(result) > 4:

        img_array, patch_locations, patch_feats, top_idx, importance = result[3:]
        
        # Load classifier again for GradCAM (needed for gradient computation)
        checkpoint = torch.load("checkpoints/pure_style_512.pt", map_location=device)
        style_dim = checkpoint.get("style_dim", 25)

        classifier = PureStyleClassifier(style_dim=style_dim).to(device)
        classifier.load_state_dict(checkpoint["model"])
        classifier.eval()
        
        # Compute GradCAM
        patch_importance = compute_patch_gradcam(
            classifier, patch_feats, top_idx, importance, style_dim, device
        )
        
        # Get important patch locations
        patch_locations_info = get_important_patch_locations(
            patch_locations, patch_importance, img_array.shape, patch_size=512
        )
        
        # Create visualization
        overlay, heatmap = create_gradcam_heatmap(
            img_array, patch_locations, patch_importance, patch_size=512, stride=512
        )
        
        # Save visualization
        save_gradcam_visualization(image_path, img_array, overlay, heatmap)
    
    # Load text LLM
    model, tokenizer = load_text_llm(use_finetuned=use_finetuned)
    
    explanation = interpret_features_with_context(
        features, prob_fake, top_pairs, model, tokenizer, patch_locations_info
    )
    

    sentences = explanation.strip().split('\n\n')
    extracted_sentences = sentences[0:2]

    # Join them back together with a double newline for clean formatting
    result = '\n'.join(extracted_sentences)

    print("FEATURE-BASED EXPLANATION")
    print(f"\nPrediction: {'FAKE' if prob_fake > 0.5 else 'REAL'}")
    print(f"Confidence: {max(prob_fake, 1-prob_fake):.1%}")
    print(f"Prob Fake: {prob_fake:.3f}")
    
    print("\n")
    print("TOP FEATURE PAIRS (with coherency and values):")
    for pair in top_pairs[:2]:  # Show top 5 pairs
        if 'features' in pair and 'coherency' in pair and 'values' in pair:
            print(f"  [{pair['features'][0]}, {pair['features'][1]}]")
            print(f"    coherency: {pair['coherency']:.4f}")
            print(f"    values: [{pair['values'][0]:.4f}, {pair['values'][1]:.4f}]")
    
    print("EXPLANATION:")
    print(result)
    


if __name__ == "__main__":
    main()

