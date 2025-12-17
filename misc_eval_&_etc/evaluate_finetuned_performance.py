import sys
import re
import os
import cv2
import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
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
# --- Configuration ---
FAKE_TEST_FILE = "test_metadata_fake_unused.json"
REAL_TEST_FILE = "test_metadata_real_unused.json"
SAMPLE_SIZE = 30 # Number of samples for manual review
OUTPUT_FILE_NAME = "manual_review_samples.json" # New output file 
device = "cuda" if torch.cuda.is_available() else "cpu"



def load_text_llm(use_finetuned=True):
    """Load a text-only LLM (Qwen 2.5 1.5B Instruct or fine-tuned Qwen)"""
    if not use_finetuned:
        raise NotImplementedError("Base model loading not implemented in this evaluation script.")
        
    print("Loading FINE-TUNED Qwen 2.5 1.5B Instruct (Feature Interpreter)...\n")
    
    model_path = "trained_qwen2.5_1.5b_feature_interpreter/model"
    tokenizer_path = "trained_qwen2.5_1.5b_feature_interpreter/tokenizer"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with quantization 
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
    
    model.eval()
    print("Model loaded and set to evaluation mode\n")
    
    return model, tokenizer


def generate_explanation(model, tokenizer, item):
    """
    Generates explanation, mirroring the logic of interpret_features_with_context.
    Enforces top 2 pairs limit which mitigates hallucination for the finetuned model.
    """
    prob_fake = item.get("prob_fake", 0.5)
    top_pairs = item["top_pairs"]
    
    validated_pairs = []
    for pair in top_pairs[:2]:
        if "features" not in pair or len(pair.get("features", [])) != 2: continue
        if "coherency" not in pair: continue
        if "values" not in pair or len(pair.get("values", [])) != 2: continue
        
        try:
            validated_pairs.append({
                "features": pair["features"],
                "coherency": float(pair["coherency"]),
                "values": [float(v) for v in pair["values"]]
            })
        except (ValueError, TypeError):
            continue
    
    if not validated_pairs:
        # Return empty strings if generation is not possible
        return "", "" 

    # Format interactions as JSON string
    interactions_json = json.dumps({"top_pairs": validated_pairs}, indent=2, ensure_ascii=False)
    
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
    
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    del outputs 
    del inputs
    torch.cuda.empty_cache() 
    
    return generated_text.strip(), interactions_json


def main():
    GLOBAL_PAIR_FREQ = load_pair_frequencies()

    # Load Data 
    test_data = {}
    for file_path in [FAKE_TEST_FILE, REAL_TEST_FILE]:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_data.update(data)
        else:
            print(f"Warning: Test file not found: {file_path}. Evaluation aborted.")
            return

    if not test_data:
        print("Evaluation aborted: No test data found.")
        return

    # Sample the Data
    all_keys = list(test_data.keys())
    sample_ids = random.sample(all_keys, min(SAMPLE_SIZE, len(test_data)))
    test_subset = {id: test_data[id] for id in sample_ids}
    
    print(f"\n Starting Evaluation on Sample Size: {len(test_subset)} ")

    model, tokenizer = load_text_llm(use_finetuned=True)

    review_samples = []
    
    # Generate Outputs and Collect References
    print("\n Generating Captions ")
    for sample_id, item in tqdm(test_subset.items(), desc="Generation"):
        
        reference_text = item["caption"].strip()
        
        # Generate Model Output and get the JSON used for prompting
        model_output, raw_json_input = generate_explanation(model, tokenizer, item)
        
        
        ref_parts = reference_text.strip().split('\n\n')
        truncated_reference = '\n\n'.join(ref_parts[:3])

        gen_parts = model_output.strip().split('\n\n')
        truncated_generated = '\n\n'.join(gen_parts[:3])

        review_samples.append({
            "ID": sample_id,
            "Raw_Features": raw_json_input,
            "Reference_2_Sentences": truncated_reference,
            "Generated_2_Sentences": truncated_generated,
        })
        

    try:
        with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f:
            json.dump(review_samples, f, indent=4, ensure_ascii=False)
        print(f"\nEvaluation results saved to {OUTPUT_FILE_NAME} for manual review.")
    except Exception as e:
        print(f"\nERROR saving results to file: {e}")
    # Print Results for Manual Review
    print(f"       MANUAL FUNCTIONAL ACCURACY REVIEW ({len(review_samples)} Samples)")
        
    
if __name__ == "__main__":
    main()