"""
Visual Grounding: Map Feature Interpretations to Image Regions

Takes Qwen 2.5 1.5B Instruct feature explanations and uses Qwen-VL to identify WHERE
in the image these statistical anomalies manifest visually.
"""
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from explain_features_with_interactions import (
    extract_style_features_and_interactions,
    load_text_llm,
    interpret_features_with_context
)
from patches_and_gradcam.patch_importance import (
    compute_patch_gradcam,
    get_important_patch_locations
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print("\nLoading base Qwen2-VL...")
    
    # Try 7B first for better reasoning, fallback to 2B if VRAM is insufficient
    # The 7B model should provide more detailed region-level analysis
    # Using base models (not fine-tuned)
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print(f"  Using {model_id}")
    except Exception as e:
        print(f"  Failed to load 7B model: {e}")
        print("  Falling back to 2B model...")
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Use 4-bit quantization for large model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    model.eval()
    print("  Model loaded!\n")
    
    return model, processor


def ground_explanation_to_image(image_path, feature_explanation, prob_fake, model, processor):
    """Use Qwen-VL to identify visual regions matching the feature explanation"""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    prediction = "DEEPFAKE" if prob_fake > 0.5 else "AUTHENTIC"
    
    # Use the cleaned feature explanation from the LLM
    # The explanation already includes quadrants and feature interactions with values
    prompt = f"""A statistical analysis of this image determined it is {prediction}. 

The analysis identified specific feature interactions and the quadrants where they occur:

{feature_explanation}

Based on this statistical analysis, identify and describe the visual regions in the image that correspond to the quadrants mentioned above. Focus on describing the visual characteristics (texture, color patterns, edges, etc.) in those specific regions without making assumptions about semantic content (people, objects, faces, etc.)."""
    
    # Qwen-VL uses a specific message format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]
    
    # Apply chat template and prepare inputs
    # Qwen2-VL expects messages to be processed through apply_chat_template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Prepare inputs - Qwen2-VL processes images separately
    inputs = processor(
        text=text,
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,  # Increased for more detailed analysis
            do_sample=False,
        )
    
    # Decode - extract only the generated part (remove input tokens)
    # inputs is a dict, so access input_ids as a key
    input_ids = inputs['input_ids']
    
    # Handle both single and batch cases
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if generated_ids.dim() == 1:
        generated_ids = generated_ids.unsqueeze(0)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    
    generated_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return generated_text.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visual_grounding.py <image_path>")
        print("  Uses base Qwen 2.5 1.5B Instruct and base Qwen2-VL models")
        return
    
    image_path = sys.argv[1]
    # Always use base models (not fine-tuned)
    use_finetuned = False
    
    print("="*70)
    print("VISUAL GROUNDING: Feature Interpretation → Image Regions")
    print("="*70)
    print(f"Image: {image_path}\n")
    
    # Step 1: Get feature interpretation from base Qwen 2.5 1.5B Instruct
    print("="*70)
    print("STEP 1: Feature Interpretation (Base Qwen 2.5 1.5B Instruct)")
    print("="*70)
    
    # Extract features and get all return values including patch data for GradCAM
    result = extract_style_features_and_interactions(image_path, device)
    features, prob_fake, top_features, top_pairs = result[:4]
    
    # Unpack additional data for GradCAM
    patch_locations_info = None
    if len(result) > 4:
        img_array, patch_locations, patch_feats, top_idx, importance = result[4:]
        
        # Load classifier for GradCAM computation
        checkpoint = torch.load("checkpoints/pure_style_512.pt", map_location=device)
        style_dim = checkpoint.get("style_dim", 25)
        from models.mlp_classifier import PureStyleClassifier
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
    
    model_qwen, tokenizer = load_text_llm(use_finetuned=use_finetuned)
    
    # Get feature explanation with patch location info
    feature_explanation = interpret_features_with_context(
        features, prob_fake, top_features, top_pairs, model_qwen, tokenizer, [], patch_locations_info
    )
    
    print("\n" + "-"*70)
    print("FEATURE INTERPRETATION:")
    print("-"*70)
    print(feature_explanation)
    print()
    
    # Step 2: Visual grounding with base VLM
    print("="*70)
    print("STEP 2: Visual Grounding (Base Qwen2-VL)")
    print("="*70)
    
    vlm_model, vlm_processor = load_model()
    
    visual_regions = ground_explanation_to_image(
        image_path, 
        feature_explanation,
        prob_fake,
        vlm_model, 
        vlm_processor
    )
    
    # Display results
    print("\n" + "="*70)
    print("VISUAL REGION ANALYSIS")
    print("="*70)
    print(f"\nPrediction: {'FAKE' if prob_fake > 0.5 else 'REAL'}")
    print(f"Confidence: {max(prob_fake, 1-prob_fake):.1%}\n")
    
    print("-"*70)
    print("STATISTICAL ANALYSIS (Base Qwen 2.5 1.5B Instruct):")
    print("-"*70)
    print(feature_explanation)
    
    print("\n" + "-"*70)
    print("VISUAL REGIONS (Base Qwen2-VL):")
    print("-"*70)
    print(visual_regions)


if __name__ == "__main__":
    main()

