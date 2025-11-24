"""
Generate DeepSeek R1 Distill Llama 8B captions for training samples.

Loads training data, formats prompts with feature interactions, values, and quadrants,
then generates explanations using base DeepSeek R1 Distill Llama 8B.
Saves results as {image_id: explanation} for merging back into the dataset.
"""
import json
import re
import torch
import logging
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from patches_and_gradcam.prompts import get_fake_prompt, get_real_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get SLURM array task ID for sharding
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"caption_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)


def format_patch_locations_simple(patch_info):
    """
    Format patch locations directly from the JSON format.
    Simpler version that doesn't require y, x coordinates.
    """
    if not patch_info or "important_regions" not in patch_info:
        return ""
    
    important_regions = patch_info["important_regions"]
    lines = []
    
    if important_regions.get("high"):
        lines.append("HIGH IMPORTANCE QUADRANTS:")
        for region in important_regions["high"]:
            loc_desc = region.get("location", "")
            imp = region.get("importance", 0.0)
            lines.append(f"  - {loc_desc} quadrant (importance: {imp:.2f})")
        lines.append("")
    
    if important_regions.get("medium_high"):
        lines.append("MEDIUM-HIGH IMPORTANCE QUADRANTS:")
        for region in important_regions["medium_high"]:
            loc_desc = region.get("location", "")
            imp = region.get("importance", 0.0)
            lines.append(f"  - {loc_desc} quadrant (importance: {imp:.2f})")
        lines.append("")
    
    if important_regions.get("medium_low"):
        lines.append("MEDIUM-LOW IMPORTANCE QUADRANTS:")
        for region in important_regions["medium_low"]:
            loc_desc = region.get("location", "")
            imp = region.get("importance", 0.0)
            lines.append(f"  - {loc_desc} quadrant (importance: {imp:.2f})")
        lines.append("")
    
    return "\n".join(lines)


def format_interactions_as_json(top_pairs):
    """Format feature interactions as raw JSON object (uncleaned)"""
    # Return the top pairs as a JSON object with all fields (features, coherency, values)
    interactions_json = {
        "top_pairs": top_pairs[:5]  # Top 5 pairs with all their data
    }
    return json.dumps(interactions_json, indent=2, ensure_ascii=False)


def format_top_features(style_features, top_pairs):
    """
    Extract top 3 feature values from the top pairs.
    Since we don't have pre-computed top_features, we'll use features from pairs.
    """
    feature_scores = {}
    for pair in top_pairs[:5]:
        for feat_name in pair['features']:
            if feat_name not in feature_scores:
                # Get the value from the pair
                feat_idx = pair['features'].index(feat_name)
                feature_scores[feat_name] = pair['values'][feat_idx]
    
    # Format top 3
    top_items = sorted(feature_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    return "\n".join([
        f"  {feat} = {val:.2f}"
        for feat, val in top_items
    ])


def generate_explanation(prompt, model, tokenizer):
    """Generate explanation from prompt using DeepSeek R1 Distill Llama 8B"""
    try:
        # Format as messages for chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32768,
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part (skip the input)
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        # Log the FULL LLM output for debugging/monitoring
        logger.debug(f"Full LLM output ({len(generated_text)} chars):\n{generated_text}")
        
        # Return full output - <START> tag filtering will be done in post-processing
        return generated_text.strip()
    except Exception as e:
        logger.error(f"Error in generate_explanation: {e}", exc_info=True)
        raise


def process_samples(json_path, model, tokenizer, max_samples=None, checkpoint_interval=10):
    """
    Process samples from JSON file and generate explanations.
    
    Args:
        json_path: Path to JSON file
        model: DeepSeek R1 Distill Llama 8B model
        tokenizer: DeepSeek R1 Distill Llama 8B tokenizer
        max_samples: Maximum number of samples to process
        checkpoint_interval: Save checkpoint every N samples
    
    Returns:
        Dictionary mapping image_id to explanation
    """
    logger.info(f"Loading data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} total samples from {json_path}")
    
    results = {}
    processed = 0
    skipped = 0
    errors = 0
    
    # If max_samples is None, process all samples
    if max_samples is None:
        max_samples = len(data)
    
    logger.info(f"Processing up to {max_samples} samples...")
    
    # Create checkpoint file path
    checkpoint_file = json_path.replace(".json", "_checkpoint_captions.json")
    
    # Try to load existing checkpoint
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading checkpoint from {checkpoint_file}...")
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed = len(results)
            logger.info(f"Resumed from checkpoint: {processed} samples already processed")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
    
    for idx, item in enumerate(data):
        if processed >= max_samples:
            logger.info(f"Reached max_samples limit ({max_samples})")
            break
        
        image_id = item.get("image_id")
        if not image_id:
            skipped += 1
            continue
        
        # Skip if already processed (from checkpoint)
        if image_id in results:
            processed += 1
            if processed % 50 == 0:
                logger.info(f"Progress: {processed}/{max_samples} processed, {skipped} skipped, {errors} errors")
            continue
        
        # Skip if no feature interactions
        if "feature_interactions" not in item:
            logger.debug(f"Skipping {image_id}: no feature_interactions")
            skipped += 1
            continue
        
        feature_interactions = item["feature_interactions"]
        top_pairs = feature_interactions.get("top_pairs", [])
        
        if not top_pairs:
            logger.debug(f"Skipping {image_id}: no top_pairs")
            skipped += 1
            continue
        
        # Get probability
        prob_fake = item.get("prob_fake", 0.5)
        prob_real = item.get("prob_real", 1 - prob_fake)
        
        # Format interactions as raw JSON object (uncleaned)
        interactions_json = format_interactions_as_json(top_pairs)
        
        # Format top features (using style_features if available)
        style_features = item.get("style_features", {})
        top_features_str = format_top_features(style_features, top_pairs)
        
        # Format patch locations
        patch_info = feature_interactions.get("patch_info", {})
        location_str = format_patch_locations_simple(patch_info)
        
        # Generate prompt - pass JSON object instead of formatted string
        if prob_fake > 0.5:
            prompt = get_fake_prompt(prob_fake, interactions_json, top_features_str, location_str)
        else:
            prompt = get_fake_prompt(prob_real, interactions_json, top_features_str, location_str)
        
        # Generate explanation
        try:
            explanation = generate_explanation(prompt, model, tokenizer)
            results[image_id] = explanation
            processed += 1
            
            # Log progress
            if processed % 1 == 0:
                logger.info(f"Progress: {processed}/{max_samples} processed | Sample {image_id} | {len(explanation)} chars")
            
            print(explanation)
            # Save checkpoint periodically
            if processed % checkpoint_interval == 0:
                
                logger.info(f"Saving checkpoint at {processed} samples...")
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Checkpoint saved to {checkpoint_file}")
                
        except Exception as e:
            errors += 1
            logger.error(f"Error processing {image_id}: {e}", exc_info=True)
            continue
    
    logger.info(f"Completed processing: {processed} processed, {skipped} skipped, {errors} errors")
    return results


def main():
    logger.info("="*70)
    logger.info("GENERATE TRAINING CAPTIONS WITH DEEPSEEK R1 DISTILL LLAMA 8B")
    logger.info(f"SLURM Array Task ID: {task_id}")
    logger.info("="*70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Device: {device}")
    
    # Determine input file based on task_id
    # Assuming you have shards named like: llm_training_data_00.json, llm_training_data_01.json, etc.
    # Or: fakes/shards/llm_training_data_00.json
    shard_file = f"real/shards/llm_training_data_{task_id:02d}.json"
    
    # Fallback to check if file exists in current directory
    if not os.path.exists(shard_file):
        shard_file = f"llm_training_data_{task_id:02d}.json"
    
    if not os.path.exists(shard_file):
        logger.error(f"Shard file not found: {shard_file}")
        logger.error("Please ensure shard files exist or update the file path pattern")
        return
    
    logger.info(f"Processing shard: {shard_file}")
    
    # Load base DeepSeek R1 Distill Llama 8B model (once, not in a loop)
    logger.info("Loading base DeepSeek R1 Distill Llama 8B...")
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load primary model: {e}", exc_info=True)
        return
    
    start_time = datetime.now()
    
    # Process the shard (process all samples in the shard)
    logger.info("="*70)
    logger.info(f"PROCESSING SHARD {task_id}")
    logger.info("="*70)
    try:
        results = process_samples(
            shard_file,
            model, tokenizer,
            max_samples=None,  # Process all samples in the shard
            checkpoint_interval=10
        )
        logger.info(f"Shard {task_id}: {len(results)} explanations generated")
    except Exception as e:
        logger.error(f"Error processing shard {task_id}: {e}", exc_info=True)
        results = {}
    
    # Save results with task_id in filename
    output_path = f"deepseek_generated_captions_{task_id:02d}.json"
    logger.info(f"Saving {len(results)} explanations to {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)
    
    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time
    
    logger.info("="*70)
    logger.info("COMPLETE!")
    logger.info("="*70)
    logger.info(f"Shard {task_id}: Generated {len(results)} explanations")
    logger.info(f"Total time: {elapsed_time}")
    logger.info(f"Average time per sample: {elapsed_time.total_seconds() / max(len(results), 1):.2f} seconds")
    logger.info(f"Output saved to: {output_path}")
    logger.info("\nAfter all shards complete, merge them using image_id as the key.")
    logger.info("Note: Generated captions use DeepSeek R1 Distill Llama 8B model")
    logger.info("="*70)


if __name__ == "__main__":
    main()

