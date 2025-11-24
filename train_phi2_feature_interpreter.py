"""
Fine-tune Qwen 2.5 1.5B Instruct on Feature Interaction Dataset

Uses your domain-specific dataset (feature interactions) to fine-tune
Qwen 2.5 1.5B Instruct for deepfake feature interpretation.

This demonstrates:
- Fine-tuning (training algorithm design)
- Domain-specific dataset
- Understanding LLMs
"""
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from patches_and_gradcam.prompts import get_inference_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureInteractionDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=768, max_samples=None):
        print(f"Loading dataset from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.data = []
        
        # Handle dictionary structure (keys are image IDs)
        if isinstance(data, dict):
            items = data.values()
        else:
            items = data
        
        # Process each sample
        skipped_incomplete = 0
        for item in items:
            # Skip if no top_pairs
            if "top_pairs" not in item or not item["top_pairs"]:
                skipped_incomplete += 1
                continue
            
            # Skip if no caption (target output)
            if "caption" not in item or not item["caption"]:
                skipped_incomplete += 1
                continue
            
            prob_fake = item.get("prob_fake", 0.5)
            prediction = "FAKE" if prob_fake > 0.5 else "REAL"
            
            # Get top feature pairs (with features, coherency, and values)
            raw_top_pairs = item["top_pairs"][:5]  # Use top 5 pairs to match inference format
            
            # Validate and ensure all pairs have features, coherency, and values
            top_pairs = []
            for pair in raw_top_pairs:
                # Ensure all required fields are present
                if "features" not in pair or len(pair.get("features", [])) != 2:
                    continue
                if "coherency" not in pair:
                    continue
                if "values" not in pair or len(pair.get("values", [])) != 2:
                    continue
                
                # Ensure coherency and values are numeric
                try:
                    coherency = float(pair["coherency"])
                    values = [float(v) for v in pair["values"]]
                except (ValueError, TypeError):
                    continue
                
                # Store validated pair with all three components
                top_pairs.append({
                    "features": pair["features"],
                    "coherency": coherency,
                    "values": values
                })
            
            # Skip if no valid pairs after validation
            if not top_pairs:
                skipped_incomplete += 1
                continue
            
            # Format interactions as JSON string (matching inference format)
            # This JSON includes ALL THREE required components for each pair:
            # 1. features: [feature_name_1, feature_name_2] - the feature pair
            # 2. coherency: float - the interaction strength/coherency value
            # 3. values: [value_1, value_2] - the actual feature values (z-scores)
            # The caption (target output) is directly tied to these values and describes
            # the relationships between coherency, feature pairs, and their values
            interactions_json = json.dumps({"top_pairs": top_pairs}, indent=2, ensure_ascii=False)
            
            # Extract individual features from pairs for top values
            feature_scores = {}
            for pair in top_pairs:
                for feat in pair['features']:
                    if feat not in feature_scores:
                        # Get actual value from pair
                        if 'values' in pair and len(pair['values']) == 2:
                            feat_idx = pair['features'].index(feat)
                            feature_scores[feat] = pair['values'][feat_idx]
                        else:
                            # Fallback to coherency if values not available
                            feature_scores[feat] = pair.get('coherency', 0.0)
            
            # Format top features to match inference format (top 3, with value)
            # top_features_items = sorted(feature_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            # top_features_str = "\n".join([
            #     f"  {feat} = {val:.2f}"
            #     for feat, val in top_features_items
            # ])
            
            # Use caption as the target output (ground truth)
            # The caption describes the relationships between:
            # - Feature pairs (which features interact)
            # - Coherency values (strength of interaction: strong/moderate/weak)
            # - Feature values (z-scores: above/below mean, magnitudes, signs)
            # The model will learn to generate captions that directly reference these values
            caption = item["caption"].strip()
            
            self.data.append({
                "prob_fake": prob_fake,
                "prediction": prediction,
                "interactions_json": interactions_json,
                # "top_features": top_features_str,
                "caption": caption
            })
        
        # Limit to max_samples if specified
        if max_samples is not None and len(self.data) > max_samples:
            self.data = self.data[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if skipped_incomplete > 0:
            print(f"Skipped {skipped_incomplete} samples with incomplete data (missing pairs/coherency/values/caption)")
        print(f"Loaded {len(self.data)} samples with feature interactions")
        if len(self.data) > 0:
            # Verify first sample has all required fields
            sample = self.data[0]
            sample_json = json.loads(sample['interactions_json'])
            first_pair = sample_json['top_pairs'][0] if sample_json['top_pairs'] else {}
            print(f"Sample verification - First pair contains:")
            print(f"  - features: {'✓' if 'features' in first_pair else '✗'}")
            print(f"  - coherency: {'✓' if 'coherency' in first_pair else '✗'}")
            print(f"  - values: {'✓' if 'values' in first_pair else '✗'}")
            print(f"  - caption: {'✓' if sample.get('caption') else '✗'}")
        print()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = get_inference_prompt(
            item['prob_fake'],
            item['interactions_json'],
            # item['top_features'],
        )
        # Use prompt functions from prompts.py to match inference format exactly
        # Pass empty string for location_str (patch information not used)
        # if item['prob_fake'] > 0.5:
        #     # FAKE format
        #     prompt = get_fake_prompt(
        #         item['prob_fake'],
        #         item['interactions_json'],
        #         item['top_features'],
        #         ''  # No patch location information
        #     )
        # else:
        #     # REAL format
        #     prob_real = 1 - item['prob_fake']
        #     prompt = get_fake_prompt(
        #         prob_real,
        #         item['interactions_json'],
        #         item['top_features'],
        #         ''  # No patch location information
        #     )
        
        # Append the caption (ground truth target output)
        # The prompt contains the interactions_json with pairs, coherency, and values
        # The caption is the expected output that interprets these specific values
        # This creates a direct mapping: (pairs + coherency + values) -> caption
        text = prompt + "\n" + item['caption']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


def main():
    print("="*70)
    print("Fine-Tuning Qwen 2.5 1.5B Instruct for Feature Interpretation")
    print("="*70)
    
    # Load model with quantization
    print("\nLoading Qwen 2.5 1.5B Instruct with 4-bit quantization...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
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
    
    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration for Qwen architecture
    # Qwen uses q_proj, k_proj, v_proj, o_proj for attention and gate_proj, up_proj, down_proj for MLP
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("\nModel loaded!\n")
    
    # Load datasets from merged metadata files (limited to 450 each)
    fake_dataset = FeatureInteractionDataset(
        "merged_metadata_fake.json",
        tokenizer,
        max_length=768,
        max_samples=400
    )
    
    # Load real examples
    try:
        real_dataset = FeatureInteractionDataset(
            "merged_metadata_real.json",
            tokenizer,
            max_length=768,
            max_samples=400
        )
        # Combine datasets
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([fake_dataset, real_dataset])
        print(f"Combined dataset: {len(combined_dataset)} samples (fake + real)\n")
        train_dataset = combined_dataset
    except Exception as e:
        print(f"Note: Could not load real examples: {e}")
        print("Using only fake examples\n")
        train_dataset = fake_dataset
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="trained_qwen2.5_1.5b_feature_interpreter",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=50,
        report_to="none",
        gradient_checkpointing=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("="*70)
    print("Training Configuration:")
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print("="*70 + "\n")
    
    print("Starting training...")
    print("This will take approximately 30-60 minutes on your 6GB GPU\n")
    
    trainer.train()
    
    # Save
    print("\nSaving fine-tuned model...")
    model.save_pretrained("trained_qwen2.5_1.5b_feature_interpreter/model")
    tokenizer.save_pretrained("trained_qwen2.5_1.5b_feature_interpreter/tokenizer")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model saved to: trained_qwen2.5_1.5b_feature_interpreter/")
    print("\nThis demonstrates:")
    print("  - Fine-tuning (training algorithm design)")
    print("  - Domain-specific dataset (feature interactions)")
    print("  - Understanding LLMs (feature interpretation task)")
    print("\nNext: Use explain_features_with_interactions.py with --finetuned flag")
    print("="*70)


if __name__ == "__main__":
    main()
