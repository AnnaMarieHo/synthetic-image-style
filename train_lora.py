"""
Fine-tune Qwen 2.5 1.5B Instruct on Feature Interaction Dataset

Qwen 2.5 1.5B Instruct for deepfake feature interpretation.

"""
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Subset, ConcatDataset
import random
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from patches_and_gradcam.prompts import get_training_prompt

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
            raw_top_pairs = item["top_pairs"][:5]  # Use top 5 pairs to match training caption data
            
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
       
            caption = item["caption"].strip()
            
            # Store prompt separately for evaluation
            prompt = get_training_prompt(
                prob_fake,
                interactions_json,
            )
            
            self.data.append({
                "prob_fake": prob_fake,
                "prediction": prediction,
                "interactions_json": interactions_json,
                "caption": caption,
                "prompt": prompt
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
            print(f"  - features: {'true' if 'features' in first_pair else 'false'}")
            print(f"  - coherency: {'true' if 'coherency' in first_pair else 'false'}")
            print(f"  - values: {'true' if 'values' in first_pair else 'false'}")
            print(f"  - caption: {'true' if sample.get('caption') else 'false'}")
        print()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use stored prompt
        prompt = item['prompt']
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
    
    def get_prompt_and_caption(self, idx):
        """Helper method to get prompt and caption for evaluation"""
        item = self.data[idx]
        return item['prompt'], item['caption']


def compute_metrics(eval_pred):
    """
    Bare minimum: Just return empty dict.
    Perplexity is computed from eval_loss by the PerplexityCallback.
    This avoids accumulating predictions in memory.
    """
    return {}


def main():
    print("Fine-Tuning Qwen 2.5 1.5B Instruct for Feature Interpretation")

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
    
    print("\nModel loaded\n")
    
    # Load datasets from merged metadata files (limited to 900 each)
    fake_dataset = FeatureInteractionDataset(
        "merged_metadata_fake.json",
        tokenizer,
        max_length=768,
        max_samples=900
    )
    
    # Load real examples
    try:
        real_dataset = FeatureInteractionDataset(
            "merged_metadata_real.json",
            tokenizer,
            max_length=768,
            max_samples=900
        )
        # Combine datasets
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([fake_dataset, real_dataset])
        print(f"Combined dataset: {len(combined_dataset)} samples (fake + real)\n")
        full_dataset = combined_dataset
    except Exception as e:
        print(f"Note: Could not load real examples: {e}")
        print("Using only fake examples\n")
        full_dataset = fake_dataset
    
    # Stratified split to ensure balanced fake/real distribution in train and eval
    
    random.seed(42)
    
    if isinstance(full_dataset, ConcatDataset) and len(full_dataset.datasets) == 2:
        # do stratified split
        fake_dataset_obj = full_dataset.datasets[0]
        real_dataset_obj = full_dataset.datasets[1]
        fake_len = len(fake_dataset_obj)
        real_len = len(real_dataset_obj)
        
        # Create indices for fake and real separately
        fake_indices = list(range(fake_len))
        real_indices = list(range(fake_len, fake_len + real_len))
        
        # Shuffle each class separately
        random.shuffle(fake_indices)
        random.shuffle(real_indices)
        
        # Split each class 90/10
        fake_train_size = int(0.9 * fake_len)
        real_train_size = int(0.9 * real_len)
        
        fake_train_indices = fake_indices[:fake_train_size]
        fake_eval_indices = fake_indices[fake_train_size:]
        real_train_indices = real_indices[:real_train_size]
        real_eval_indices = real_indices[real_train_size:]
        
        # Combine train and eval indices
        train_indices = fake_train_indices + real_train_indices
        eval_indices = fake_eval_indices + real_eval_indices
        
        # Shuffle the combined indices to mix fake/real within each set
        random.shuffle(train_indices)
        random.shuffle(eval_indices)
        
        # Create Subsets
        train_dataset = Subset(full_dataset, train_indices)
        eval_dataset = Subset(full_dataset, eval_indices)
        
        print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        print(f"  Train: {len(fake_train_indices)} fake, {len(real_train_indices)} real (balanced)")
        print(f"  Eval: {len(fake_eval_indices)} fake, {len(real_eval_indices)} real (balanced)")
    else:
        # Single dataset (only fake or only real)
        all_indices = list(range(len(full_dataset)))
        random.shuffle(all_indices)
        
        train_size = int(0.9 * len(full_dataset))
        train_indices = all_indices[:train_size]
        eval_indices = all_indices[train_size:]
        
        train_dataset = Subset(full_dataset, train_indices)
        eval_dataset = Subset(full_dataset, eval_indices)
        
        print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        print(f"  (Single class dataset - no balancing needed)")
    print()
    
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="trained_qwen2.5_1.5b_feature_interpreter",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        eval_steps=20,  # Evaluate every 20 steps (more frequent to catch issues early)
        eval_strategy="steps",  # Enable evaluation during training
        save_strategy="steps",  # Must match eval_strategy when load_best_model_at_end=True
        save_steps=20,  # Save every 20 steps (same as eval_steps)
        save_total_limit=2,
        warmup_steps=50,
        report_to="none",
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_perplexity",
        greater_is_better=False,  # Lower perplexity is better
        dataloader_pin_memory=False,  # help with memory
        prediction_loss_only=True,  # Only compute loss, don't accumulate predictions to save memory
        # Note to self: dataloader_shuffle defaults to True for training, False for eval
    )
    
    # Perplexity is computed from eval_loss by the callback
    # Custom callback to add perplexity from eval_loss
    
    class PerplexityCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            # Add perplexity to metrics if eval_loss is present
            # Using 'eval_perplexity' prefix to match Trainer's metric naming
            if metrics is not None and 'eval_loss' in metrics:
                eval_loss = metrics['eval_loss']
                perplexity = np.exp(min(eval_loss, 50))  # Clamp to avoid overflow
                metrics['eval_perplexity'] = float(perplexity)
                # Also log it explicitly so it appears in output
                if state.is_local_process_zero:
                    print(f"  eval_perplexity: {perplexity:.4f}")
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            # Ensure perplexity appears in logs if eval_loss is present
            if logs is not None and 'eval_loss' in logs and 'eval_perplexity' not in logs:
                eval_loss = logs['eval_loss']
                perplexity = np.exp(min(eval_loss, 50))
                logs['eval_perplexity'] = float(perplexity)
    
    perplexity_callback = PerplexityCallback()
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[perplexity_callback],
    )
    
    print("Training Configuration:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Evaluation strategy: {training_args.eval_strategy}")
    print(f"  Eval steps: {training_args.eval_steps}")
    print(f"  Metrics: Perplexity (PPL) only")
    print()


    trainer.train()
    
    # Save
    print("\nSaving fine-tuned model...")
    model.save_pretrained("trained_qwen2.5_1.5b_feature_interpreter/model")
    tokenizer.save_pretrained("trained_qwen2.5_1.5b_feature_interpreter/tokenizer")
    
    print("TRAINING COMPLETE")
    print(f"Model saved to: trained_qwen2.5_1.5b_feature_interpreter/")
    print("\nNext: Use explain_features_with_interactions.py with --finetuned flag")


if __name__ == "__main__":
    main()

