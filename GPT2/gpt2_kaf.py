import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torch
import datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments
from datasets import load_dataset
from typing import Optional
import os
import tempfile
import shutil

"""
For Training GPT2_KAF from scratch, replace the activation function in the GPT2MLP class(in the file 
lib\python3.9\site-packages\transformers\models\gpt2\modeling_gpt2.py) with RFFActivation().
"""


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )


class GPT2Trainer:
    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
    def initialize_model(self) -> None:
        """Initialize model and tokenizer"""
        print("🚀 Initializing GPT-2 (training from scratch)...")
        self.config = GPT2Config()
        self.model = GPT2LMHeadModel(self.config).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)  # Using Fast Tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
    def load_dataset(self, cache_dir: str) -> None:
        """Load preprocessed dataset"""
        print("📥 Loading preprocessed openwebtext dataset...")
        try:
            dataset = load_dataset("NeelNanda/openwebtext-tokenized-9b", cache_dir=cache_dir)
            print(f"Dataset cache location: {dataset['train'].cache_files}")  # Add this line to check
            # Convert tokens to input_ids
            def convert_to_features(examples):
                return {
                    "input_ids": examples["tokens"],
                    "attention_mask": [1] * len(examples["tokens"]),
                    "labels": examples["tokens"]
                }
            
            # Apply conversion
            dataset = dataset.map(
                convert_to_features,
                remove_columns=dataset["train"].column_names,
                desc="Converting data format"
            )
            
            # Split into train and validation sets
            print("📊 Splitting into train and validation sets...")
            split_dataset = dataset["train"].train_test_split(
                test_size=0.05,
                seed=42
            )
            self.tokenized_datasets = datasets.DatasetDict({
                'train': split_dataset['train'],
                'validation': split_dataset['test']
            })
            
            print(f"📊 Dataset contains following splits: {self.tokenized_datasets.keys()}")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            raise
        
    def setup_training(self) -> None:
        """Setup training parameters"""
        print("⚙️ Configuring training parameters...")
        self.training_args = TrainingArguments(
            output_dir="./gpt2_from_scratch",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,
            weight_decay=0.01,
            num_train_epochs=5,
            warmup_ratio=0.1,
            fp16=True,
            logging_dir="./logs",
            logging_steps=500,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
        )
        
    def train(self) -> None:
        """Train the model"""
        print("🚀 Starting training...")
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets.get("validation", None),  # Use if available, else None
        )
        trainer.train()
        
    @staticmethod
    def evaluate_ppl(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, text: str, device: str) -> float:
        """Calculate perplexity"""
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
    
    def compare_models(self, text_sample: str) -> None:
        """Compare model performance"""
        print("📊 Calculating PPL (perplexity) comparison...")
        model_original = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        
        ppl_original = self.evaluate_ppl(model_original, self.tokenizer, text_sample, self.device)
        ppl_new = self.evaluate_ppl(self.model, self.tokenizer, text_sample, self.device)
        
        print(f"✅ Pretrained GPT-2 PPL: {ppl_original:.2f}")
        print(f"✅ From-scratch GPT-2 PPL: {ppl_new:.2f}")

def main():
    trainer = GPT2Trainer()
    
    # Initialize model and tokenizer
    trainer.initialize_model()
    
    # Load dataset
    trainer.load_dataset(cache_dir="data/cache")
    
    # Setup training parameters
    trainer.setup_training()
    
    # Train model
    trainer.train()
    
    trainer.compare_models("The quick brown fox jumps over the lazy dog.")
    
    print("🎉 Training complete!")

if __name__ == "__main__":
    main()
