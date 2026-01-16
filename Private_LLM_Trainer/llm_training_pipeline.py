#!/usr/bin/env python3
"""
Private LLM Trainer - Ultimate Production Version
Zero errors, zero bloat, maximum performance
"""

import os
import gc
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Privacy settings before any imports
os.environ.update({
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'WANDB_DISABLED': 'true',
    'TOKENIZERS_PARALLELISM': 'false'
})

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2TokenizerFast,
    get_cosine_schedule_with_warmup
)


@dataclass
class Config:
    # Model
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    max_length: int = 1024
    dropout: float = 0.1
    
    # Training  
    batch_size: int = 4
    accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Performance
    mixed_precision: bool = torch.cuda.is_available()
    gradient_checkpointing: bool = True
    compile: bool = hasattr(torch, 'compile')
    
    # Data
    train_split: float = 0.9
    min_text_length: int = 50
    
    # Paths
    output_dir: str = "./model"
    
    # Monitoring
    log_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500


class DataHandler:
    """Simple, efficient data processing"""
    
    @staticmethod
    def load_texts(sources: Dict) -> List[str]:
        """Load texts from various sources"""
        texts = []
        
        # Process directory
        if 'directory' in sources and sources['directory']:
            path = Path(sources['directory'])
            if path.exists():
                # Process text files
                for file in path.rglob('*.txt'):
                    try:
                        content = file.read_text(encoding='utf-8', errors='ignore')
                        if content:
                            texts.append(content)
                    except Exception:
                        pass
                
                # Process JSON files
                for file in path.rglob('*.json'):
                    try:
                        with open(file, encoding='utf-8', errors='ignore') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                texts.extend([str(item) for item in data if item])
                            elif isinstance(data, dict):
                                text = data.get('text', str(data))
                                if text:
                                    texts.append(str(text))
                            else:
                                texts.append(str(data))
                    except Exception:
                        pass
        
        # Add raw texts
        if 'texts' in sources and sources['texts']:
            if isinstance(sources['texts'], list):
                texts.extend([str(t) for t in sources['texts'] if t])
            else:
                texts.append(str(sources['texts']))
        
        # Filter valid texts
        return [t.strip() for t in texts if t and len(t.strip()) >= 50]


class TextDataset(Dataset):
    """Efficient text dataset"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.data = []
        
        for text in texts:
            tokens = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            self.data.append({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'labels': tokens['input_ids'].squeeze()
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class LLMTrainer:
    """Core training system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Setup paths
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_model(self):
        """Create optimized model"""
        config = GPT2Config(
            vocab_size=self.config.vocab_size,
            n_embd=self.config.hidden_size,
            n_layer=self.config.num_layers,
            n_head=self.config.num_heads,
            n_positions=self.config.max_length,
            resid_pdrop=self.config.dropout,
            embd_pdrop=self.config.dropout,
            attn_pdrop=self.config.dropout,
            use_cache=False
        )
        
        model = GPT2LMHeadModel(config)
        
        # Enable optimizations
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        if self.config.compile and hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        return model.to(self.device)
    
    def prepare_data(self, texts: List[str]) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare training and validation data"""
        
        # Shuffle and split
        random.shuffle(texts)
        split_idx = int(len(texts) * self.config.train_split)
        split_idx = max(1, split_idx)  # Ensure at least 1 training sample
        
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:] if split_idx < len(texts) else None
        
        # Create datasets
        train_dataset = TextDataset(train_texts, self.tokenizer, self.config.max_length)
        val_dataset = TextDataset(val_texts, self.tokenizer, self.config.max_length) if val_texts else None
        
        # Adjust batch size if needed
        train_batch_size = min(self.config.batch_size, len(train_dataset))
        val_batch_size = min(self.config.batch_size, len(val_dataset)) if val_dataset else 1
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            num_workers=0,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0
        ) if val_dataset else None
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Execute training"""
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.epochs // self.config.accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.accumulation_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Step
                    if self.scaler:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % self.config.log_steps == 0:
                        lr = scheduler.get_last_lr()[0]
                        avg_loss = epoch_loss / (batch_idx + 1) * self.config.accumulation_steps
                        print(f"[Epoch {epoch+1}/{self.config.epochs}] "
                              f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    
                    # Evaluation
                    if val_loader and global_step % self.config.eval_steps == 0:
                        val_loss = self._evaluate(val_loader)
                        print(f"Validation Loss: {val_loss:.4f}")
                        
                        if val_loss < best_loss:
                            best_loss = val_loss
                            self._save_model("best")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self._save_model(f"checkpoint_{global_step}")
                
                epoch_loss += loss.item() * self.config.accumulation_steps
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        self._save_model("final")
        print(f"Training complete! Model saved to {self.config.output_dir}")
    
    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        
        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
        
        return total_loss / len(val_loader)
    
    def _save_model(self, name: str):
        """Save model checkpoint"""
        save_path = Path(self.config.output_dir) / name
        save_path.mkdir(exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Saved checkpoint: {save_path}")
    
    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        """Generate text"""
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        if self.device.type == 'cuda' and self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def run_pipeline(self, data_sources: Dict):
        """Complete training pipeline"""
        print("="*50)
        print("Private LLM Training Pipeline")
        print("="*50)
        
        # Load data
        texts = DataHandler.load_texts(data_sources)
        if not texts:
            raise ValueError("No valid texts found")
        
        print(f"Loaded {len(texts)} documents")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(texts)
        print(f"Training batches: {len(train_loader)}")
        if val_loader:
            print(f"Validation batches: {len(val_loader)}")
        
        # Train
        self.train(train_loader, val_loader)
        
        # Cleanup
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    """Main execution"""
    
    # Configuration
    config = Config(
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        batch_size=2,
        epochs=2,
        learning_rate=3e-4,
        mixed_precision=True,
        gradient_checkpointing=True
    )
    
    # Initialize trainer
    trainer = LLMTrainer(config)
    
    # Data sources
    data_sources = {
        'directory': './training_data',  # Optional
        'texts': [
            "This is a completely private language model trainer.",
            "All processing happens locally on your machine.",
            "No data is sent to external servers.",
            "The model trains efficiently with modern optimizations.",
            "You have full control over your AI model."
        ]
    }
    
    # Train
    trainer.run_pipeline(data_sources)
    
    # Test generation
    prompts = [
        "The future of AI is",
        "Private machine learning means",
        "Technology should be"
    ]
    
    print("\n" + "="*50)
    print("Testing generated outputs:")
    print("="*50)
    
    for prompt in prompts:
        output = trainer.generate(prompt, max_tokens=50)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")


if __name__ == "__main__":
    main()