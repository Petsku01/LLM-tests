#!/usr/bin/env python3
"""Fine-tune Llama-4 models with QLoRA and Unsloth."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


def get_rank() -> int:
    """Get the rank of the current process in distributed training."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def print_rank0(message: str = ""):
    """Print only from main process."""
    if is_main_process():
        print(message)


DEFAULT_CONFIG = {
    "model_name": "unsloth/Llama-4-8B-Instruct",
    "dataset_path": "data/sample_sharegpt.json",
    "output_dir": "./outputs/llama4_finetuned",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "use_rslora": False,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "logging_steps": 10,
    "save_steps": 100,
    "packing": False,
}


def validate_environment():
    """Check CUDA availability and GPU setup."""
    if not torch.cuda.is_available():
        print_rank0(" ERROR: CUDA not available. GPU is required for training.")
        print_rank0("Please ensure:")
        print_rank0("  1. You have an NVIDIA GPU")
        print_rank0("  2. CUDA drivers are installed")
        print_rank0("  3. PyTorch is installed with CUDA support")
        sys.exit(1)
    
    print_rank0(f" GPU detected: {torch.cuda.get_device_name(0)}")
    
    num_gpus = torch.cuda.device_count()
    print_rank0(f" Number of GPUs available: {num_gpus}")
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_vram < 16:
        print_rank0(f" WARNING: GPU has {total_vram:.1f}GB VRAM. 16GB+ recommended for 8B models.")
    
    print_rank0(f" PyTorch version: {torch.__version__}")
    print_rank0(f" CUDA version: {torch.version.cuda}")
    print()


def validate_dataset(dataset_path: str) -> bool:
    """Validate that dataset exists and has correct format."""
    if not os.path.exists(dataset_path):
        print_rank0(f" ERROR: Dataset not found: {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print_rank0(" ERROR: Dataset must be a JSON array")
            return False
        
        if len(data) == 0:
            print_rank0(" ERROR: Dataset is empty")
            return False
        
        first_entry = data[0]
        if "conversations" not in first_entry:
            print_rank0(" ERROR: Dataset must use ShareGPT format with 'conversations' field")
            return False
        
        print_rank0(f" Dataset validated: {len(data)} examples")
        return True
        
    except json.JSONDecodeError as e:
        print_rank0(f" ERROR: Invalid JSON: {e}")
        return False
    except Exception as e:
        print_rank0(f" ERROR: Failed to load dataset: {e}")
        return False


def estimate_vram_usage(config: Dict):
    """Estimate VRAM requirements based on configuration."""
    base_model_size = 8
    
    if config["load_in_4bit"]:
        model_vram = base_model_size * 0.5
    else:
        model_vram = base_model_size * 2
    
    lora_params = (config["lora_r"] * config["max_seq_length"] * 4 * 32) / 1024**3
    
    batch_vram = (
        config["per_device_train_batch_size"]
        * config["max_seq_length"]
        * 4
        * 4
        / 1024**3
    )
    
    optimizer_vram = model_vram * 0.5
    total_vram = model_vram + lora_params + batch_vram + optimizer_vram
    
    print_rank0(" Estimated VRAM usage:")
    print_rank0(f"  - Model: {model_vram:.1f} GB")
    print_rank0(f"  - LoRA adapters: {lora_params:.1f} GB")
    print_rank0(f"  - Batch activations: {batch_vram:.1f} GB")
    print_rank0(f"  - Optimizer states: {optimizer_vram:.1f} GB")
    print_rank0(f"  - Total estimated: {total_vram:.1f} GB")
    print()


def measure_actual_vram(model):
    """Measure actual VRAM usage after model loading."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print_rank0(f" Actual VRAM usage:")
        print_rank0(f"  - Allocated: {allocated:.2f} GB")
        print_rank0(f"  - Reserved: {reserved:.2f} GB")
        print()


def print_training_config(config: dict):
    """Display training configuration."""
    print()
    print(f"Model: {config['model_name']}")
    print(f"Dataset: {config['dataset_path']}")
    print(f"Output directory: {config['output_dir']}")
    print()
    print("Model Configuration:")
    print(f"  - Max sequence length: {config['max_seq_length']}")
    print(f"  - Quantization: {'4-bit (NF4)' if config['load_in_4bit'] else 'None'}")
    print()
    print("LoRA Configuration:")
    print(f"  - Rank (r): {config['lora_r']}")
    print(f"  - Alpha: {config['lora_alpha']}")
    print(f"  - Dropout: {config['lora_dropout']}")
    print(f"  - RS-LoRA: {config['use_rslora']}")
    print()
    print("Training Configuration:")
    print(f"  - Batch size per device: {config['per_device_train_batch_size']}")
    print(f"  - Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    print(f"  - Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  - Epochs: {config['num_train_epochs']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - LR scheduler: {config['lr_scheduler_type']}")
    print(f"  - Weight decay: {config['weight_decay']}")
    print(f"  - Warmup ratio: {config['warmup_ratio']}")
    print(f"  - Sample packing: {config['packing']}")
    print(f"  - Gradient checkpointing: {config['use_gradient_checkpointing']}")
    print(f"  - Random seed: {config['seed']}")
    print()


def load_and_prepare_model(config: Dict):
    """Load model with 4-bit quantization and add LoRA adapters."""
    print(" Loading model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
        device_map="auto",
    )
    
    print(f" Model loaded: {config['model_name']}")
    print(f" Model dtype: {model.dtype}")
    print()
    
    measure_actual_vram(model)
    
    print(" Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
        random_state=config["seed"],
        use_rslora=config["use_rslora"],
        loftq_config=None,
    )
    
    print(" LoRA adapters added successfully")
    print()
    
    return model, tokenizer


def load_and_prepare_dataset(dataset_path: str, tokenizer):
    """Load and prepare training dataset."""
    print(f" Loading dataset from {dataset_path}...")
    
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        if len(dataset) == 0:
            print_rank0(" ERROR: Dataset is empty after loading")
            return None
        
        print(f" Dataset loaded: {len(dataset)} examples")
        print(f" Dataset columns: {dataset.column_names}")
        
        if "conversations" not in dataset.column_names:
            print_rank0(" ERROR: Dataset missing 'conversations' field (ShareGPT format required)")
            return None
        
        first_conversation = dataset[0]["conversations"]
        if not isinstance(first_conversation, list) or len(first_conversation) == 0:
            print_rank0(" ERROR: 'conversations' field must be a non-empty list")
            return None
        
        first_message = first_conversation[0]
        if "from" not in first_message or "value" not in first_message:
            print_rank0(" ERROR: Each conversation turn must have 'from' and 'value' fields")
            return None
        
        print(" Dataset format validated (ShareGPT format)")
        print()
        return dataset
        
    except Exception as e:
        print_rank0(f" ERROR loading dataset: {e}")
        import traceback
        print_rank0(traceback.format_exc())
        return None


def create_trainer(model, tokenizer, dataset, config: Dict):
    """Create SFTTrainer with checkpoint detection."""
    output_dir = Path(config["output_dir"])
    checkpoint_dir = None
    
    if output_dir.exists():
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f" Found checkpoint: {latest_checkpoint}")
            
            response = input(" Resume from this checkpoint? (y/n): ").strip().lower()
            if response == 'y':
                checkpoint_dir = str(latest_checkpoint)
                print(f" Will resume from {checkpoint_dir}")
            else:
                print(" Starting fresh training")
    
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=config["seed"],
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="conversations",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=2,
        packing=config["packing"],
        args=training_args,
    )
    
    return trainer, checkpoint_dir


def train_model(trainer: SFTTrainer, checkpoint_dir=None):
    """Execute training loop."""
    print_rank0(" Starting training...")
    print_rank0()
    
    try:
        if checkpoint_dir:
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            trainer.train()
        print_rank0()
        print_rank0(" Training completed successfully")
        print_rank0()
    except Exception as e:
        print_rank0(f" ERROR during training: {e}")
        import traceback
        print_rank0(traceback.format_exc())
        sys.exit(1)


def save_and_merge_model(model, tokenizer, config: Dict):
    """Save LoRA adapters and create merged models."""
    output_dir = Path(config["output_dir"])
    
    lora_dir = output_dir / "lora_adapters"
    print_rank0(f" Saving LoRA adapters to {lora_dir}...")
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print_rank0(" LoRA adapters saved")
    
    merged_16bit_dir = output_dir / "merged_16bit"
    print_rank0(f" Merging and saving 16-bit model to {merged_16bit_dir}...")
    model.save_pretrained_merged(
        str(merged_16bit_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    print_rank0(" 16-bit merged model saved")
    
    merged_4bit_dir = output_dir / "merged_4bit"
    print_rank0(f" Merging and saving 4-bit model to {merged_4bit_dir}...")
    model.save_pretrained_merged(
        str(merged_4bit_dir),
        tokenizer,
        save_method="merged_4bit",
    )
    print_rank0(" 4-bit merged model saved (recommended for inference)")
    print_rank0()


def parse_arguments() -> Dict:
    """Parse command-line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-4-8B-Instruct with QLoRA and Unsloth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_CONFIG["model_name"],
        help="Hugging Face model name or local path"
    )
    model_group.add_argument(
        "--max_seq_length",
        type=int,
        default=DEFAULT_CONFIG["max_seq_length"],
        help="Maximum sequence length (context window)"
    )
    model_group.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=DEFAULT_CONFIG["load_in_4bit"],
        help="Load model in 4-bit quantization"
    )
    
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=DEFAULT_CONFIG["lora_r"],
        help="LoRA rank (higher = more capacity, more VRAM)"
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=DEFAULT_CONFIG["lora_alpha"],
        help="LoRA alpha (scaling factor)"
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=DEFAULT_CONFIG["lora_dropout"],
        help="LoRA dropout probability"
    )
    lora_group.add_argument(
        "--use_rslora",
        action="store_true",
        default=DEFAULT_CONFIG["use_rslora"],
        help="Use RS-LoRA (rank-stabilized LoRA)"
    )
    
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_CONFIG["dataset_path"],
        help="Path to training dataset (ShareGPT JSON format)"
    )
    train_group.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help="Output directory for checkpoints and final model"
    )
    train_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=DEFAULT_CONFIG["per_device_train_batch_size"],
        help="Batch size per GPU"
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
        help="Gradient accumulation steps (effective_batch_size = batch_size * this)"
    )
    train_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=DEFAULT_CONFIG["num_train_epochs"],
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Peak learning rate"
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_CONFIG["weight_decay"],
        help="Weight decay for regularization"
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=DEFAULT_CONFIG["warmup_ratio"],
        help="Ratio of steps for learning rate warmup"
    )
    train_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=DEFAULT_CONFIG["lr_scheduler_type"],
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type"
    )
    train_group.add_argument(
        "--packing",
        action="store_true",
        default=DEFAULT_CONFIG["packing"],
        help="Enable sample packing for efficiency"
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed for reproducibility"
    )
    
    log_group = parser.add_argument_group("Logging & Checkpointing")
    log_group.add_argument(
        "--logging_steps",
        type=int,
        default=DEFAULT_CONFIG["logging_steps"],
        help="Log training metrics every N steps"
    )
    log_group.add_argument(
        "--save_steps",
        type=int,
        default=DEFAULT_CONFIG["save_steps"],
        help="Save checkpoint every N steps"
    )
    
    args = parser.parse_args()
    
    config = {
        "model_name": args.model_name,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "use_rslora": args.use_rslora,
        "use_gradient_checkpointing": "unsloth",
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "packing": args.packing,
    }
    
    return config


def main():
    """Main execution pipeline."""
    print_rank0("\nLLAMA-4-8B-INSTRUCT FINE-TUNING WITH UNSLOTH + QLORA\n")
    
    config = parse_arguments()
    validate_environment()
    
    if not validate_dataset(config["dataset_path"]):
        sys.exit(1)
    
    print_training_config(config)
    estimate_vram_usage(config)
    
    model, tokenizer = load_and_prepare_model(config)
    dataset = load_and_prepare_dataset(config["dataset_path"], tokenizer)
    
    if dataset is None:
        print_rank0(" FATAL: Dataset could not be loaded. Check errors above.")
        sys.exit(1)
    
    trainer, checkpoint_dir = create_trainer(model, tokenizer, dataset, config)
    train_model(trainer, checkpoint_dir)
    save_and_merge_model(model, tokenizer, config)
    
    print_rank0("\n FINE-TUNING COMPLETED SUCCESSFULLY")
    print_rank0("\nNext steps:")
    print_rank0("1. Test the model with inference scripts")
    print_rank0("2. Evaluate on validation/test sets")
    print_rank0("3. Deploy using the merged_4bit model for efficiency")
    print_rank0()


if __name__ == "__main__":
    main()
