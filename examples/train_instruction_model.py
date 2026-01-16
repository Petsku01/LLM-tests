#!/usr/bin/env python3
"""Example: Fine-tune Llama-4 for instruction following."""

import subprocess
import sys


def main():
    """Run instruction-following training."""
    
    print("TRAINING INSTRUCTION-FOLLOWING MODEL")
    print()
    print("This example fine-tunes Llama-4-8B for instruction-following tasks.")
    print()
    
    # Training command
    cmd = [
        sys.executable,
        "finetune_llama4_company.py",
        
        # Dataset
        "--dataset_path", "data/sample_sharegpt.json",
        "--output_dir", "outputs/llama4-instruct-v1",
        
        # Model config
        "--max_seq_length", "4096",
        "--lora_r", "64",
        "--lora_alpha", "32",
        "--lora_dropout", "0.1",
        
        # Training config (conservative for quality)
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "1e-4",
        "--weight_decay", "0.01",
        "--warmup_ratio", "0.05",
        "--lr_scheduler_type", "cosine",
        
        # Logging
        "--logging_steps", "5",
        "--save_steps", "50",
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Execute training
    try:
        subprocess.run(cmd, check=True)
        print("\n Training completed successfully!")
        print("\nTest the model:")
        print("  python inference.py --model_path outputs/llama4-instruct-v1/merged_4bit")
    except subprocess.CalledProcessError as e:
        print(f"\n Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
