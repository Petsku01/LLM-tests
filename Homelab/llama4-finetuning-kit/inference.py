#!/usr/bin/env python3
"""Run inference with fine-tuned Llama-4 models."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel


# Configuration

DEFAULT_CONFIG = {
    "max_seq_length": 8192,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "max_new_tokens": 512,
}


# Model loading

def load_model(model_path: str, max_seq_length: int = 8192, load_in_4bit: bool = True):
    """Load fine-tuned model for inference."""
    print(f" Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        print(f" ERROR: Model not found at {model_path}")
        sys.exit(1)
    
    try:
        # Load model with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit,
        )
        

        print(f" Model dtype: {model.dtype}")
        print(f" Device: {next(model.parameters()).device}")
        print()
        
        return model, tokenizer
    
    except Exception as e:
        print(f" ERROR: Failed to load model: {e}")
        sys.exit(1)


# Inference functions

def format_prompt(messages: List[Dict[str, str]], tokenizer) -> str:
    """Format messages using chat template."""
    # Convert to Llama-4 chat format
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stream: bool = False
) -> str:
    """
    Generate a response from the model.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling
        repetition_penalty: Penalty for repeating tokens
        stream: Whether to stream output
    
    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Setup streamer if needed
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
    
    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
    
    # Decode output
    if not stream:
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()
    else:
        return ""  # Text already streamed to console


def chat_loop(model, tokenizer, config: Dict[str, Any]):
    """
    Interactive chat loop.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        config: Generation configuration
    """
    print("\nINTERACTIVE CHAT MODE")
    print("\nCommands:")
    print("  - Type your message and press Enter")
    print("  - 'reset' to clear conversation history")
    print("  - 'config' to show current settings")
    print("  - 'exit' or 'quit' to exit")
    print()
    
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'reset':
            conversation_history = []
            print(" Conversation history cleared\n")
            continue
        
        if user_input.lower() == 'config':
            print(json.dumps(config, indent=2))
            print()
            continue
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Format prompt
        prompt = format_prompt(conversation_history, tokenizer)
        
        # Validate input length
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tokens = len(input_ids)
        max_seq_length = config.get("max_seq_length", 2048)
        max_new_tokens = config["max_new_tokens"]
        
        if prompt_tokens + max_new_tokens > max_seq_length:
            print(f"\n WARNING: Input is too long!")
            print(f"   - Prompt tokens: {prompt_tokens}")
            print(f"   - Max new tokens: {max_new_tokens}")
            print(f"   - Total needed: {prompt_tokens + max_new_tokens}")
            print(f"   - Max allowed: {max_seq_length}")
            print(f"   - Please shorten your input or clear history with 'clear'\n")
            # Remove the last user message
            conversation_history.pop()
            continue
        
        # Generate response
        print("\nAssistant: ", end="", flush=True)
        start_time = time.time()
        
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            repetition_penalty=config["repetition_penalty"],
            stream=True  # Stream in chat mode
        )
        
        # Calculate generation time
        end_time = time.time()
        generation_time = end_time - start_time
        
        # For streaming, we need to get the response from conversation
        # In practice, we'd capture it from the streamer
        # For now, regenerate without streaming to get the text
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            repetition_penalty=config["repetition_penalty"],
            stream=False
        )
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response})
        
        # Show stats
        tokens_generated = len(tokenizer.encode(response))
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        print(f"\n\n[Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)]")
        print()


def batch_inference(
    model,
    tokenizer,
    input_file: str,
    output_file: str,
    config: Dict[str, Any]
):
    """
    Run batch inference on a file of prompts.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        input_file: Path to input JSON file with prompts
        output_file: Path to output JSON file for results
        config: Generation configuration
    """
    print(f" Loading prompts from: {input_file}")
    
    # Load input data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f" ERROR: Failed to load input file: {e}")
        sys.exit(1)
    
    if not isinstance(data, list):
        print(" ERROR: Input file must contain a list of prompts")
        sys.exit(1)
    
    print(f" Loaded {len(data)} prompts")
    print(" Starting batch inference...\n")
    
    results = []
    total_time = 0
    total_tokens = 0
    
    for i, item in enumerate(data, 1):
        print(f"Processing {i}/{len(data)}...")
        
        # Format prompt
        if isinstance(item, str):
            messages = [{"role": "user", "content": item}]
        elif isinstance(item, dict) and "messages" in item:
            messages = item["messages"]
        elif isinstance(item, dict) and "prompt" in item:
            messages = [{"role": "user", "content": item["prompt"]}]
        else:
            print(f"  Skipping invalid item: {item}")
            continue
        
        prompt = format_prompt(messages, tokenizer)
        
        # Generate
        start_time = time.time()
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            repetition_penalty=config["repetition_penalty"],
            stream=False
        )
        end_time = time.time()
        
        generation_time = end_time - start_time
        tokens_generated = len(tokenizer.encode(response))
        
        total_time += generation_time
        total_tokens += tokens_generated
        
        # Store result
        results.append({
            "input": messages,
            "output": response,
            "generation_time": generation_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / generation_time if generation_time > 0 else 0
        })
        
        print(f"   Generated {tokens_generated} tokens in {generation_time:.2f}s\n")
    
    # Save results
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f" Results saved to: {output_file}")
    except Exception as e:
        print(f" ERROR: Failed to save results: {e}")
        sys.exit(1)
    
    # Show summary
    avg_time = total_time / len(results) if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    
    print("\nBATCH INFERENCE SUMMARY")
    print(f"Total prompts processed: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Average time per prompt: {avg_time:.2f}s")
    print(f"Average tokens per prompt: {avg_tokens:.1f}")
    print(f"Average speed: {avg_speed:.1f} tokens/second")


# CLI

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned Llama-4 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["chat", "single", "batch"],
        default="chat",
        help="Inference mode"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt for inference (for 'single' mode)"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file with prompts (for 'batch' mode)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for results (for 'batch' mode)"
    )
    
    # Generation parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_CONFIG["max_new_tokens"],
        help="Maximum tokens to generate"
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CONFIG["temperature"],
        help="Sampling temperature (0.0 = greedy, higher = more random)"
    )
    gen_group.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_CONFIG["top_p"],
        help="Nucleus sampling threshold"
    )
    gen_group.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_CONFIG["top_k"],
        help="Top-k sampling"
    )
    gen_group.add_argument(
        "--repetition_penalty",
        type=float,
        default=DEFAULT_CONFIG["repetition_penalty"],
        help="Repetition penalty (1.0 = no penalty)"
    )
    
    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument(
        "--max_seq_length",
        type=int,
        default=DEFAULT_CONFIG["max_seq_length"],
        help="Maximum sequence length"
    )
    model_group.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    print("\nLLAMA-4 INFERENCE ENGINE\n")
    
    args = parse_arguments()
    
    # Validate CUDA
    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available, using CPU (will be slow)")
    else:
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    
    print()
    
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        args.max_seq_length,
        args.load_in_4bit
    )
    
    # Prepare config
    config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    
    # Run inference based on mode
    if args.mode == "chat":
        chat_loop(model, tokenizer, config)
    
    elif args.mode == "single":
        if not args.prompt:
            print(" ERROR: --prompt required for single mode")
            sys.exit(1)
        
        messages = [{"role": "user", "content": args.prompt}]
        prompt = format_prompt(messages, tokenizer)
        
        print("Generating response...\n")
        response = generate_response(
            model, tokenizer, prompt,
            **config,
            stream=False
        )
        
        print("Response:")
        print("-" * 80)
        print(response)
        print("-" * 80)
    
    elif args.mode == "batch":
        if not args.input_file or not args.output_file:
            print(" ERROR: --input_file and --output_file required for batch mode")
            sys.exit(1)
        
        batch_inference(model, tokenizer, args.input_file, args.output_file, config)


if __name__ == "__main__":
    main()
