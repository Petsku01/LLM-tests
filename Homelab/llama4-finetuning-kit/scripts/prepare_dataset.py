#!/usr/bin/env python3
"""Convert datasets to ShareGPT format for fine-tuning."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm


# Format converters

def convert_alpaca_to_sharegpt(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Alpaca format to ShareGPT format.
    
    Alpaca format:
    {
        "instruction": "What is machine learning?",
        "input": "",  # Optional context
        "output": "Machine learning is..."
    }
    """
    conversations = []
    
    # Combine instruction and input
    user_message = sample["instruction"]
    if sample.get("input", "").strip():
        user_message += f"\n\n{sample['input']}"
    
    conversations.append({
        "from": "human",
        "value": user_message
    })
    
    conversations.append({
        "from": "gpt",
        "value": sample["output"]
    })
    
    return {"conversations": conversations}


def convert_oasst_to_sharegpt(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenAssistant format to ShareGPT format.
    
    OASST format:
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }
    """
    conversations = []
    
    role_mapping = {
        "user": "human",
        "assistant": "gpt",
        "system": "system"
    }
    
    for msg in sample["messages"]:
        role = role_mapping.get(msg["role"], msg["role"])
        conversations.append({
            "from": role,
            "value": msg["content"]
        })
    
    return {"conversations": conversations}


def convert_dolly_to_sharegpt(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Dolly format to ShareGPT format.
    
    Dolly format:
    {
        "instruction": "Summarize the following",
        "context": "Long text here...",
        "response": "Summary here..."
    }
    """
    conversations = []
    
    # Combine instruction and context
    user_message = sample["instruction"]
    if sample.get("context", "").strip():
        user_message += f"\n\nContext:\n{sample['context']}"
    
    conversations.append({
        "from": "human",
        "value": user_message
    })
    
    conversations.append({
        "from": "gpt",
        "value": sample["response"]
    })
    
    return {"conversations": conversations}


def convert_custom_to_sharegpt(sample: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Convert custom format using user-provided field mapping.
    
    Args:
        sample: Input sample
        mapping: Field mapping, e.g. {"user_field": "question", "assistant_field": "answer"}
    """
    conversations = []
    
    # Add user message
    if "user_field" in mapping:
        conversations.append({
            "from": "human",
            "value": sample[mapping["user_field"]]
        })
    
    # Add assistant message
    if "assistant_field" in mapping:
        conversations.append({
            "from": "gpt",
            "value": sample[mapping["assistant_field"]]
        })
    
    return {"conversations": conversations}


# Validation

def validate_sharegpt_format(sample: Dict[str, Any]) -> bool:
    """Validate that a sample is in correct ShareGPT format."""
    if "conversations" not in sample:
        return False
    
    conversations = sample["conversations"]
    if not isinstance(conversations, list) or len(conversations) == 0:
        return False
    
    for conv in conversations:
        if "from" not in conv or "value" not in conv:
            return False
        if conv["from"] not in ["human", "gpt", "system"]:
            return False
        if not isinstance(conv["value"], str):
            return False
    
    return True


def validate_dataset(data: List[Dict[str, Any]]) -> tuple[int, int]:
    """
    Validate entire dataset.
    
    Returns:
        tuple: (valid_count, invalid_count)
    """
    valid = 0
    invalid = 0
    
    for sample in data:
        if validate_sharegpt_format(sample):
            valid += 1
        else:
            invalid += 1
    
    return valid, invalid


# Main conversion logic

def load_data(input_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON or JSONL file."""
    path = Path(input_path)
    
    if not path.exists():
        print(f" ERROR: File not found: {input_path}")
        sys.exit(1)
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.jsonl':
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
        
        if not isinstance(data, list):
            print(" ERROR: Data must be a list of samples")
            sys.exit(1)
        
        print(f" Loaded {len(data)} samples from {input_path}")
        return data
    
    except json.JSONDecodeError as e:
        print(f" ERROR: Invalid JSON format: {e}")
        sys.exit(1)
    except Exception as e:
        print(f" ERROR: Failed to load data: {e}")
        sys.exit(1)


def convert_dataset(
    data: List[Dict[str, Any]],
    format_type: str,
    custom_mapping: Dict[str, str] = None
) -> List[Dict[str, Any]]:
    """
    Convert dataset to ShareGPT format.
    
    Args:
        data: Input data
        format_type: Format type (alpaca, oasst, dolly, custom, sharegpt)
        custom_mapping: Custom field mapping (for custom format)
    
    Returns:
        Converted data in ShareGPT format
    """
    if format_type == "sharegpt":
        print(" Data already in ShareGPT format, validating...")
        return data
    
    converters = {
        "alpaca": convert_alpaca_to_sharegpt,
        "oasst": convert_oasst_to_sharegpt,
        "dolly": convert_dolly_to_sharegpt,
    }
    
    if format_type == "custom":
        if not custom_mapping:
            print(" ERROR: Custom mapping required for custom format")
            sys.exit(1)
        converter = lambda x: convert_custom_to_sharegpt(x, custom_mapping)
    elif format_type in converters:
        converter = converters[format_type]
    else:
        print(f" ERROR: Unknown format type: {format_type}")
        sys.exit(1)
    
    converted_data = []
    failed = 0
    
    print(f"Converting {len(data)} samples...")
    for sample in tqdm(data):
        try:
            converted = converter(sample)
            converted_data.append(converted)
        except Exception as e:
            failed += 1
            if failed <= 5:  # Show first 5 errors
                print(f"  Failed to convert sample: {e}")
    
    if failed > 0:
        print(f"  WARNING: {failed} samples failed to convert")
    
    return converted_data


def save_data(data: List[Dict[str, Any]], output_path: str):
    """Save data to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f" Saved {len(data)} samples to {output_path}")
    
    except Exception as e:
        print(f" ERROR: Failed to save data: {e}")
        sys.exit(1)


# CLI

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert datasets to ShareGPT format for Llama-4 fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset file (JSON or JSONL)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset file (JSON)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["alpaca", "oasst", "dolly", "custom", "sharegpt"],
        default="alpaca",
        help="Input format type"
    )
    
    parser.add_argument(
        "--user-field",
        type=str,
        help="User message field name (for custom format)"
    )
    
    parser.add_argument(
        "--assistant-field",
        type=str,
        help="Assistant message field name (for custom format)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the dataset without converting"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        help="Only process first N samples (for testing)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    print("DATASET PREPARATION FOR LLAMA-4 FINE-TUNING")
    print()
    
    args = parse_arguments()
    
    # Load data
    data = load_data(args.input)
    
    # Sample data if requested
    if args.sample:
        data = data[:args.sample]
        print(f" Using first {args.sample} samples")
    
    # Validate only mode
    if args.validate_only:
        print("Validating dataset...")
        valid, invalid = validate_dataset(data)
        print(f" Valid samples: {valid}")
        if invalid > 0:
            print(f" Invalid samples: {invalid}")
            sys.exit(1)
        else:
            print(" All samples are valid!")
        return
    
    # Prepare custom mapping if needed
    custom_mapping = None
    if args.format == "custom":
        if not args.user_field or not args.assistant_field:
            print(" ERROR: --user-field and --assistant-field required for custom format")
            sys.exit(1)
        custom_mapping = {
            "user_field": args.user_field,
            "assistant_field": args.assistant_field
        }
    
    # Convert dataset
    converted_data = convert_dataset(data, args.format, custom_mapping)
    
    # Validate converted data
    print("Validating converted data...")
    valid, invalid = validate_dataset(converted_data)
    print(f" Valid samples: {valid}")
    if invalid > 0:
        print(f"  WARNING: {invalid} samples are invalid after conversion")
    
    # Save data
    save_data(converted_data, args.output)
    
    # Show example
    if len(converted_data) > 0:
        print("\n Example converted sample:")
        print(json.dumps(converted_data[0], indent=2, ensure_ascii=False)[:500])
        print("...")
    
    print("\n DATASET PREPARATION COMPLETED")
    print(f"\nYou can now use this dataset for training:")
    print(f"  python finetune_llama4_company.py --dataset_path {args.output}")


if __name__ == "__main__":
    main()
