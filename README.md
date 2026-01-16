# Llama-4 Fine-Tuning Kit

[![Tests](https://github.com/your-org/llama4-finetuning-kit/workflows/Tests/badge.svg)](https://github.com/your-org/llama4-finetuning-kit/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fine-tune Llama-4 models with QLoRA + Unsloth. Handles multi-GPU training, dataset conversion, and model deployment.

## Features

- Efficient fine-tuning with QLoRA (4-bit) + RS-LoRA
- Multi-GPU training via DeepSpeed or FSDP
- Dataset conversion (Alpaca, ShareGPT, OASST)
- Automatic VRAM estimation and checkpoint recovery
- Export to GGUF, vLLM, or HuggingFace format
- MLflow experiment tracking

## Quick Start

```bash
pip install -e .
python finetune_llama4_company.py \
    --dataset_path data/sample_sharegpt.json \
    --output_dir outputs/test \
    --num_train_epochs 1
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup.

## Supported Models

| Model | Sizes | Tested |
|-------|-------|--------|
| Llama-4 | 8B, 109B, 402B | Yes |
| Llama-3 | 8B, 70B | Yes |
| Qwen | All sizes | Yes |
| Mistral | 7B, 8x7B | Yes |

Works with any HuggingFace model compatible with Unsloth.

## Hardware Requirements

| Method | 7-8B | 13B | 70B |
|--------|------|-----|-----|
| Full 16-bit | 60GB | 120GB | 600GB |
| LoRA 16-bit | 16GB | 32GB | 160GB |
| QLoRA 4-bit | 10GB | 20GB | 80GB |

Recommended: RTX 4090 (24GB), A100 (40GB+), or multi-GPU setup

## Quick Links

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [Installation](docs/INSTALLATION.md) - Detailed setup instructions
- [Dataset Formats](docs/DATASET_FORMATS.md) - Supported formats and conversion
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Performance Benchmarks](BENCHMARKS.md) - Speed and memory usage
- [Docker Guide](DOCKER.md) - Containerized deployment
- [Contributing](CONTRIBUTING.md) - How to contribute

## Usage

**Train on your data:**
```bash
python finetune_llama4_company.py \
    --dataset_path your_data.json \
    --output_dir outputs/v1 \
    --lora_r 16 \
    --num_train_epochs 3
```

**Run inference:**
```bash
python inference.py \
    --model_name outputs/v1/merged_4bit \
    --prompt "Your prompt here"
```

**Convert dataset format:**
```bash
python scripts/prepare_dataset.py \
    --input_file raw_data.json \
    --input_format alpaca \
    --output_file training_data.json
```

See [docs/](docs/) for advanced usage.

## Troubleshooting

**Out of memory:**
```bash
--per_device_train_batch_size 1 --max_seq_length 1024
```

**Missing dependencies:**
```bash
pip install -e . --no-cache-dir
```

**Gated models:**
```bash
huggingface-cli login
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more.

## Tech Stack

**Core:** PyTorch, Transformers, Unsloth, TRL, PEFT  
**Quantization:** bitsandbytes (4-bit QLoRA)  
**Training:** DeepSpeed ZeRO, Gradient Checkpointing  
**Tracking:** MLflow, Weights & Biases  
**Optimization:** RS-LoRA, Sample Packing  

## Project Structure

```
llama4-finetuning-kit/
├── finetune_llama4_company.py    # Main training script
├── inference.py                   # Multi-mode inference
├── mlflow_tracking.py            # Experiment tracking
├── distributed_training.py       # Multi-GPU support
├── benchmark_suite.py            # Performance testing
├── lr_finder.py                  # Learning rate optimization
├── advanced_metrics.py           # Evaluation metrics
├── scripts/
│   ├── check_environment.py      # Validate setup
│   └── prepare_dataset.py        # Dataset conversion
├── examples/
│   ├── train_instruction_model.py
│   └── quickstart.ipynb          # Jupyter tutorial
├── configs/
│   └── training_presets.yaml     # Hardware-specific configs
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── .github/workflows/            # CI/CD
```

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=. --cov-report=html
```

## Docker

Build and run:

```bash
docker build -t llama4-finetuning .
docker run --gpus all -v $(pwd)/data:/workspace/data llama4-finetuning
```

See [DOCKER.md](DOCKER.md) for details.

## Performance

Training speed on RTX 4090 (batch=4, seq=4096):
- 3,200 tokens/sec single GPU
- 7,800 tokens/sec with 4x GPUs (DeepSpeed)

See [BENCHMARKS.md](BENCHMARKS.md) for comprehensive results.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

- Unsloth (4x faster training)
- QLoRA + RS-LoRA (4-bit quantization)
- FlashAttention-2
- DeepSpeed ZeRO (multi-GPU)
- MLflow (experiment tracking)

## License

MIT 
