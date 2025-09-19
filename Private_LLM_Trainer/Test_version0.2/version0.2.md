# Theoretical Private LLM Trainer 

This is a self-contained, high-performance Python script for training GPT-2 style language models privately and offline on local hardware. It prioritizes ensuring no information leaves your machine during training.

## Key Features

- **Offline Operation**: Runs entirely locally after initial setup. Uses Hugging Face Transformers in offline mode.
- **Data Privacy**: Loads training data from local directories, files, or raw text strings. Supports TXT, MD, JSON, and JSONL formats.
- **Customizable Model**: Builds GPT-2 architectures with configurable parameters like hidden size, layers, heads, and context length.
- **Training Optimizations**: Includes mixed precision (AMP), gradient accumulation, checkpointing, model compilation (if available), and TF32 for compatible GPUs.
- **Efficient Data Handling**: Automatic cleaning, filtering, tokenization, and caching of datasets.
- **Monitoring & Saving**: Logs training progress, evaluates on validation set, saves checkpoints, and includes generation testing.
- **REPL-Friendly**: Designed for easy integration and extension.

## Requirements

- Python 3.8 or higher
- PyTorch (with CUDA for GPU support)
- Transformers library (`pip install transformers`)

**Note**: The tokenizer may download from Hugging Face on first run if not cached. Subsequent runs can be fully offline by setting environment variables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

No internet access is needed during training. Additional libraries like NumPy are not required but can be used if needed for extensions.

## Installation

1. Download or copy the script: `llm_training_pipeline.py`.
2. Install dependencies:
   ```
   pip install torch transformers
   ```
3. For GPU acceleration: Ensure PyTorch is installed with CUDA support.

## Quick Start

Run the script with default settings and example data:

```
python llm_training_pipeline.py

```

This will:
- Initialize a small GPT-2 model (6 layers, 768 hidden sizes).
- Train on provided sample texts for 2 epochs.
- Save the trained model to `./model/`.
- Generate and print sample text completions.

## Customization

Edit the `main()` function to configure training:

```python
config = Config(
    vocab_size=50257,      # Vocabulary size
    hidden_size=768,       # Embedding dimension
    num_layers=12,         # Transformer layers
    num_heads=12,          # Attention heads
    max_length=1024,       # Maximum sequence length
    dropout=0.1,           # Dropout rate
    batch_size=4,          # Batch size
    accumulation_steps=4,  # Gradient accumulation steps
    learning_rate=3e-4,    # Initial learning rate
    min_lr=1e-6,           # Minimum LR for scheduler
    weight_decay=0.01,     # Weight decay
    epochs=3,              # Number of epochs
    warmup_ratio=0.1,      # Warmup steps ratio
    max_grad_norm=1.0,     # Gradient clipping norm
    mixed_precision=True,  # Use AMP (GPU only)
    gradient_checkpointing=True,  # Memory optimization
    compile_model=True,    # Use torch.compile if available
    tf32=True,             # TF32 for Ampere GPUs
    train_split=0.9,       # Train/validation split
    min_text_len=32,       # Minimum text length
    model_dir="./model",   # Model save directory
    cache_dir="./cache",   # Cache directory
    log_every=10,          # Log frequency (steps)
    eval_every=100,        # Evaluation frequency
    save_every=500,        # Checkpoint frequency
    seed=42,               # Random seed
    workers=0              # DataLoader workers
)

trainer = Trainer(config)

data_sources = {
    'directory': './data',  # Load from directory (recursive)
    'files': ['file1.txt', 'file2.json'],  # Specific files
    'texts': ["Sample text 1", "Sample text 2"]  # Raw strings
}

trainer.run(data_sources)
```

### Data Sources

Provide data via a dictionary:
- `'directory'`: Path to a folder; loads all supported files recursively.
- `'files'`: List of file paths.
- `'texts'`: List of strings.

Data is cleaned (normalized whitespace, limited newlines) and filtered by minimum length.

## Training Process

1. **Data Loading & Processing**: Uses `DataProcessor` to load, clean, and filter texts.
2. **Tokenization**: Caches tokenized datasets in `./cache/`.
3. **Model Creation**: Builds GPT-2 with advanced configs and optimizations.
4. **Training Loop**: Uses AdamW optimizer, cosine scheduler with warmup, gradient clipping.
5. **Evaluation**: Optional validation set; saves best model based on loss.
6. **Checkpoints**: Saves to `./model/{name}/` (e.g., `best`, `final`, `step_XXX`).
7. **Generation**: Post-training text generation with sampling options.

## Text Generation

After training, generate text using:

```python
output = trainer.generate(
    prompt="Your prompt here",
    max_tokens=100,
    temperature=0.8,
    top_p=0.9,
    top_k=50
)
print(output)
```

## Performance Tips

- **Memory Management**: Reduce `batch_size` or enable `gradient_checkpointing` for larger models.
- **Speed**: Use GPU with ≥8GB VRAM; enable `compile_model` on PyTorch 2.0+.
- **Caching**: Delete `./cache/` to re-tokenize data if changed.
- **Scaling**: Increase `num_layers`/`hidden_size` for better models, but monitor VRAM usage.

## Limitations

- GPT-2 based; for advanced architectures, extend the `ModelFactory`.
- Single-device training only (no distributed support).
- Data must fit in memory during initial loading/tokenization.

-pk
