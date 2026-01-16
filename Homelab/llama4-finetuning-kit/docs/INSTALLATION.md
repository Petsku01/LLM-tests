# Installation Guide

Complete installation instructions for the Llama-4 Fine-Tuning Kit.

## Prerequisites

Before installing, ensure you have:

- **Python 3.10 or higher**
- **NVIDIA GPU** with CUDA support (16GB+ VRAM recommended)
- **50GB+ free disk space**
- **Git** (for cloning the repository)

---

## Quick Install (Recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/llama4-finetuning-kit.git
cd llama4-finetuning-kit
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA (choose your CUDA version)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Install Unsloth (required for 4x speedup)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Step 4: Verify Installation

```bash
python scripts/check_environment.py
```

You should see:
```
 Python version OK
 CUDA available
 GPU: NVIDIA GeForce RTX 4090
 transformers: 4.45.0
 unsloth: 2024.11
...
 Environment setup complete!
```

---

## Manual Install

If you prefer to install packages individually:

### Core Dependencies

```bash
pip install torch>=2.3.0
pip install transformers>=4.45.0
pip install accelerate>=0.34.0
pip install peft>=0.12.0
pip install trl>=0.11.0
pip install bitsandbytes>=0.44.0
pip install datasets>=2.20.0
```

### Unsloth (Critical for Performance)

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Optional: Monitoring Tools

```bash
pip install wandb>=0.17.0       # Weights & Biases
pip install tensorboard>=2.17.0  # TensorBoard
```

### Optional: Evaluation Tools

```bash
pip install evaluate>=0.4.2
pip install rouge-score>=0.1.2
pip install sacrebleu>=2.4.0
```

---

## Development Setup

For contributors and developers:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

This adds:
- `pytest` for testing
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

---

## Troubleshooting Installation

### Issue: CUDA Not Found

**Error:**
```
RuntimeError: CUDA is not available
```

**Solution:**
1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Issue: Unsloth Installation Failed

**Error:**
```
ERROR: Could not find a version that satisfies the requirement unsloth
```

**Solution:**
Install directly from GitHub:
```bash
pip install git+https://github.com/unslothai/unsloth.git
```

### Issue: Version Conflicts

**Solution:**
Create a fresh virtual environment:
```bash
python -m venv fresh_venv
source fresh_venv/bin/activate
pip install --upgrade pip
# Then follow installation steps
```

---

## Platform-Specific Instructions

### Windows

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3.10 python3-pip python3-venv git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### macOS (No CUDA Support)

**Note:** Training on macOS without NVIDIA GPU is not recommended due to performance.

For CPU-only (very slow):
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

---

## Docker Installation (Coming Soon)

A Docker image will be provided for easy deployment:

```bash
docker pull yourorg/llama4-finetuning:latest
docker run --gpus all -v $(pwd)/data:/data llama4-finetuning
```

---

## Next Steps

After installation:

1. **Verify setup**: `python scripts/check_environment.py`
2. **Prepare dataset**: See [DATASET_FORMATS.md](DATASET_FORMATS.md)
3. **Start training**: `python finetune_llama4_company.py --help`
4. **Read examples**: Check `examples/` directory

---

## Getting Help

If you encounter issues:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Run environment checker: `python scripts/check_environment.py`
3. Open an issue on GitHub with your error details
