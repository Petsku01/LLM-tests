# Docker Usage Guide

## Quick Start

### Build Image

```bash
docker build -t llama4-finetuning .
```

### Run Training

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  llama4-finetuning \
  --dataset_path data/your_data.json \
  --output_dir outputs/v1 \
  --num_train_epochs 1
```

## Common Commands

### Interactive Shell

```bash
docker run --gpus all -it --entrypoint /bin/bash llama4-finetuning
```

### Mount Custom Config

```bash
docker run --gpus all \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/data:/workspace/data \
  llama4-finetuning \
  --dataset_path data/train.json
```

### Run Inference

```bash
docker run --gpus all llama4-finetuning \
  python3 inference.py \
  --model_name outputs/v1/merged_4bit \
  --prompt "Your question"
```

## Docker Compose (Multi-GPU)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  training:
    build: .
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    volumes:
      - ./data:/workspace/data
      - ./outputs:/workspace/outputs
    command: >
      --dataset_path data/train.json
      --output_dir outputs/v1
```

Run:

```bash
docker-compose up
```

## Building for Different CUDA Versions

### CUDA 11.8

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# ... rest of Dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only (Not Recommended)

```dockerfile
FROM python:3.10-slim
# ... install dependencies without CUDA
```

## Volume Mounts

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/configs:/workspace/configs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  llama4-finetuning
```

## Environment Variables

```bash
docker run --gpus all \
  -e HF_TOKEN=your_token \
  -e WANDB_API_KEY=your_key \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  llama4-finetuning
```

## Tips

- Always use `--gpus all` to enable GPU access
- Mount volumes for data persistence
- Use `.dockerignore` to reduce build context
- Multi-stage builds can reduce image size
- Consider using pre-built base images for faster builds
