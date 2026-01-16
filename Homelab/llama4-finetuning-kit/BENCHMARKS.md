# Performance Benchmarks

## Training Performance

### Single GPU Performance

| GPU | VRAM | Batch Size | Seq Length | Tokens/sec | Notes |
|-----|------|------------|------------|------------|-------|
| RTX 4090 | 24GB | 2 | 8192 | 2,400 | With sample packing |
| RTX 4090 | 24GB | 4 | 4096 | 3,200 | Optimal for most tasks |
| RTX 3090 | 24GB | 2 | 8192 | 1,800 | Similar to 4090 |
| A100 40GB | 40GB | 4 | 8192 | 4,800 | Best single-GPU option |
| A100 80GB | 80GB | 8 | 8192 | 6,400 | Maximum throughput |

### Multi-GPU Performance (DeepSpeed ZeRO-2)

| Setup | Effective Batch | Tokens/sec | Speedup |
|-------|----------------|------------|---------|
| 2x RTX 4090 | 8 | 4,200 | 1.75x |
| 4x RTX 4090 | 16 | 7,800 | 3.25x |
| 2x A100 40GB | 16 | 8,900 | 1.85x |
| 4x A100 40GB | 32 | 16,500 | 3.44x |

## Memory Usage

### Model Size vs VRAM (Llama-4-8B)

| Configuration | Model VRAM | Peak VRAM | Recommended |
|--------------|-----------|-----------|-------------|
| 16-bit Full | 32GB | 45GB | Not practical |
| 16-bit LoRA | 16GB | 24GB | High-end GPUs |
| 4-bit QLoRA r=64 | 6GB | 14GB | RTX 3090+ |
| 4-bit QLoRA r=32 | 5GB | 11GB | RTX 3080+ |
| 4-bit QLoRA r=16 | 4.5GB | 9GB | Budget GPUs |

### LoRA Rank Impact

| LoRA Rank | Trainable Params | Additional VRAM | Quality |
|-----------|------------------|----------------|---------|
| 16 | ~8M | +0.5GB | Good |
| 32 | ~16M | +1GB | Better |
| 64 | ~32M | +2GB | Best |
| 128 | ~64M | +4GB | Overkill |

## Inference Performance

### Tokens per Second

| Model Type | GPU | Batch=1 | Batch=4 | Batch=8 |
|------------|-----|---------|---------|---------|
| Merged 4-bit | RTX 4090 | 85 | 280 | 450 |
| Merged 16-bit | RTX 4090 | 65 | 220 | 380 |
| Merged 4-bit | A100 40GB | 110 | 380 | 620 |

### Latency (First Token)

| Model | GPU | Latency |
|-------|-----|---------|
| 4-bit | RTX 4090 | 45ms |
| 16-bit | RTX 4090 | 65ms |
| 4-bit | A100 | 32ms |

## Training Time Estimates

### Time to Complete 1 Epoch

| Dataset Size | Seq Length | GPU | Batch | Time |
|--------------|------------|-----|-------|------|
| 1K samples | 2048 | RTX 4090 | 4 | 8 min |
| 10K samples | 2048 | RTX 4090 | 4 | 80 min |
| 100K samples | 2048 | RTX 4090 | 4 | 13 hours |
| 10K samples | 8192 | A100 40GB | 4 | 45 min |

## Cost Estimates (Cloud)

### AWS EC2 On-Demand (us-east-1)

| Instance | GPU | Cost/hour | 1K samples | 10K samples |
|----------|-----|-----------|------------|-------------|
| g5.xlarge | A10G 24GB | $1.01 | $0.13 | $1.35 |
| p4d.24xlarge | A100 40GB x8 | $32.77 | $0.18 | $1.80 |
| p5.48xlarge | H100 80GB x8 | $98.32 | $0.12 | $1.20 |

### Lambda Labs GPU Cloud

| GPU | Cost/hour | 1K samples | 10K samples |
|-----|-----------|------------|-------------|
| RTX 4090 | $0.50 | $0.07 | $0.67 |
| A100 40GB | $1.29 | $0.10 | $0.97 |
| A100 80GB | $1.69 | $0.08 | $0.75 |

## Optimization Tips

**For Maximum Speed:**
- Use `--packing=true` (30% faster)
- Enable gradient checkpointing with Unsloth
- Use A100 or H100 GPUs
- Multi-GPU with DeepSpeed ZeRO-2

**For Maximum Quality:**
- Increase LoRA rank to 64-128
- Train for 3+ epochs
- Use longer sequences (8192)
- Lower learning rate (1e-4)

**For Memory Efficiency:**
- Use 4-bit quantization
- Lower LoRA rank (16-32)
- Reduce batch size to 1
- Shorter sequences (2048-4096)

## Benchmarking Your Setup

Run the built-in benchmark:

```bash
python benchmark_suite.py \
    --model_name unsloth/Llama-4-8B-Instruct-bnb-4bit \
    --batch_size 2 \
    --seq_length 4096
```

This will output:
- Training throughput (tokens/sec)
- Peak memory usage
- Inference latency
- Comparison to reference hardware

## Notes

- All benchmarks use Unsloth optimizations
- Performance varies by dataset complexity
- Numbers are approximate and may vary
- Multi-GPU scaling is not linear due to communication overhead
