# Troubleshooting Guide

## Common Issues and Solutions

### 1. Out of Memory (OOM) Errors

#### Symptoms
```
RuntimeError: CUDA out of memory
```

#### Solutions

**Reduce Batch Size:**
```bash
python finetune_llama4_company.py --per_device_train_batch_size 1
```

**Reduce Sequence Length:**
```bash
python finetune_llama4_company.py --max_seq_length 4096
```

**Reduce LoRA Rank:**
```bash
python finetune_llama4_company.py --lora_r 32 --lora_alpha 16
```

**Disable Packing (uses less memory but slower):**
```bash
python finetune_llama4_company.py --packing false
```

**Use Memory-Efficient Config:**
```bash
# Use the 16GB preset
python finetune_llama4_company.py \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --lora_r 32
```

---

### 2. Slow Training Speed

#### Symptoms
- Training is much slower than expected
- Low GPU utilization

#### Solutions

**Enable Sample Packing:**
```bash
python finetune_llama4_company.py --packing true
```

**Increase Batch Size:**
```bash
python finetune_llama4_company.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2
```

**Check Unsloth Installation:**
```bash
pip uninstall unsloth -y
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Verify CUDA is being used:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

### 3. Poor Model Quality

#### Symptoms
- Model outputs nonsense
- Model doesn't follow instructions
- Repetitive or incoherent text

#### Solutions

**Increase LoRA Rank:**
```bash
python finetune_llama4_company.py --lora_r 128 --lora_alpha 64
```

**Train for More Epochs:**
```bash
python finetune_llama4_company.py --num_train_epochs 3
```

**Lower Learning Rate:**
```bash
python finetune_llama4_company.py --learning_rate 1e-4
```

**Check Dataset Quality:**
- Ensure dataset is properly formatted
- Remove low-quality samples
- Balance dataset distribution

**Increase Dataset Size:**
- Aim for at least 1000+ high-quality samples
- Use data augmentation if needed

---

### 4. Import Errors

#### Symptoms
```
ModuleNotFoundError: No module named 'unsloth'
ImportError: cannot import name 'FastLanguageModel'
```

#### Solutions

**Install Unsloth:**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Install All Dependencies:**
```bash
pip install -r requirements.txt
```

**Check Environment:**
```bash
python scripts/check_environment.py
```

**Verify Python Version:**
```bash
python --version  # Should be 3.10+
```

---

### 5. Dataset Format Errors

#### Symptoms
```
ValueError: Dataset must have 'conversations' key
KeyError: 'from' or 'value'
```

#### Solutions

**Validate Dataset:**
```bash
python scripts/prepare_dataset.py --input your_data.json --format alpaca --validate-only
```

**Convert to ShareGPT Format:**
```bash
python scripts/prepare_dataset.py \
    --input your_data.json \
    --output datasets/converted.json \
    --format alpaca
```

**Check Format:**
```python
import json
with open('datasets/your_data.json') as f:
    data = json.load(f)
print(data[0])  # Should have 'conversations' key
```

---

### 6. CUDA Version Mismatch

#### Symptoms
```
RuntimeError: CUDA error: no kernel image is available for execution
Undefined symbol: cudaGetDriverEntryPointByVersion
```

#### Solutions

**Check CUDA Version:**
```bash
nvidia-smi
nvcc --version
```

**Reinstall PyTorch with Correct CUDA:**
```bash
# For CUDA 12.1
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 7. Training Crashes or Hangs

#### Symptoms
- Training stops unexpectedly
- Process hangs without progress
- No error message

#### Solutions

**Check Disk Space:**
```bash
df -h  # Ensure enough space for checkpoints
```

**Reduce Checkpoint Frequency:**
```bash
python finetune_llama4_company.py --save_steps 500
```

**Monitor GPU Memory:**
```bash
watch -n 1 nvidia-smi
```

**Check for NaN Loss:**
- Lower learning rate
- Check for corrupted data samples
- Ensure proper gradient clipping

---

### 8. Model Inference Issues

#### Symptoms
- Loaded model generates poor outputs
- Inference is very slow
- Model doesn't match training performance

#### Solutions

**Use Correct Model Path:**
```bash
# Use merged_4bit for inference
python inference.py --model_path outputs/llama4-v1/merged_4bit
```

**Adjust Generation Parameters:**
```bash
python inference.py \
    --model_path outputs/llama4-v1/merged_4bit \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.1
```

**Enable Inference Mode:**
```python
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)
```

---

## Performance Optimization Tips

### 1. Maximize GPU Utilization
- Use `nvidia-smi` to monitor GPU usage
- Aim for 90%+ VRAM utilization
- Increase batch size if VRAM is underutilized

### 2. Optimize Sequence Length
- Use shortest sequence length that covers your data
- Most samples should not be padded excessively
- Check average input length in your dataset

### 3. LoRA Configuration
- Start with r=64, alpha=32
- Increase rank if underfitting
- Decrease rank if overfitting or OOM

### 4. Learning Rate
- Start with 2e-4 for most cases
- Lower to 1e-4 for larger models or fine-tuning
- Higher to 5e-4 for small datasets or quick experiments

### 5. Batch Size vs Gradient Accumulation
- Effective batch size = batch_size * gradient_accumulation_steps
- Aim for effective batch size of 8-16
- Larger batches = more stable training

---

## Getting Help

If you're still experiencing issues:

1. **Check the logs** in `outputs/*/logs/`
2. **Run environment check**: `python scripts/check_environment.py`
3. **Search GitHub Issues**: Check if someone else had the same problem
4. **Create an issue** with:
   - Full error message
   - Hardware specifications
   - Command used
   - Environment details

---

## Additional Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [PyTorch CUDA Guide](https://pytorch.org/get-started/locally/)
