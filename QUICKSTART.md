# Quick Start

## Install

```bash
pip install -e .
python scripts/check_environment.py
```

## Train

```bash
python finetune_llama4_company.py \
    --dataset_path data/sample_sharegpt.json \
    --output_dir outputs/test \
    --num_train_epochs 1
```

## Your Data

Dataset format (ShareGPT):
```json
[
  {"conversations": [
    {"from": "human", "value": "Question"},
    {"from": "gpt", "value": "Answer"}
  ]}
]
```

Train:
```bash
python finetune_llama4_company.py \
    --dataset_path your_data.json \
    --output_dir outputs/v1 \
    --lora_r 16
```

## Inference

```bash
python inference.py \
    --model_name outputs/v1/merged_4bit \
    --prompt "Your prompt"
```

## Troubleshooting

Out of memory: `--per_device_train_batch_size 1`

See [README.md](README.md) for details.
