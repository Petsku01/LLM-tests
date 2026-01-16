# Dataset Formats Guide

This guide explains the supported dataset formats and how to convert between them.

## ShareGPT Format (Recommended)

The **ShareGPT format** is the native format used by the training script.

### Structure

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "What is machine learning?"
      },
      {
        "from": "gpt",
        "value": "Machine learning is a subset of artificial intelligence..."
      },
      {
        "from": "human",
        "value": "Can you give an example?"
      },
      {
        "from": "gpt",
        "value": "Sure! A common example is email spam filtering..."
      }
    ]
  }
]
```

### Key Features
- Supports multi-turn conversations
- Clear role separation (`human`, `gpt`, `system`)
- Flexible for various tasks

---

## Alpaca Format

Used by Stanford Alpaca and many instruction datasets.

### Structure

```json
[
  {
    "instruction": "Explain quantum computing",
    "input": "",
    "output": "Quantum computing is a type of computing..."
  },
  {
    "instruction": "Summarize the following text",
    "input": "Long text here...",
    "output": "Summary here..."
  }
]
```

### Conversion

```bash
python scripts/prepare_dataset.py \
    --input data/alpaca_data.json \
    --output datasets/converted.json \
    --format alpaca
```

---

## OpenAssistant (OASST) Format

Used by OpenAssistant datasets.

### Structure

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      },
      {
        "role": "assistant",
        "content": "I'm doing well, thank you!"
      }
    ]
  }
]
```

### Conversion

```bash
python scripts/prepare_dataset.py \
    --input data/oasst_data.json \
    --output datasets/converted.json \
    --format oasst
```

---

## Dolly Format

Used by Databricks Dolly datasets.

### Structure

```json
[
  {
    "instruction": "Summarize the following",
    "context": "Long context text...",
    "response": "Summary of the context..."
  }
]
```

### Conversion

```bash
python scripts/prepare_dataset.py \
    --input data/dolly_data.json \
    --output datasets/converted.json \
    --format dolly
```

---

## Custom Format

For datasets with custom field names.

### Example Structure

```json
[
  {
    "question": "What is AI?",
    "answer": "AI stands for Artificial Intelligence..."
  }
]
```

### Conversion

```bash
python scripts/prepare_dataset.py \
    --input data/custom_data.json \
    --output datasets/converted.json \
    --format custom \
    --user-field question \
    --assistant-field answer
```

---

## Best Practices

### 1. Data Quality
-  Remove duplicates
-  Filter low-quality samples
-  Check for proper formatting
-  Balance dataset distribution

### 2. Conversation Structure
- Keep conversations focused and coherent
- Include diverse topics and styles
- Maintain consistent formatting
- Add system messages when needed

### 3. Dataset Size
- **Minimum**: 500 samples for basic fine-tuning
- **Recommended**: 1,000-10,000 samples
- **Optimal**: 10,000+ high-quality samples

### 4. Token Length
- Check average token length
- Remove extremely long samples (>8192 tokens)
- Consider using packing for variable lengths

---

## Validation

Always validate your dataset before training:

```bash
python scripts/prepare_dataset.py \
    --input datasets/my_data.json \
    --validate-only
```

This will check for:
- Proper JSON formatting
- Required keys present
- Valid role names
- Non-empty values

---

## Example Datasets

Sample datasets are provided in the `data/` directory:

- `sample_sharegpt.json` - ShareGPT format examples
- `sample_alpaca.json` - Alpaca format examples

Use these as templates for your own datasets!
