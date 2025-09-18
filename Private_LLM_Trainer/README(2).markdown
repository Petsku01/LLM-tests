# Private LLM Training Pipeline

## Overview
This project provides a highly optimized, privacy-focused training pipeline for fine-tuning a GPT-2 based language model. The pipeline is designed to run locally, ensuring no data is sent to external servers. It incorporates modern training optimizations such as mixed precision training, gradient checkpointing, and model compilation for maximum performance.

## Features
- **Privacy First**: All data processing and training occur locally, with environment settings to disable external connections (e.g., Hugging Face Hub, Weights & Biases).
- **Efficient Training**: Supports mixed precision training, gradient accumulation, and gradient checkpointing to optimize memory usage and speed.
- **Flexible Data Handling**: Processes text from both directories (`.txt` and `.json` files) and raw text inputs.
- **Customizable Model**: Configurable GPT-2 architecture with adjustable hyperparameters (e.g., hidden size, number of layers, heads).
- **Text Generation**: Includes a generation function to produce text based on trained models.

## Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- A CUDA-compatible GPU (optional, for faster training)

Install dependencies using:
```bash
pip install torch transformers
```

## Usage
1. **Prepare Training Data**:
   - Place text files (`.txt`) or JSON files (`.json`) in a directory (e.g., `./training_data`), or provide raw text directly in the `data_sources` dictionary.
   - Ensure texts are at least 50 characters long to be included in training.

2. **Configure the Pipeline**:
   - Modify the `Config` class in `llm_training_pipeline.py` to adjust model architecture, training parameters, or file paths.
   - Example configuration:
     ```python
     config = Config(
         hidden_size=768,
         num_layers=6,
         num_heads=12,
         batch_size=2,
         epochs=2,
         learning_rate=3e-4,
         mixed_precision=True,
         gradient_checkpointing=True
     )
     ```

3. **Run the Training**:
   - Execute the script:
     ```bash
     python llm_training_pipeline.py
     ```
   - The pipeline will:
     - Load and preprocess data from the specified sources.
     - Train the model with the provided configuration.
     - Save checkpoints and the final model to the `./model` directory.
     - Generate sample outputs for testing.

4. **Test Text Generation**:
   - After training, the script automatically tests the model with sample prompts.
   - Example output:
     ```
     Prompt: The future of AI is
     Output: The future of AI is bright, with advancements in privacy-focused training enabling secure, local model development.
     ```

## File Structure
- `llm_training_pipeline.py`: Main script containing the training pipeline, model configuration, and data handling.
- `./training_data/`: Directory for input text or JSON files (optional).
- `./model/`: Directory where trained models and checkpoints are saved.

## Data Sources
The pipeline supports two types of data sources:
- **Directory**: Specify a path to a folder containing `.txt` or `.json` files.
- **Raw Texts**: Provide a list of text strings directly in the `data_sources` dictionary.

Example `data_sources`:
```python
data_sources = {
    'directory': './training_data',
    'texts': [
        "This is a completely private language model trainer.",
        "All processing happens locally on your machine."
    ]
}
```

## Training Optimizations
- **Mixed Precision**: Enabled by default if a CUDA GPU is available, reducing memory usage and speeding up training.
- **Gradient Checkpointing**: Reduces memory consumption by trading compute for memory.
- **Gradient Accumulation**: Allows larger effective batch sizes on limited hardware.
- **Cosine Learning Rate Schedule**: Includes warmup for stable training.
- **Model Compilation**: Uses `torch.compile` for faster execution (if supported).

## Output
- **Checkpoints**: Saved at regular intervals (every 500 steps by default) in `./model/checkpoint_<step>`.
- **Best Model**: Saved when validation loss improves (if validation data is available).
- **Final Model**: Saved at the end of training in `./model/final`.

## Notes
- Ensure sufficient disk space for model checkpoints.
- Training on CPU is supported but significantly slower than GPU.
- For large datasets, adjust `batch_size` and `accumulation_steps` to fit your hardware.
- The minimum text length (50 characters) can be modified in the `Config` class.

## License
This project is licensed under the MIT License.