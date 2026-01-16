# Finnish Culture LLM

A custom transformer-based language model for Finnish text generation. Upload Finnish texts, train a model, and generate new content.

## Features

- Upload text files, PDFs, or images (OCR support)
- Train custom transformer model (24 layers, 512 embedding dim)
- Generate Finnish text with trained model
- Web-based interface with React
- Model persistence (saves trained models)
- GPU acceleration support

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for image processing)

**Windows:**
- Download from https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-fin
```

**Mac:**
```bash
brew install tesseract tesseract-lang
```

### 3. Verify PyTorch Installation

Check if CUDA is available:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

If CUDA is not available but you have an NVIDIA GPU, reinstall PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### 1. Start Backend Server

```bash
python app.py
```

Server runs on http://localhost:5000

### 2. Open Frontend

Open `index.html` in your web browser, or serve it with:

```bash
python -m http.server 8000
```

Then navigate to http://localhost:8000

### 3. Train Model

1. **Upload Data**: Upload Finnish text files, PDFs, or images
   - Text: `.txt` files (e.g., Kalevala from Project Gutenberg)
   - PDF: Digital or scanned PDFs
   - Images: Scanned manuscripts (requires Tesseract)

2. **Train**: Click "Start Training" (requires 1000+ characters)
   - Training takes 30-60 minutes per epoch on GPU
   - 5-10 minutes per epoch on modern CPU
   - Model auto-saves after training

3. **Generate**: Enter seed text and generate new Finnish content

## Model Architecture

- **Type**: Transformer Encoder
- **Vocabulary**: 20,000 BPE tokens
- **Embedding**: 512 dimensions
- **Layers**: 24 transformer layers
- **Attention Heads**: 16
- **Feed-Forward**: 2048 dimensions
- **Context Length**: 128 tokens
- **Parameters**: ~100M trainable parameters

## API Endpoints

### POST /upload
Upload file for training data.

**Request:**
- Form data with file field
- Supported: .txt, .pdf, .jpg, .jpeg, .png

**Response:**
```json
{
  "message": "File processed",
  "chars_extracted": 5000,
  "total_chars": 15000
}
```

### POST /train
Train model on uploaded data.

**Response:**
```json
{
  "log": ["Epoch 1/5, Average Loss: 4.2341", ...],
  "message": "Training completed successfully",
  "dataset_size": 15000
}
```

### POST /generate
Generate text from seed.

**Request:**
```json
{
  "seed": "Väinämöinen, vanha viisas",
  "max_length": 200
}
```

**Response:**
```json
{
  "generated": "Väinämöinen, vanha viisas...",
  "seed": "Väinämöinen, vanha viisas",
  "length": 850
}
```

### GET /status
Check system status.

**Response:**
```json
{
  "model_trained": true,
  "tokenizer_ready": true,
  "dataset_size": 15000,
  "device": "cuda",
  "cuda_available": true
}
```

### POST /reset
Reset all data and models.

## Performance

| Hardware | Training Speed | Inference Speed |
|----------|---------------|----------------|
| RTX 4090 | ~2 min/epoch | ~50 tokens/sec |
| RTX 3080 | ~5 min/epoch | ~30 tokens/sec |
| CPU (16 cores) | ~10 min/epoch | ~5 tokens/sec |

## Data Sources

**Public Finnish Texts:**
- [Kalevala - Project Gutenberg](https://www.gutenberg.org/ebooks/5186)
- [SKVR - Finnish Folk Poetry](https://skvr.fi)
- [Finnish Literature - Kirjasampo](https://kirjasampo.fi)

## Troubleshooting

### ImportError: No module named 'flask_cors'
```bash
pip install flask-cors
```

### Tesseract not found
Ensure Tesseract is installed and in PATH.

### CUDA out of memory
Reduce batch_size in app.py (line 28):
```python
batch_size = 8  # Reduce from 32
```

### Model not training
Ensure dataset has at least 1000 characters.

## License

MIT License - Free for educational and commercial use.

## Warning

This is a demonstration project. For production use:
- Add authentication
- Implement rate limiting
- Use production WSGI server (gunicorn/uwsgi)
- Add model versioning
- Implement proper error handling
- Add input validation and sanitization
