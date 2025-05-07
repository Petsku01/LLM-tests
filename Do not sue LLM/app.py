import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tqdm import tqdm
import numpy as np
import PyPDF2
from PIL import Image
import pytesseract

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'jpg', 'jpeg', 'png'}

# Parmeters
vocab_size = 20000  # For CPU-killer
embed_dim = 512     # Large embeddings
num_layers = 24     # Large transformer
num_heads = 16      # More attention heads
ff_dim = 2048       # Large feed-forward
context_length = 128
batch_size = 32     # Adjusted for larger model
dropout = 0.1
num_epochs = 5
learning_rate = 2e-4
warmup_steps = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforming model
class TransformerLLM(nn.Module):
    def __init__(self):
        super(TransformerLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, context_length, embed_dim))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x) + self.pos_embedding[:, :T, :]
        x = self.dropout(x)
        x = self.transformer(x)
        logits = self.fc(x)
        return logits

# Global model and tokens
model = None
tokenizer = None
dataset_text = ""

# File handling
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(filepath):
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif filepath.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        elif filepath.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(filepath)
            text = pytesseract.image_to_string(image, lang='fin')  # Finnish OCR
            return text
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return ""
    return ""

# Train tokenizing
def train_tokenizer(text):
    global tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.train_from_iterator([text], trainer)

# Tokenize text
def tokenize(text):
    encoded = tokenizer.encode(text)
    return encoded.ids

# Prepare dataset
def get_batches(tokens, batch_size, context_length):
    sequences = []
    for i in range(0, len(tokens) - context_length, 1):
        seq = tokens[i:i + context_length]
        target = tokens[i + 1:i + context_length + 1]
        sequences.append((seq, target))
    sequences = sequences[:len(sequences) // batch_size * batch_size]
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        inputs = torch.tensor([seq for seq, _ in batch_seqs], dtype=torch.long).to(device)
        targets = torch.tensor([tgt for _, tgt in batch_seqs], dtype=torch.long).to(device)
        yield inputs, targets

# Sample token
def sample_token(logits, top_k=50, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs <= top_p
    sorted_probs = sorted_probs * mask.float()
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, 1)
    token = top_k_indices.gather(-1, sorted_indices[idx])
    return token.item()

# Generate text
def generate(seed_text, max_length=200):
    global model, tokenizer
    model.eval()
    tokens = tokenize(seed_text)[:context_length]
    generated = tokens.copy()
    
    for _ in range(max_length):
        input_tokens = torch.tensor([generated[-context_length:]], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(input_tokens)[:, -1, :]
        next_token = sample_token(logits)
        generated.append(next_token)
    
    return tokenizer.decode(generated)

# Training loop
def train_model():
    global model, dataset_text
    model = TransformerLLM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(float(step) / warmup_steps, 1.0))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    tokens = tokenize(dataset_text)
    training_log = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in get_batches(tokens, batch_size, context_length):
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        training_log.append(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    
    return training_log

# API endpoints
@app.route('/upload', methods=['POST'])
def upload_file():
    global dataset_text
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = process_file(filepath)
        if text:
            dataset_text += text + "\n"
            return jsonify({'message': f'File {filename} processed, {len(text)} characters extracted'}), 200
        return jsonify({'error': 'No text extracted'}), 400
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/train', methods=['POST'])
def train():
    global dataset_text
    if not dataset_text:
        return jsonify({'error': 'No dataset available'}), 400
    train_tokenizer(dataset_text)
    log = train_model()
    return jsonify({'log': log}), 200

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    seed = data.get('seed', 'Väinämöinen, vanha viisas')
    generated = generate(seed)
    return jsonify({'generated': generated}), 200

if __name__ == '__main__':
    app.run(debug=True)
