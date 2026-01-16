"""
Improved Finnish Culture LLM - Production-Ready Version
Fixes: Proper architecture, security, concurrency, background tasks
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from werkzeug.utils import secure_filename
from tqdm import tqdm
import numpy as np
import PyPDF2
from PIL import Image
import pytesseract
import logging
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(title="Finnish Culture LLM", version="2.0.0")

# CORS - Restrict to localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
MODEL_FOLDER = BASE_DIR / 'models'
SESSIONS_FOLDER = BASE_DIR / 'sessions'

UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_FOLDER.mkdir(exist_ok=True)
SESSIONS_FOLDER.mkdir(exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TOTAL_DATASET_SIZE = 500 * 1024 * 1024  # 500MB
MAX_GENERATION_LENGTH = 2000
MIN_DATASET_SIZE = 1000
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'jpg', 'jpeg', 'png'}

# Model hyperparameters
CONFIG = {
    'vocab_size': 30000,
    'embed_dim': 512,
    'num_layers': 12,
    'num_heads': 8,
    'ff_dim': 2048,
    'context_length': 256,  # Increased from 128
    'batch_size': 8,  # Reduced to be safe
    'dropout': 0.1,
    'num_epochs': 3,
    'learning_rate': 5e-4,
    'weight_decay': 0.01,  # Added weight decay
    'warmup_steps': 1000,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ============================================================================
# IMPROVED MODEL ARCHITECTURE
# ============================================================================

class CausalSelfAttention(nn.Module):
    """Self-attention with causal masking for generation."""
    
    def __init__(self, d_model: int, num_heads: int, context_length: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Causal mask (lower triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)).unsqueeze(0)
        )
    
    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~self.mask[:, :T, :T], float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        return out


class TransformerLLMImproved(nn.Module):
    """Improved decoder-only transformer for text generation."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        
        # Sinusoidal positional embeddings (better than learned)
        self.register_buffer("pos_emb", self._get_positional_embeddings(
            config['context_length'], config['embed_dim']
        ))
        
        self.dropout = nn.Dropout(config['dropout'])
        
        # Decoder layers with residual connections
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': CausalSelfAttention(config['embed_dim'], config['num_heads'], config['context_length']),
                'norm1': nn.LayerNorm(config['embed_dim']),
                'ff': nn.Sequential(
                    nn.Linear(config['embed_dim'], config['ff_dim']),
                    nn.GELU(),
                    nn.Linear(config['ff_dim'], config['embed_dim']),
                ),
                'norm2': nn.LayerNorm(config['embed_dim']),
            })
            for _ in range(config['num_layers'])
        ])
        
        self.final_norm = nn.LayerNorm(config['embed_dim'])
        self.lm_head = nn.Linear(config['embed_dim'], config['vocab_size'])
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _get_positional_embeddings(seq_len: int, d_model: int) -> torch.Tensor:
        """Sinusoidal positional embeddings."""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        
        # Embeddings + positional
        x = self.embedding(x)
        x = x + self.pos_emb[:, :T, :]
        x = self.dropout(x)
        
        # Decoder layers with residual connections
        for layer in self.decoder_layers:
            # Attention block
            x_norm = layer['norm1'](x)
            attn_out = layer['attn'](x_norm)
            x = x + attn_out
            
            # Feed-forward block
            x_norm = layer['norm2'](x)
            ff_out = layer['ff'](x_norm)
            x = x + ff_out
        
        # Final norm and head
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits


# ============================================================================
# SESSION MANAGEMENT - Per-user isolation
# ============================================================================

class SessionManager:
    """Manages per-user sessions with data isolation."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self) -> str:
        """Create new session."""
        session_id = str(uuid.uuid4())[:8]
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        self.sessions[session_id] = {
            'created_at': datetime.now(),
            'dataset_path': session_dir / 'dataset.txt',
            'model_path': session_dir / 'model.pt',
            'tokenizer_path': session_dir / 'tokenizer.json',
            'status': 'idle',
            'progress': 0,
            'error': None,
        }
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Dict:
        """Get session data."""
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        return self.sessions[session_id]
    
    def update_status(self, session_id: str, status: str, progress: int = 0, error: str = None):
        """Update session status."""
        session = self.get_session(session_id)
        session['status'] = status
        session['progress'] = progress
        session['error'] = error


session_manager = SessionManager(SESSIONS_FOLDER)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(filepath: Path) -> str:
    """Extract text from file with size limits."""
    try:
        file_size = filepath.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size} > {MAX_FILE_SIZE}")
        
        if filepath.suffix.lower() == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        elif filepath.suffix.lower() == '.pdf':
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    try:
                        text += page.extract_text() or ""
                    except Exception as e:
                        logger.warning(f"Error extracting PDF page: {e}")
                return text
        
        elif filepath.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            image = Image.open(filepath)
            text = pytesseract.image_to_string(image, lang='fin')
            return text
    
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        raise
    
    return ""


def validate_max_length(length: int) -> int:
    """Validate and clamp max_length."""
    if not isinstance(length, int):
        raise ValueError("max_length must be integer")
    if length < 1:
        raise ValueError("max_length must be > 0")
    return min(length, MAX_GENERATION_LENGTH)


# ============================================================================
# TOKENIZER MANAGEMENT
# ============================================================================

def train_tokenizer(text: str, tokenizer_path: Path) -> Tokenizer:
    """Train BPE tokenizer with proper handling."""
    if len(text) < 100:
        raise ValueError("Text too short for tokenizer training")
    
    tokenizer = Tokenizer(BPE(dropout=0.1))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=CONFIG['vocab_size'],
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
    )
    
    # Train on full text (not chunks) for better BPE merges
    tokenizer.train_from_iterator([text], trainer, length=len(text))
    tokenizer.save(str(tokenizer_path))
    
    logger.info(f"Tokenizer trained with vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """Load existing tokenizer."""
    return Tokenizer.from_file(str(tokenizer_path))


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_batches(tokens: List[int], batch_size: int, context_length: int):
    """Generator for memory-efficient batch creation."""
    sequences = []
    
    for i in range(len(tokens) - context_length):
        seq = tokens[i:i + context_length]
        target = tokens[i + 1:i + context_length + 1]
        sequences.append((seq, target))
    
    # Shuffle
    indices = np.random.permutation(len(sequences))
    sequences = [sequences[i] for i in indices]
    
    # Batch creation
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        
        inputs = torch.tensor(
            [seq for seq, _ in batch_seqs],
            dtype=torch.long
        ).to(device)
        targets = torch.tensor(
            [tgt for _, tgt in batch_seqs],
            dtype=torch.long
        ).to(device)
        
        yield inputs, targets


# ============================================================================
# TEXT GENERATION
# ============================================================================

def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> int:
    """Sample next token with temperature, top-k, and nucleus sampling."""
    logits = logits / max(temperature, 1e-6)
    
    # Top-k
    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
    
    # Nucleus (top-p)
    probs = torch.softmax(logits_filtered, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumsum_probs > top_p
    sorted_indices_to_remove[0] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum()
    
    if probs.sum() == 0:
        return torch.argmax(logits).item()
    
    return torch.multinomial(probs, 1).item()


def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    seed_text: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9
) -> str:
    """Generate text from seed."""
    model.eval()
    context_length = CONFIG['context_length']
    
    tokens = tokenizer.encode(seed_text).ids[:context_length]
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            input_tokens = torch.tensor(
                [generated[-context_length:]],
                dtype=torch.long
            ).to(device)
            
            logits = model(input_tokens)[0, -1, :]
            next_token = sample_token(logits, temperature, top_k, top_p)
            
            generated.append(next_token)
            
            if next_token == tokenizer.token_to_id("[EOS]"):
                break
    
    return tokenizer.decode(generated)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model_background(session_id: str, dataset_text: str):
    """Background training task."""
    try:
        session = session_manager.get_session(session_id)
        session_manager.update_status(session_id, 'training', 0)
        
        # Train tokenizer
        logger.info(f"[{session_id}] Training tokenizer...")
        session_manager.update_status(session_id, 'training', 10)
        
        tokenizer = train_tokenizer(dataset_text, Path(session['tokenizer_path']))
        
        # Tokenize dataset
        logger.info(f"[{session_id}] Tokenizing dataset...")
        session_manager.update_status(session_id, 'training', 20)
        
        tokens = tokenizer.encode(dataset_text).ids
        if len(tokens) < CONFIG['context_length']:
            raise ValueError(f"Tokenized dataset too small: {len(tokens)}")
        
        # Create model
        logger.info(f"[{session_id}] Creating model...")
        model = TransformerLLMImproved(CONFIG).to(device)
        logger.info(f"[{session_id}] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training setup
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['num_epochs'],
            eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(CONFIG['num_epochs']):
            model.train()
            total_loss = 0
            num_batches = 0
            
            batches = prepare_batches(tokens, CONFIG['batch_size'], CONFIG['context_length'])
            
            for inputs, targets in batches:
                optimizer.zero_grad()
                
                logits = model(inputs)
                loss = criterion(
                    logits.view(-1, CONFIG['vocab_size']),
                    targets.view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            avg_loss = total_loss / max(num_batches, 1)
            progress = int(20 + (epoch + 1) / CONFIG['num_epochs'] * 70)
            
            logger.info(f"[{session_id}] Epoch {epoch + 1}/{CONFIG['num_epochs']}, Loss: {avg_loss:.4f}")
            session_manager.update_status(session_id, 'training', progress)
        
        # Save model
        logger.info(f"[{session_id}] Saving model...")
        session_manager.update_status(session_id, 'training', 95)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
        }, session['model_path'])
        
        logger.info(f"[{session_id}] Training complete!")
        session_manager.update_status(session_id, 'ready', 100)
    
    except Exception as e:
        logger.error(f"[{session_id}] Training failed: {e}")
        session_manager.update_status(session_id, 'error', error=str(e))


# ============================================================================
# API ENDPOINTS
# ============================================================================

class UploadRequest(BaseModel):
    session_id: str


class TrainRequest(BaseModel):
    session_id: str


class GenerateRequest(BaseModel):
    session_id: str
    seed: str = "Väinämöinen, vanha viisas"
    max_length: int = 200
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9


@app.post("/session/create")
def create_session():
    """Create new session."""
    try:
        session_id = session_manager.create_session()
        return {"session_id": session_id, "status": "ready"}
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload file to session."""
    try:
        session = session_manager.get_session(session_id)
        
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail=f"File type not allowed: {file.filename}")
        
        # Save uploaded file
        upload_path = UPLOAD_FOLDER / secure_filename(file.filename)
        with open(upload_path, 'wb') as f:
            contents = file.file.read(MAX_FILE_SIZE + 1)
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            f.write(contents)
        
        # Process file
        text = process_file(upload_path)
        
        # Append to dataset
        dataset_path = Path(session['dataset_path'])
        current_size = dataset_path.stat().st_size if dataset_path.exists() else 0
        
        if current_size + len(text) > MAX_TOTAL_DATASET_SIZE:
            raise HTTPException(status_code=413, detail="Total dataset size exceeded")
        
        with open(dataset_path, 'a', encoding='utf-8') as f:
            f.write(text + "\n")
        
        upload_path.unlink()  # Delete temporary file
        
        new_size = dataset_path.stat().st_size
        logger.info(f"[{session_id}] Uploaded {file.filename}: +{len(text)} chars, total: {new_size}")
        
        return {
            "message": f"File {file.filename} processed",
            "chars_extracted": len(text),
            "total_chars": new_size
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start training in background."""
    try:
        session = session_manager.get_session(request.session_id)
        dataset_path = Path(session['dataset_path'])
        
        if not dataset_path.exists():
            raise HTTPException(status_code=400, detail="No dataset uploaded")
        
        dataset_text = dataset_path.read_text(encoding='utf-8', errors='ignore')
        
        if len(dataset_text) < MIN_DATASET_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset too small: {len(dataset_text)} < {MIN_DATASET_SIZE}"
            )
        
        # Start background training
        background_tasks.add_task(train_model_background, request.session_id, dataset_text)
        
        return {"message": "Training started", "session_id": request.session_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Train request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
def generate_text(request: GenerateRequest):
    """Generate text from seed."""
    try:
        session = session_manager.get_session(request.session_id)
        
        if session['status'] != 'ready':
            raise HTTPException(status_code=400, detail=f"Model not ready: {session['status']}")
        
        model_path = Path(session['model_path'])
        tokenizer_path = Path(session['tokenizer_path'])
        
        if not model_path.exists() or not tokenizer_path.exists():
            raise HTTPException(status_code=400, detail="Model not trained")
        
        # Load model and tokenizer
        checkpoint = torch.load(model_path, map_location=device)
        model = TransformerLLMImproved(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        tokenizer = load_tokenizer(tokenizer_path)
        
        # Validate and clamp parameters
        max_length = validate_max_length(request.max_length)
        
        # Generate
        generated = generate(
            model,
            tokenizer,
            request.seed,
            max_length,
            request.temperature,
            request.top_k,
            request.top_p
        )
        
        logger.info(f"[{request.session_id}] Generated {len(generated)} chars")
        
        return {
            "generated": generated,
            "seed": request.seed,
            "length": len(generated)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{session_id}")
def get_status(session_id: str):
    """Get session status."""
    try:
        session = session_manager.get_session(session_id)
        return {
            "session_id": session_id,
            "status": session['status'],
            "progress": session['progress'],
            "error": session['error'],
            "created_at": session['created_at'].isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    """Reset session."""
    try:
        session = session_manager.get_session(session_id)
        
        # Clear files
        for key in ['dataset_path', 'model_path', 'tokenizer_path']:
            path = Path(session[key])
            if path.exists():
                path.unlink()
        
        session['status'] = 'idle'
        session['progress'] = 0
        session['error'] = None
        
        logger.info(f"[{session_id}] Session reset")
        
        return {"message": "Session reset"}
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available()
    }


if __name__ == "__main__":
    logger.info("Starting Finnish Culture LLM v2.0")
    logger.info(f"Config: {CONFIG}")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5000,
        log_level="info"
    )
