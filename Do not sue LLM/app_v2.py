"""
Improved Finnish Culture LLM - Production-Ready Version
Fixes: Proper architecture, security, concurrency, background tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import PyPDF2
from PIL import Image
import pytesseract
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List

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
SESSIONS_FOLDER = BASE_DIR / 'sessions'
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
        session = self.get_session(session_id)
        session.update({'status': status, 'progress': progress, 'error': error})


session_manager = SessionManager(SESSIONS_FOLDER)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(filepath: Path) -> str:
    """Extract text from file."""
    suffix = filepath.suffix.lower()
    
    try:
        if suffix == '.txt':
            return filepath.read_text(encoding='utf-8', errors='ignore')
        
        if suffix == '.pdf':
            with open(filepath, 'rb') as f:
                text = ''.join((page.extract_text() or '') for page in PyPDF2.PdfReader(f).pages)
                return text
        
        if suffix in {'.jpg', '.jpeg', '.png'}:
            return pytesseract.image_to_string(Image.open(filepath), lang='fin')
    
    except Exception as e:
        logger.error(f"File processing failed {filepath}: {e}")
        raise


def validate_max_length(length: int) -> int:
    if length < 1 or length > MAX_GENERATION_LENGTH:
        raise ValueError(f"max_length must be 1-{MAX_GENERATION_LENGTH}")
    return length


# ============================================================================
# TOKENIZER MANAGEMENT
# ============================================================================

def train_tokenizer(text: str, tokenizer_path: Path) -> Tokenizer:
    if len(text) < 100:
        raise ValueError("Dataset too short for tokenizer")
    
    tokenizer = Tokenizer(BPE(dropout=0.1))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=CONFIG['vocab_size'],
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        min_frequency=2
    )
    
    tokenizer.train_from_iterator([text], trainer, length=len(text))
    tokenizer.save(str(tokenizer_path))
    logger.info(f"Tokenizer ready: vocab_size={tokenizer.get_vocab_size()}")
    return tokenizer


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))


def prepare_batches(tokens: List[int], batch_size: int, context_length: int):
    """Yield shuffled batches of token sequences."""
    sequences = [
        (tokens[i:i + context_length], tokens[i + 1:i + context_length + 1])
        for i in range(len(tokens) - context_length)
    ]
    np.random.shuffle(sequences)
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        inputs = torch.tensor([seq for seq, _ in batch_seqs], dtype=torch.long).to(device)
        targets = torch.tensor([tgt for _, tgt in batch_seqs], dtype=torch.long).to(device)
        yield inputs, targets


# ============================================================================
# TEXT GENERATION
# ============================================================================

def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> int:
    """Sample token with temperature, top-k, and nucleus sampling."""
    logits = logits / max(temperature, 1e-6)
    
    # Top-k filtering
    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
    
    # Nucleus (top-p) sampling
    probs = torch.softmax(logits_filtered, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumsum > top_p
    sorted_indices_to_remove[0] = False
    probs[sorted_indices[sorted_indices_to_remove]] = 0.0
    probs = probs / (probs.sum() + 1e-10)
    
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

def train_model_background(session_id: str, dataset_text: str):
    """Background training with progress updates."""
    try:
        session = session_manager.get_session(session_id)
        
        # Train tokenizer
        session_manager.update_status(session_id, 'training', 10)
        tokenizer = train_tokenizer(dataset_text, Path(session['tokenizer_path']))
        
        # Tokenize
        session_manager.update_status(session_id, 'training', 20)
        tokens = tokenizer.encode(dataset_text).ids
        if len(tokens) < CONFIG['context_length']:
            raise ValueError(f"Dataset too small: {len(tokens)} tokens")
        
        # Create model
        model = TransformerLLMImproved(CONFIG).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[{session_id}] Model: {num_params:,} params")
        
        # Setup training
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        for epoch in range(CONFIG['num_epochs']):
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            for inputs, targets in prepare_batches(tokens, CONFIG['batch_size'], CONFIG['context_length']):
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits.view(-1, CONFIG['vocab_size']), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            progress = int(20 + (epoch + 1) / CONFIG['num_epochs'] * 75)
            logger.info(f"[{session_id}] Epoch {epoch + 1}/{CONFIG['num_epochs']} loss={avg_loss:.4f}")
            session_manager.update_status(session_id, 'training', progress)
        
        # Save
        torch.save({'model_state_dict': model.state_dict(), 'config': CONFIG}, session['model_path'])
        logger.info(f"[{session_id}] Training complete")
        session_manager.update_status(session_id, 'ready', 100)
    
    except Exception as e:
        logger.error(f"[{session_id}] Training failed: {e}")
        session_manager.update_status(session_id, 'error', error=str(e))


# ============================================================================
# API MODELS & ENDPOINTS
# ============================================================================

class SessionIdRequest(BaseModel):
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
    try:
        session_id = session_manager.create_session()
        return {"session_id": session_id, "status": "ready"}
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
def upload_file(session_id: str, file: UploadFile = File(...)):
    try:
        session_manager.get_session(session_id)
        
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="File type not allowed")
        
        # Save temporarily
        upload_path = SESSIONS_FOLDER / f".tmp_{file.filename}"
        contents = file.file.read(MAX_FILE_SIZE + 1)
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        upload_path.write_bytes(contents)
        
        # Process
        text = process_file(upload_path)
        upload_path.unlink()
        
        # Append to dataset
        dataset_path = Path(session['dataset_path'])
        current_size = dataset_path.stat().st_size if dataset_path.exists() else 0
        if current_size + len(text) > MAX_TOTAL_DATASET_SIZE:
            raise HTTPException(status_code=413, detail="Dataset size limit exceeded")
        
        dataset_path.write_text(dataset_path.read_text() + text + "\n", encoding='utf-8')
        new_size = dataset_path.stat().st_size
        logger.info(f"[{session_id}] Upload: {file.filename} (+{len(text)} chars)")
        return {"message": "File processed", "chars_extracted": len(text), "total_chars": new_size}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train(request: SessionIdRequest, background_tasks: BackgroundTasks):
    try:
        session = session_manager.get_session(request.session_id)
        dataset_path = Path(session['dataset_path'])
        
        if not dataset_path.exists():
            raise HTTPException(status_code=400, detail="No dataset uploaded")
        
        dataset_text = dataset_path.read_text(encoding='utf-8', errors='ignore')
        if len(dataset_text) < MIN_DATASET_SIZE:
            raise HTTPException(status_code=400, detail="Dataset too small")
        
        background_tasks.add_task(train_model_background, request.session_id, dataset_text)
        return {"message": "Training started", "session_id": request.session_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Train failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
def generate_text(request: GenerateRequest):
    try:
        session = session_manager.get_session(request.session_id)
        if session['status'] != 'ready':
            raise HTTPException(status_code=400, detail=f"Model not ready: {session['status']}")
        
        model_path = Path(session['model_path'])
        tokenizer_path = Path(session['tokenizer_path'])
        
        checkpoint = torch.load(model_path, map_location=device)
        model = TransformerLLMImproved(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = load_tokenizer(tokenizer_path)
        
        max_length = validate_max_length(request.max_length)
        generated = generate(model, tokenizer, request.seed, max_length, request.temperature, request.top_k, request.top_p)
        
        logger.info(f"[{request.session_id}] Generated {len(generated)} chars")
        return {"generated": generated, "seed": request.seed, "length": len(generated)}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{session_id}")
def get_status(session_id: str):
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
    try:
        session = session_manager.get_session(session_id)
        for key in ['dataset_path', 'model_path', 'tokenizer_path']:
            Path(session[key]).unlink(missing_ok=True)
        session.update({'status': 'idle', 'progress': 0, 'error': None})
        logger.info(f"[{session_id}] Session reset")
        return {"message": "Session reset"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "device": str(device), "cuda_available": torch.cuda.is_available()}


if __name__ == "__main__":
    logger.info("Starting Finnish Culture LLM v2.0")
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
