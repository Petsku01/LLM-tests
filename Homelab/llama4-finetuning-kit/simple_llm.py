"""
Simple transformer-based language model with autoregressive text generation.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from typing import List, Optional, Dict, Tuple

class Tokenizer:
    """Basic word-level tokenizer with vocabulary management."""
    
    def __init__(self, max_vocab_size: int = 5000):
        # Initialize with special tokens
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.max_vocab_size = max_vocab_size

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from corpus, keeping most frequent tokens."""
        word_counts = {}
        
        for text in texts:
            # Now handles punctuation better
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequencies and limit vocabulary size
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_limit = self.max_vocab_size - len(self.vocab)
        sorted_words = sorted_words[:vocab_limit]
        
        # Add most used words to vocabulary
        for word, _ in sorted_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)} tokens")

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Convert text to token IDs with optional BOS/EOS tokens."""
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        if add_special:
            tokens = [self.vocab['<BOS>']] + tokens + [self.vocab['<EOS>']]
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Convert token IDs back to text."""
        tokens = []
        special = {'<PAD>', '<UNK>', '<BOS>', '<EOS>'}
        
        for id in token_ids:
            token = self.inverse_vocab.get(id, '<UNK>')
            if skip_special and token in special:
                continue
            tokens.append(token)
        
        # Join with better spacing
        text = ''
        for i, token in enumerate(tokens):
            if i == 0:
                text = token
            elif token in '.,!?;:':  # Punctuation - no space before
                text += token
            else:
                text += ' ' + token
        return text

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encodings for sequence positions."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term for sine/cosine frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerLM(nn.Module):
    """Transformer decoder for autoregressive language modeling."""
    
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 4, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        # Model dimensions
        self.d_model = d_model
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer decoder layers (autoregressive)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with Xavier uniform."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_proj.bias.data.zero_()
        self.output_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x: Input token IDs [batch_size, seq_len]
            memory: Optional encoder memory for encoder-decoder setup
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        seq_len = x.size(1)
        
        # Create causal mask (upper triangular)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), 
            diagonal=1
        ).bool()
        
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through transformer decoder
        if memory is None:
            # Use self-attention only (language modeling)
            memory = x
        
        x = self.transformer(x, memory, tgt_mask=causal_mask)
        
        # Project to vocabulary
        return self.output_proj(x)

class SimpleLLM:
    """Language model trainer and text generator."""
    
    def __init__(self, max_vocab_size: int = 5000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer(max_vocab_size)
        self.model = None
        print(f"Using device: {self.device}")

    def prepare_data(self, texts: List[str]) -> List[List[int]]:
        """Tokenize texts and prepare for training."""
        # Build vocabulary from corpus
        self.tokenizer.build_vocab(texts)
        
        # Tokenize all texts
        sequences = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special=True)
            if len(tokens) > 2:  # Need more than just BOS/EOS
                sequences.append(tokens)
        
        return sequences

    def train(self, texts: List[str], epochs: int = 10, batch_size: int = 8, 
              lr: float = 0.001, max_seq_len: int = 128):
        """
        Train the language model on provided texts.
        Args:
            texts: List of training texts
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            max_seq_len: Maximum sequence length (longer sequences are truncated)
        """
        # Prepare data
        print("Preparing training data...")
        sequences = self.prepare_data(texts)
        
        if not sequences:
            raise ValueError("No valid training sequences found")
        
        # Initialize model
        vocab_size = len(self.tokenizer.vocab)
        self.model = TransformerLM(vocab_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Training on {len(sequences)} sequences...")
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle sequences
            import random
            random.shuffle(sequences)
            
            # Process in batches
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i + batch_size]
                
                # Truncate to max length and pad
                batch_tokens = []
                for seq in batch_seqs:
                    # Truncate if necessary
                    if len(seq) > max_seq_len:
                        seq = seq[:max_seq_len]
                    batch_tokens.append(seq)
                
                # Pad sequences to same length
                max_len = max(len(seq) for seq in batch_tokens)
                padded = []
                for seq in batch_tokens:
                    pad_len = max_len - len(seq)
                    padded_seq = seq + [self.tokenizer.vocab['<PAD>']] * pad_len
                    padded.append(padded_seq)
                
                # Convert to tensors
                input_ids = torch.tensor(padded, device=self.device)
                
                # Shift for autoregressive training
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Compute loss (ignore padding)
                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.tokenizer.vocab['<PAD>']
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Print epoch statistics
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 1.0, 
                 top_k: int = 50) -> str:
        """
        Generate text from a prompt.
        Args:
            prompt: Starting text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top k tokens
        Returns:
            Generated text string
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        
        # Encode prompt (without special tokens for generation)
        tokens = self.tokenizer.encode(prompt, add_special=False)
        if not tokens:
            tokens = [self.tokenizer.vocab['<BOS>']]
        
        # Generate tokens
        generated = tokens.copy()
        
        for _ in range(max_length):
            # Prepare input
            input_tensor = torch.tensor([generated], device=self.device)
            
            # Get model predictions
            outputs = self.model(input_tensor)
            logits = outputs[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop at EOS token
            if next_token == self.tokenizer.vocab.get('<EOS>', -1):
                break
            
            generated.append(next_token)
        
        return self.tokenizer.decode(generated)

    def save(self, path: str):
        """Save model and tokenizer to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state': self.model.state_dict(),
            'vocab': self.tokenizer.vocab,
            'model_config': {
                'vocab_size': len(self.tokenizer.vocab),
                'd_model': self.model.d_model
            }
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and tokenizer from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore tokenizer
        self.tokenizer.vocab = checkpoint['vocab']
        self.tokenizer.inverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        
        # Restore model
        config = checkpoint['model_config']
        self.model = TransformerLM(config['vocab_size']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        
        print(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Sample training data - diverse text for better learning
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning transforms how we process information.",
        "Neural networks learn pattaerns from data.",
        "Transformers revolutionized natural language processing.",
        "Deep learning models can understand and generate text.",
        "Artificial intelligence is changing the world.",
        "Language modelis predict the next word in a sequence.",
        "Training neural networks requires lots of data.",
        "Python is a popular programming language for AI.",
        "Data science combines statistics and comfputer science.",
        "The future of technology is incredibly exciting.",
        "Computers can now understand human language.",
    ]
    
    # Initialize model with moderate vocabulary
    print("=" * 60)
    print("SIMPLE TRANSFORMER LANGUAGE MODEL DEMO")
    print("=" * 60)
    
    llm = SimpleLLM(max_vocab_size=1000)
    
    # Train the model
    print("\n Training model...")
    llm.train(
        texts, 
        epochs=30,        # More epochs for small dataset
        batch_size=4,     # Small batch for demo
        lr=0.001,         # Standard learning rate
        max_seq_len=50    # Reasonable sequence length
    )
    
    # Generate text with different settings
    print("\n Generating text with different temperatures:")
    print("-" * 60)
    
    test_prompts = [
        ("The quick", 0.5, "Conservative"),
        ("Machine learning", 0.8, "Balanced"),
        ("Neural networks", 1.2, "Creative"),
    ]
    
    for prompt, temp, style in test_prompts:
        generated = llm.generate(
            prompt, 
            max_length=20, 
            temperature=temp, 
            top_k=40
        )
        print(f"\n{style} (temp={temp}):")
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: '{generated}'")
    
    # Demonstrate saving and loading
    print("\n Saving model...")
    llm.save("demo_model.pt")
    
    print(" Loading model in new instance...")
    new_llm = SimpleLLM()
    new_llm.load("demo_model.pt")
    
    # Test loaded model
    test_generation = new_llm.generate(
        "Artificial intelligence", 
        max_length=15,
        temperature=0.7
    )
    print(f"\n Loaded model works!")
    print(f"   Generated: '{test_generation}'")
    
    print("\n" + "=" * 60)
    print("Demo complete! Check the usage guide at the top of the file")
    print("for more detailed examples and parameters.")
    print("=" * 60)
