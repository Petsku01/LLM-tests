"""
A simple transformer-based language model implementation with tokenization,
training, and text generation capabilities. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from typing import List, Optional

class Tokenizer:
    """Simple tokenizer for text preprocessing."""
    
    def __init__(self, vocab: Optional[dict] = None):
        self.vocab = vocab if vocab else {'<PAD>': 0, '<UNK>': 1}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.max_vocab_size = 5000

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        word_counts = {}
        
        for text in texts:
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Keep most frequent words, reserve space for special tokens
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_vocab_size - len(self.vocab)]
        
        # Add new words to vocabulary
        next_id = len(self.vocab)
        for word, _ in sorted_words:
            if word not in self.vocab:
                self.vocab[word] = next_id
                next_id += 1
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return ' '.join(self.inverse_vocab.get(id, '<UNK>') for id in token_ids)

class PositionalEncoding(nn.Module):
    """Add positional encodings to embeddings."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class SimpleTransformer(nn.Module):
    """Simplified transformer model for text generation."""
    
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Scale embeddings and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, mask)
        
        return self.output_layer(x)

class SimpleLLM:
    """Simple language model wrapper for training and generation."""
    
    def __init__(self, vocab_size: int = 5000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer()
        self.model = None
        self.vocab_size = vocab_size

    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(self.device)

    def train(self, texts: List[str], epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
        """Train the model on texts."""
        print("Building vocabulary...")
        self.tokenizer.build_vocab(texts)
        
        # Initialize model with actual vocab size
        actual_vocab_size = len(self.tokenizer.vocab)
        self.model = SimpleTransformer(actual_vocab_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Tokenize all texts
        print("Tokenizing texts...")
        dataset = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 1:  # Need at least 2 tokens for training
                dataset.append(tokens)
        
        if not dataset:
            raise ValueError("No valid training sequences found")
        
        print(f"Training on {len(dataset)} sequences for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            
            # Simple batching - group sequences of similar length
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                if not batch:
                    continue
                
                # Pad sequences to same length
                max_len = max(len(seq) for seq in batch)
                padded_sequences = []
                
                for seq in batch:
                    padded = seq + [self.tokenizer.vocab['<PAD>']] * (max_len - len(seq))
                    padded_sequences.append(padded)
                
                # Convert to tensors
                input_ids = torch.tensor(padded_sequences, dtype=torch.long, device=self.device)
                
                # Create inputs and targets for language modeling
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                
                if inputs.size(1) == 0:  # Skip if no valid input
                    continue
                
                # Create causal mask
                mask = self._create_causal_mask(inputs.size(1))
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs, mask)
                
                # Calculate loss (ignore padding tokens)
                loss = F.cross_entropy(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    targets.contiguous().view(-1),
                    ignore_index=self.tokenizer.vocab['<PAD>']
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            if batches > 0:
                avg_loss = total_loss / batches
                print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """Generate text based on prompt."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [self.tokenizer.vocab['<UNK>']]
        
        generated = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length - len(tokens)):
                # Prepare input
                input_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)
                mask = self._create_causal_mask(len(generated))
                
                # Get model output
                outputs = self.model(input_tensor, mask)
                logits = outputs[0, -1, :] / temperature  # Scale by temperature
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                # Stop if we generate padding token
                if next_token == self.tokenizer.vocab['<PAD>']:
                    break
        
        return self.tokenizer.decode(generated)

    def save(self, path: str) -> None:
        """Save model and tokenizer."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_vocab': self.tokenizer.vocab,
            'vocab_size': len(self.tokenizer.vocab)
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model and tokenizer."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore tokenizer
        self.tokenizer.vocab = checkpoint['tokenizer_vocab']
        self.tokenizer.inverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        
        # Restore model
        vocab_size = checkpoint['vocab_size']
        self.model = SimpleTransformer(vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Create and train a simple model
    llm = SimpleLLM()
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating and powerful.",
        "Transformers are the backbone of modern NLP.",
        "Deep learning models can generate creative text.",
        "Natural language processing enables many applications."
    ]
    
    # Train the model
    llm.train(sample_texts, epochs=20, batch_size=2)
    
    # Generate text
    generated = llm.generate("The quick brown", max_length=15, temperature=0.8)
    print(f"\nGenerated text: {generated}")
    
    # Save and reload
    llm.save("simple_model.pt")
    
    # Test loading
    new_llm = SimpleLLM()
    new_llm.load("simple_model.pt")
    generated2 = new_llm.generate("Machine learning", max_length=10)
    print(f"Generated after reload: {generated2}")
