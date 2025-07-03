"""
A simple transformer-based language model implementation with tokenization,
training, and text generation capabilities. Designed for educational purposes
and suitable for small-scale NLP tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
import logging
import re
import os

# Configure logging for better debugging and error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Tokenizer:
    """A simple tokenizer for text preprocessing."""
    def __init__(self, vocab: Optional[dict] = None):
        try:
            if vocab is not None and not all(isinstance(k, str) and isinstance(v, int) for k, v in vocab.items()):
                raise ValueError("Vocabulary must map strings to integers")
            self.vocab = vocab if vocab else {'[PAD]': 0, '[UNK]': 1}
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.max_vocab_size = 5000
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise ValueError(f"Tokenizer initialization failed: {e}")

    def build_vocab(self, texts: List[str]) -> None:
        """Builds vocabulary from a list of texts."""
        try:
            if not texts or not all(isinstance(t, str) for t in texts):
                raise ValueError("Input texts must be a non-empty list of strings")
            word_counts = {}
            for text in texts:
                words = re.findall(r'\w+|[^\w\s]', text.lower())
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1

            # Sort by frequency and limit vocab size
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:self.max_vocab_size - 2]
            for i, (word, _) in enumerate(sorted_words, start=2):
                self.vocab[word] = i
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"Vocabulary built with {len(self.vocab)} tokens")
        except Exception as e:
            logger.error(f"Error building vocabulary: {e}")
            raise RuntimeError(f"Vocabulary building failed: {e}")

    def encode(self, text: str) -> List[int]:
        """Converts text to token IDs."""
        try:
            if not isinstance(text, str):
                raise ValueError("Input text must be a string")
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            if not words:
                logger.warning("Empty or invalid text provided for encoding")
                return []
            return [self.vocab.get(word, self.vocab['[UNK]']) for word in words]
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return []

    def decode(self, token_ids: List[int]) -> str:
        """Converts token IDs back to text."""
        try:
            if not token_ids or not all(isinstance(t, int) for t in token_ids):
                raise ValueError("Token IDs must be a non-empty list of integers")
            return ' '.join(self.inverse_vocab.get(id, '[UNK]') for id in token_ids)
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            return ""

class PositionalEncoding(nn.Module):
    """Adds positional encodings to input embeddings."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        try:
            if d_model <= 0 or max_len <= 0:
                raise ValueError("d_model and max_len must be positive")
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        except Exception as e:
            logger.error(f"Error initializing positional encoding: {e}")
            raise RuntimeError(f"Positional encoding initialization failed: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return x + self.pe[:, :x.size(1)]
        except Exception as e:
            logger.error(f"Error in positional encoding forward: {e}")
            raise RuntimeError(f"Positional encoding forward failed: {e}")

class SimpleTransformer(nn.Module):
    """A simplified transformer model for text generation."""
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super(SimpleTransformer, self).__init__()
        try:
            if vocab_size <= 0 or d_model <= 0 or nhead <= 0 or num_layers <= 0:
                raise ValueError("vocab_size, d_model, nhead, and num_layers must be positive")
            if d_model % nhead != 0:
                raise ValueError("d_model must be divisible by nhead")
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
            self.fc = nn.Linear(d_model, vocab_size)
            self.d_model = d_model
            logger.info(f"Initialized transformer with vocab_size={vocab_size}, d_model={d_model}")
        except Exception as e:
            logger.error(f"Failed to initialize transformer: {e}")
            raise RuntimeError(f"Transformer initialization failed: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.embedding(x) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            return self.fc(x)
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise RuntimeError(f"Forward pass failed: {e}")

class SimpleLLM:
    """A simple language model wrapper for training and generation."""
    def __init__(self, vocab_size: int):
        try:
            if vocab_size <= 0:
                raise ValueError("vocab_size must be positive")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SimpleTransformer(vocab_size).to(self.device)
            self.tokenizer = Tokenizer()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            logger.info(f"LLM initialized with device={self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}")

    def train(self, texts: List[str], epochs: int = 10, batch_size: int = 32):
        """Trains the model on a list of texts."""
        try:
            if not texts or not all(isinstance(t, str) for t in texts):
                raise ValueError("Input texts must be a non-empty list of strings")
            if epochs <= 0 or batch_size <= 0:
                raise ValueError("epochs and batch_size must be positive")
            self.tokenizer.build_vocab(texts)
            if len(self.tokenizer.vocab) > self.model.embedding.num_embeddings:
                raise ValueError(f"Vocabulary size ({len(self.tokenizer.vocab)}) exceeds model vocab_size ({self.model.embedding.num_embeddings})")
            dataset = [self.tokenizer.encode(text) for text in texts]
            dataset = [d for d in dataset if len(d) > 1]  # Filter sequences too short for training
            if not dataset:
                raise ValueError("No valid training data after tokenization")

            for epoch in range(epochs):
                total_loss = 0
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i + batch_size]
                    if not batch:
                        logger.warning("Empty batch encountered, skipping")
                        continue
                    max_len = max(len(seq) for seq in batch)
                    if max_len <= 1:
                        logger.warning("Batch contains only single-token sequences, skipping")
                    padded = [seq + [0] * (max_len - len(seq)) for seq in batch]
                    inputs = torch.tensor(padded, dtype=torch.long, device=self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs[:, :-1])
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), inputs[:, 1:].reshape(-1))
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                if total_loss > 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / max(1, len(dataset) // batch_size)}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}, No valid batches processed")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")

    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Generates text based on a prompt."""
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string")
            if max_length <= 0:
                raise ValueError("max_length must be positive")
            self.model.eval()
            tokens = self.tokenizer.encode(prompt)
            if not tokens:
                raise ValueError("Invalid prompt or tokenization failed")
                
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            generated = tokens.copy()

            with torch.no_grad():
                for _ in range(max_length - len(tokens)):
                    outputs = self.model(input_ids)
                    next_token = torch.argmax(outputs[:, -1, :], dim=-1)
                    generated.append(next_token.item())
                    input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)
            
            return self.tokenizer.decode(generated)
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Error in text generation: {e}"

    def save(self, path: str) -> None:
        """Saves the model and tokenizer to a file."""
        try:
            if not path:
                raise ValueError("Path must be a non-empty string")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer_vocab': self.tokenizer.vocab
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Model saving failed: {e}")

    def load(self, path: str) -> None:
        """Loads the model and tokenizer from a file."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file {path} not found")
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.tokenizer = Tokenizer(vocab=checkpoint['tokenizer_vocab'])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading faled: {e}")

if __name__ == "__main__":
    try:
        # Example usage
        llm = SimpleLLM(vocab_size=5000)
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating and powerful.",
            "Transformers are the backbone of modern NLP."
        ]
        llm.train(sample_texts, epochs=5)
        generated_text = llm.generate("The quick brown", max_length=20)
        print(f"Generated text: {generated_text}")
        llm.save("model.pt")
        llm.load("model.pt")
    except Exception as e:
        logger.error(f"Error in man execution: {e}")
```
