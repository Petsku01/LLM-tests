```python
"""
A theory Markov Chain-based language model for character-level text generation.
Designed for educational purposes with robust error handling and clear documentation.
"""

import random
from typing import List, Dict, Optional
import logging
import pickle
import os

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarkovTokenizer:
    """A character-level tokenizer for Markov Chain model."""
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        try:
            if vocab is not None and not all(isinstance(k, str) and len(k) == 1 and isinstance(v, int) for k, v in vocab.items()):
                raise ValueError("Vocabulary must map single characters to integers")
            self.vocab = vocab if vocab else {'<PAD>': 0, '<UNK>': 1}
            # Ensure <PAD> and <UNK> are always present
            if '<PAD>' not in self.vocab:
                self.vocab['<PAD>'] = len(self.vocab)
            if '<UNK>' not in self.vocab:
                self.vocab['<UNK>'] = len(self.vocab)
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.max_vocab_size = 100  # Suitable for character-level vocab
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise ValueError(f"Tokenizer initialization failed: {e}")

    def build_vocab(self, texts: List[str]) -> None:
        """Builds a character-level vocabulary from a list of texts."""
        try:
            if not texts or not all(isinstance(t, str) for t in texts):
                raise ValueError("Input texts must be a non-empty list of strings")
            chars = set()
            for text in texts:
                chars.update(text)
            if len(chars) > self.max_vocab_size - 2:
                logger.warning(f"Found {len(chars)} unique characters, limiting to {self.max_vocab_size - 2} due to max_vocab_size")
            sorted_chars = sorted(list(chars))[:self.max_vocab_size - 2]
            for i, char in enumerate(sorted_chars, start=len(self.vocab)):
                self.vocab[char] = i
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"Vocabulary built with {len(self.vocab)} characters")
        except Exception as e:
            logger.error(f"Error building vocabulary: {e}")
            raise RuntimeError(f"Vocabulary building failed: {e}")

    def encode(self, text: str) -> List[int]:
        """Converts text to a list of character token IDs."""
        try:
            if not isinstance(text, str):
                raise ValueError("Input text must be a string")
            if not text:
                logger.warning("Empty text provided for encoding")
                return []
            return [self.vocab.get(char, self.vocab['<UNK>']) for char in text]
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return []

    def decode(self, token_ids: List[int]) -> str:
        """Converts token IDs back to text."""
        try:
            if not token_ids or not all(isinstance(t, int) for t in token_ids):
                raise ValueError("Token IDs must be a non-empty list of integers")
            return ''.join(self.inverse_vocab.get(id, '<UNK>') for id in token_ids)
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            return ""

class MarkovChain:
    """A Markov Chain model for character-level text generation."""
    def __init__(self, order: int = 2, random_seed: Optional[int] = None):
        try:
            if order <= 0:
                raise ValueError("Order must be a positive integer")
            self.order = order
            self.transitions: Dict[str, Dict[str, int]] = {}
            self.tokenizer = MarkovTokenizer()
            if random_seed is not None:
                random.seed(random_seed)
            logger.info(f"Initialized Markov Chain with order={order}, random_seed={random_seed}")
        except Exception as e:
            logger.error(f"Failed to initialize Markov Chain: {e}")
            raise RuntimeError(f"Markov Chain initialization failed: {e}")

    def train(self, texts: List[str]) -> None:
        """Trains the Markov Chain on a list of texts."""
        try:
            if not texts or not all(isinstance(t, str) for t in texts):
                raise ValueError("Input texts must be a non-empty list of strings")
            self.tokenizer.build_vocab(texts)
            for text in texts:
                if len(text) < self.order + 1:
                    logger.warning(f"Skipping text too short for order {self.order}: {text[:20]}...")
                    continue
                for i in range(len(text) - self.order):
                    context = text[i:i + self.order]
                    next_char = text[i + self.order]
                    if context not in self.transitions:
                        self.transitions[context] = {}
                    self.transitions[context][next_char] = self.transitions[context].get(next_char, 0) + 1
            if not self.transitions:
                raise ValueError("No valid transitions learned from texts")
            # Check for empty transition dictionaries
            empty_contexts = [ctx for ctx, trans in self.transitions.items() if not trans]
            if empty_contexts:
                logger.warning(f"Found {len(empty_contexts)} contexts with no transitions")
            logger.info(f"Trained Markov Chain with {len(self.transitions)} contexts")
        except Exception as e:
            logger.error(f"Training failed for texts: {e}")
            raise RuntimeError(f"Training failed: {e}")

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generates text based on a prompt using the Markov Chain."""
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string")
            if max_length <= 0:
                raise ValueError("max_length must be positive")
            if not all(char in self.tokenizer.vocab for char in prompt):
                logger.warning("Prompt contains characters not in vocabulary, may affect generation")
            if len(prompt) < self.order:
                logger.warning(f"Prompt too short for order {self.order}, using random context")
                prompt = random.choice(list(self.transitions.keys()))[:self.order]

            generated = [c for c in prompt]  # Use list for efficient concatenation
            for _ in range(max_length - len(prompt)):
                context = ''.join(generated[-self.order:])
                if context not in self.transitions or not self.transitions[context]:
                    logger.debug(f"No transitions for context '{context}', using random context")
                    context = random.choice(list(self.transitions.keys()))
                next_chars = self.transitions[context]
                total = sum(next_chars.values())
                if total == 0:
                    logger.warning(f"Zero total count for context '{context}', using random context")
                    context = random.choice(list(self.transitions.keys()))
                    next_chars = self.transitions[context]
                    total = sum(next_chars.values())
                probs = {char: count / total for char, count in next_chars.items()}
                next_char = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
                generated.append(next_char)
            return ''.join(generated)
        except Exception as e:
            logger.error(f"Text generation failed for prompt '{prompt[:20]}...': {e}")
            return f"Error in text generation: {e}"

    def save(self, path: str) -> None:
        """Saves the Markov Chain and tokenizer to a file."""
        try:
            if not path:
                raise ValueError("Path must be a non-empty string")
            os.makedirs(os.path.dirname(path), exist_ok=True)  # Create directory if needed
            with open(path, 'wb') as f:
                pickle.dump({
                    'transitions': self.transitions,
                    'order': self.order,
                    'tokenizer_vocab': self.tokenizer.vocab
                }, f)
            logger.info(f"Model saved to {path}")
        except (OSError, pickle.PicklingError) as e:
            logger.error(f"Failed to save model to {path}: {e}")
            raise RuntimeError(f"Model saving failed: {e}")

    def load(self, path: str) -> None:
        """Loads the Markov Chain and tokenizer from a file."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file {path} not found")
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.transitions = checkpoint['transitions']
            self.order = checkpoint['order']
            self.tokenizer = MarkovTokenizer(vocab=checkpoint['tokenizer_vocab'])
            logger.info(f"Model loaded from {path}")
        except (OSError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

if __name__ == "__main__":
    try:
        # Example usage
        markov = MarkovChain(order=3, random_seed=42)
        sample_texts = [
            "The moon glows softly in the night sky.",
            "Programming is an art and a science.",
            "Data drives the future of innovation."
        ]
        markov.train(sample_texts)
        generated_text = markov.generate("The moon", max_length=50)
        print(f"Generated text: {generated_text}")
        markov.save("markov_model.pkl")
        markov.load("markov_model.pkl")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
```
