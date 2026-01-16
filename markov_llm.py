"""
A simple Markov Chain-based character-level text generator.

"""

import random
from typing import Dict, List
from collections import defaultdict, Counter


class MarkovChain:
    """A simple Markov Chain model for character-level text generation."""
    
    def __init__(self, order: int = 2):
        """Initialize with specified order (context length)."""
        if order < 1:
            raise ValueError("Order must be at least 1")
        
        self.order = order
        self.transitions = defaultdict(Counter)
    
    def train(self, texts: List[str]) -> None:
        """Train the model on a list of texts."""
        if not texts:
            raise ValueError("Training texts cannot be empty")
        
        for text in texts:
            # Skip texts that are too short
            if len(text) <= self.order:
                continue
                
            # Build transitions
            for i in range(len(text) - self.order):
                context = text[i:i + self.order]
                next_char = text[i + self.order]
                self.transitions[context][next_char] += 1
        
        if not self.transitions:
            raise ValueError("No transitions learned - texts may be too short")
    
    def generate(self, prompt: str = "", max_length: int = 100) -> str:
        """Generate text starting with the given prompt."""
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        
        # If no prompt or prompt too short, start with random context
        if len(prompt) < self.order:
            if not self.transitions:
                return ""
            prompt = random.choice(list(self.transitions.keys()))
        
        result = list(prompt)
        
        for _ in range(max_length - len(prompt)):
            # Get current context
            context = ''.join(result[-self.order:])
            
            # Get possible next characters
            if context not in self.transitions:
                # Fall back to random context if current one has no transitions
                context = random.choice(list(self.transitions.keys()))
            
            next_chars = self.transitions[context]
            if not next_chars:
                break
            
            # Choose next character based on frequency
            chars = list(next_chars.keys())
            weights = list(next_chars.values())
            next_char = random.choices(chars, weights=weights)[0]
            
            result.append(next_char)
        
        return ''.join(result)
    
    def get_stats(self) -> Dict:
        """Return basic statistics about the trained model."""
        if not self.transitions:
            return {"contexts": 0, "total_transitions": 0}
        
        total_transitions = sum(sum(counter.values()) for counter in self.transitions.values())
        return {
            "contexts": len(self.transitions),
            "total_transitions": total_transitions,
            "avg_transitions_per_context": total_transitions / len(self.transitions)
        }


def main():
    """Example usage of the Markov Chain."""
    # Create and train model
    markov = MarkovChain(order=3)
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The moon glows softly in the night sky.",
        "Programming is both an art and a science.",
        "Data drives the future of innovation and discovery."
    ]
    
    markov.train(sample_texts)
    
    # Generate some text
    print("Model Statistics:")
    stats = markov.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nGenerated Text:")
    for i in range(3):
        text = markov.generate("The", max_length=60)
        print(f"  {i+1}: {text}")


if __name__ == "__main__":
    main()
