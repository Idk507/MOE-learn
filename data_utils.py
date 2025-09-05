import torch
import urllib.request
from typing import Tuple, List, Dict

class DataLoader:
    """Handles data loading and processing for character-level language modeling."""
    
    def __init__(self, config):
        self.config = config
        self.text = self.load_tinyshakespeare()
        self.chars = sorted(list(set(self.text)))
        
        if config.vocab_limit:
            self.chars = self.chars[:config.vocab_limit]
            
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)
        
        # Encode the full text
        self.data = self.encode(self.text)
        
        # Split into train/val
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        
    def load_tinyshakespeare(self) -> str:
        """Load the tiny shakespeare dataset."""
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            txt = urllib.request.urlopen(url, timeout=10).read().decode("utf-8")
        except Exception:
            # Fallback tiny corpus
            txt = (
                "To be, or not to be, that is the question:\n"
                "Whether 'tis nobler in the mind to suffer\n"
                "The slings and arrows of outrageous fortune,\n"
                "Or to take arms against a sea of troubles\n"
                "And by opposing end them.\n"
            ) * 200
        return txt
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text string to tensor of integers."""
        return torch.tensor([self.stoi[c] for c in text if c in self.stoi], dtype=torch.long)
    
    def decode(self, tensor: torch.Tensor) -> str:
        """Decode tensor of integers to text string."""
        return ''.join([self.itos[int(i)] for i in tensor])
    
    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of data for training or validation."""
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.config.block_size - 1, (self.config.batch_size,))
        x = torch.stack([data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.config.block_size] for i in ix])
        return x.to(self.config.device), y.to(self.config.device)
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary information."""
        return {
            'vocab_size': self.vocab_size,
            'chars': self.chars,
            'stoi': self.stoi,
            'itos': self.itos
        }
