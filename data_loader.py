# data_loader.py

import os
import urllib.request
import torch
from config import BATCH_SIZE, BLOCK_SIZE, DEVICE

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"

class TextDataLoader:
    """
    Downloads, processes, and samples batches from the dataset.
    """
    def __init__(self):
        # Download and read the raw text from file or URL.
        self.raw_text = self._download_and_read()
        # Build the vocabulary by extracting unique characters.
        self.chars = sorted(list(set(self.raw_text)))
        self.vocab_size = len(self.chars)
        # Create mappings: character to index and index to character.
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}
        # Lambda functions for encoding and decoding strings.
        self.encode = lambda s: [self.char_to_idx[c] for c in s]
        self.decode = lambda lst: ''.join([self.idx_to_char[i] for i in lst])
        
        # Encode the entire dataset as a tensor of indices.
        self.data_tensor = torch.tensor(self.encode(self.raw_text), dtype=torch.long)
        
        # Split data: 90% training and 10% validation.
        split_index = int(0.9 * len(self.data_tensor))
        self.train_data = self.data_tensor[:split_index]
        self.val_data = self.data_tensor[split_index:]
    
    def _download_and_read(self):
        """
        Downloads the dataset if not available locally and returns its content.
        """
        if not os.path.exists(DATA_FILE):
            print("Downloading dataset...")
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        with open(DATA_FILE, 'r', encoding='utf-8') as file:
            return file.read()
    
    def get_batch(self, split: str):
        """
        Generates a batch of input and target sequences.
        
        Args:
            split (str): Specify 'train' or 'val' to choose the dataset split.
        
        Returns:
            tuple: (inputs, targets) as torch.Tensors on the configured DEVICE.
        """
        # Select the appropriate data split.
        data = self.train_data if split == 'train' else self.val_data
        # Randomly sample starting indices ensuring sequences are in bounds.
        indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        # Build input sequences of length BLOCK_SIZE.
        inputs = torch.stack([data[i:i + BLOCK_SIZE] for i in indices])
        # Build target sequences shifted by one token.
        targets = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in indices])
        return inputs.to(DEVICE), targets.to(DEVICE)
