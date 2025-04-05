# trainer.py

import torch
from config import MAX_ITER, EVAL_INTERVAL, EVAL_ITERS, DEVICE
from data_loader import TextDataLoader

class Trainer:
    """
    Manages training and evaluation of the model.
    """
    def __init__(self, model, optimizer, data_loader: TextDataLoader):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
    
    @torch.no_grad()
    def evaluate(self):
        """
        Evaluates the model's loss over several batches for both training and validation sets.
        
        Returns:
            dict: Average losses for 'train' and 'val' phases.
        """
        loss_summary = {}
        self.model.eval()
        for phase in ['train', 'val']:
            phase_losses = torch.zeros(EVAL_ITERS, device=DEVICE)
            for i in range(EVAL_ITERS):
                x_batch, y_batch = self.data_loader.get_batch(phase)
                _, loss = self.model(x_batch, y_batch)
                phase_losses[i] = loss.item()
            loss_summary[phase] = phase_losses.mean().item()
        self.model.train()
        return loss_summary
    
    def train(self):
        """
        Executes the main training loop.
        """
        for iter_num in range(MAX_ITER):
            if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITER - 1:
                current_losses = self.evaluate()
                print(f"Iteration {iter_num}: train loss {current_losses['train']:.4f}, val loss {current_losses['val']:.4f}")
            
            # Sample a training batch.
            x_batch, y_batch = self.data_loader.get_batch('train')
            _, loss = self.model(x_batch, y_batch)
            # Backpropagation and parameter update.
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        print("Training complete.")
