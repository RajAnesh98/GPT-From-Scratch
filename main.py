# main.py

import torch
from config import DEVICE, LEARNING_RATE
from data_loader import TextDataLoader
from models.gpt_model import GPTLanguageModel
from trainer import Trainer

def main():
    # Set random seed for reproducibility.
    torch.manual_seed(1337)
    
    # Initialize the data loader and build the dataset.
    data_loader = TextDataLoader()
    print(f"Dataset loaded with {len(data_loader.raw_text)} characters; vocabulary size: {data_loader.vocab_size}")
    
    # Instantiate the GPT model and move it to the configured device.
    model = GPTLanguageModel(data_loader.vocab_size).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total_params / 1e6:.2f}M")
    
    # Create the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize and run the trainer.
    trainer = Trainer(model, optimizer, data_loader)
    trainer.train()
    
    # Generate text from the trained model.
    init_context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_sequence = model.generate(init_context, max_new_tokens=500)
    output_text = data_loader.decode(generated_sequence[0].tolist())
    print("Generated text:")
    print(output_text)

if __name__ == '__main__':
    main()