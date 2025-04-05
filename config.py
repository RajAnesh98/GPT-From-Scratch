# config.py
import torch

# --- Hyperparameters ---
BATCH_SIZE    = 32         # Number of sequences per batch
BLOCK_SIZE    = 32         # Maximum context length for predictions
MAX_ITER      = 5000       # Total training iterations
EVAL_INTERVAL = 100        # Frequency (in iterations) to evaluate the model
EVAL_ITERS    = 200        # Number of batches used for loss evaluation
LEARNING_RATE = 1e-3       # Learning rate for the optimizer
EMBED_DIM     = 64         # Dimensionality of token embeddings
NUM_HEADS     = 4          # Number of attention heads in the transformer
NUM_LAYERS    = 4          # Number of transformer blocks (layers)
DROPOUT_RATE  = 0.0        # Dropout probability for regularization

# --- Device Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
