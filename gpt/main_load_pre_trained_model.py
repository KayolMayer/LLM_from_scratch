"""
Created on Fri Mar 14 16:03:00 2025.

@author: kayol
"""

from os import sep
from torch import load, optim
from packages.transformers import gpt_model

# GPT2 model
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,   # Context length (original: 1024)
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

# Load parameters from pre-trained model
checkpoint = load("models" + sep + "model.pth")

# Create the model
model = gpt_model(GPT_CONFIG_124M)

# Read the pre-trained model
model.load_state_dict(checkpoint["model_state_dict"])

# Create the optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

# Load the optimizer states from training
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Evaluate the model to show its architecture
model.eval()

# Set the model to the training mode (it does not execute training).
model.train()
