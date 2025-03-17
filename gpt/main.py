"""
Created on Fri Mar 14 16:03:00 2025.

@author: kayol
"""

from os import sep
from torch import manual_seed, device, cuda, no_grad
from tiktoken import get_encoding
from packages.transformers import gpt_model
from packages.dataloaders import create_dataloader_v1
from packages.loss_functions import calc_loss_loader

# Getting device to run the model.
device = device("cuda" if cuda.is_available() else "cpu")
print("The model is running in the", device)

# GPT2 model
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

# My mini GPT2 model
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

# Read the input text to train the LLM model.
with open("datasets"+sep+"the-verdict.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

# Import the GPT-2 tokenizer with BPE.
tokenizer = get_encoding("gpt2")

# Get the number of characters and tokens in the input text.
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# Split dataset into Train and validation
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# Set the seed of random numbers of PyTorch.
manual_seed(123)

# Create the train dataloader to train the GPT model.
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

# Create the validation dataloader to validate the GPT model.
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# Sanity check.
if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

# Check if train and validation loaders were loaded correctly.
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# Initialize the GPT model
model = gpt_model(GPT_CONFIG_124M)

# Get the total number of parameters of the GPT model.
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# Get the memory used by the GPT model.
total_size_bytes = total_params * 4  # each parameter has 4 bytes = 32 bits
total_size_mb = total_size_bytes / (1024 * 1024)  # conversion to MB
print(f"Total size of the model: {total_size_mb:.2f} MB")

# Pass model to device in order to run it.
model.to(device)

# For reproducibility due to the shuffling in the data loader.
manual_seed(123)

# Compute the loss function
with no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)
