"""
Created on Tue Mar 18 22:31:42 2025.

@author: kayol
"""

from os import sep
from packages.gpt_download import load_gpt2, load_weights_into_gpt
from packages.transformers import gpt_model
from torch import device, cuda, manual_seed
from packages.text_generator import generate_text, text_to_token_ids, \
    token_ids_to_text
from tiktoken import get_encoding

# Getting device to run the model.
device = device("cuda" if cuda.is_available() else "cpu")
print("The model is running in the", device)

# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

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

# Folder with the OpenAI pre-trained GPT-2 model
model_dir = "models" + sep + "gpt2" + sep + "124M"

# Load the GPT model
settings, params = load_gpt2(model_dir)

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
new_config = GPT_CONFIG_124M.copy()
new_config.update(model_configs[model_name])
new_config.update({"context_length": 1024, "qkv_bias": True})
gpt = gpt_model(new_config)
gpt.eval()

# Load GPT parameters into our model
load_weights_into_gpt(gpt, params)
gpt.to(device)

# Import the GPT-2 tokenizer with BPE.
tokenizer = get_encoding("gpt2")

# GEnerate the seed of random numbers
manual_seed(123)

token_ids = generate_text(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=100,
    context_size=new_config["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
