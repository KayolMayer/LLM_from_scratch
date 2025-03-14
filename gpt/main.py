"""
Created on Fri Mar 14 16:03:00 2025.

@author: kayol
"""

from torch import tensor, manual_seed
from tiktoken import get_encoding
from packages.transformers import gpt_model
from packages.text_generator import generate_text_simple

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

manual_seed(123)
model = gpt_model(GPT_CONFIG_124M)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

total_size_bytes = total_params * 4  # each parameter has 4 bytes = 32 bits
total_size_mb = total_size_bytes / (1024 * 1024)  # conversion to MB
print(f"Total size of the model: {total_size_mb:.2f} MB")


start_context = "Hello, I am"
tokenizer = get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()
out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6,
                           context_size=GPT_CONFIG_124M["context_length"])

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
