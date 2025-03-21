"""
Created on Tue Mar 18 22:31:42 2025.

@author: kayol
"""

from os import sep
from time import time
from torch import save as save_torch
from torch import load as load_torch
from torch import device, cuda, manual_seed, nn, no_grad, optim, linspace
from torch.utils.data import DataLoader
from tiktoken import get_encoding
from packages.gpt_download import load_gpt2, load_weights_into_gpt
from packages.transformers import gpt_model
from packages.dataloaders import dataset_spam
from packages.metrics import calc_accuracy_loader
from packages.loss_functions import calc_loss_loader_spam
from packages.training import train_classifier_spam, plot_values
from packages.text_generator import text_to_token_ids, token_ids_to_text, \
    generate_text_simple, classify_review

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

# Import the GPT-2 tokenizer with BPE.
tokenizer = get_encoding("gpt2")

# GEnerate the seed of random numbers
manual_seed(123)

# Folder with the spam datasets
dataset_dir = "datasets" + sep

# Read the datasets of training, validation, and test
train_dataset = dataset_spam(csv_file=dataset_dir + "train_spam.csv",
                             max_length=None, tokenizer=tokenizer)

val_dataset = dataset_spam(csv_file=dataset_dir + "validation_spam.csv",
                           max_length=None, tokenizer=tokenizer)

test_dataset = dataset_spam(csv_file=dataset_dir + "test_spam.csv",
                            max_length=None, tokenizer=tokenizer)

print(train_dataset.max_length)
print(val_dataset.max_length)
print(test_dataset.max_length)

# Pass the dataset to dataloaders
manual_seed(123)

num_workers = 0
batch_size = 8

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         drop_last=False)

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

# Load GPT parameters into our model
load_weights_into_gpt(gpt, params)
gpt.eval()

# Verify if the maximum length in training in smaller than the context size
assert train_dataset.max_length <= new_config["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {new_config['context_length']}. Reinitialize data sets with "
    f"`max_length={new_config['context_length']}`"
)

# Test if the model was loaded correctly
text_1 = "Every effort moves you"

token_ids = generate_text_simple(
    model=gpt,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=new_config["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))

# Verify if the model can classify spams
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)

token_ids = generate_text_simple(
    model=gpt,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=new_config["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))

# In order to finetune the model, first we freeze all the parameters
for param in gpt.parameters():
    gpt.requires_grad = False

# Then, we replace the output layer (model.out_head), which originally maps the
# layer inputs to 50,257 dimensions (the size of the vocabulary):
manual_seed(123)

num_classes = 2
gpt.out_head = nn.Linear(in_features=new_config["emb_dim"],
                         out_features=num_classes)

# Additionally, we configure the last transformer block and the final LayerNorm
# module, which connects this block to the output layer, to be trainable
for param in gpt.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in gpt.final_norm.parameters():
    param.requires_grad = True

# Compute the accuracy of the modified model before finetuning
manual_seed(123)

# Move model to device
gpt.to(device)

train_accuracy = calc_accuracy_loader(train_loader, gpt, device,
                                      num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, gpt, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, gpt, device, num_batches=10)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Disable gradient tracking for efficiency because we are not training, yet
with no_grad():
    train_loss = calc_loss_loader_spam(train_loader, gpt, device,
                                       num_batches=5)
    val_loss = calc_loss_loader_spam(val_loader, gpt, device, num_batches=5)
    test_loss = calc_loss_loader_spam(test_loader, gpt, device, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")


# Finetuning
manual_seed(123)

start_time = time()

optimizer = optim.AdamW(gpt.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_spam(gpt, train_loader, val_loader, optimizer, device,
                          num_epochs=num_epochs, eval_freq=50, eval_iter=5)

end_time = time()
execution_time_minutes = (end_time - start_time) / 60

print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# plot training metrics of loss
epochs_tensor = linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)


# Plot training metrics of accuracy
epochs_tensor = linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = linspace(0, examples_seen, len(train_accs))

plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs,
            label="accuracy")

# Print accuracies of training, validation, and test
train_accuracy = calc_accuracy_loader(train_loader, gpt, device)
val_accuracy = calc_accuracy_loader(val_loader, gpt, device)
test_accuracy = calc_accuracy_loader(test_loader, gpt, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Test the model
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(text_1, gpt, tokenizer, device,
                      max_length=train_dataset.max_length))

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(text_2, gpt, tokenizer, device,
                      max_length=train_dataset.max_length))

# Save the model
save_torch(gpt.state_dict(), "models" + sep + "review_classifier.pth")

# Load model
model_state_dict = load_torch("models" + sep + "review_classifier.pth")
gpt.load_state_dict(model_state_dict)
