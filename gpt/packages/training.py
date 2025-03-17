"""
Created on Mon Mar 17 15:32:05 2025.

@author: kayol
"""

from packages.loss_functions import calc_loss_batch, calc_loss_loader
from packages.text_generator import text_to_token_ids, token_ids_to_text, \
    generate_text
from torch import no_grad, linspace
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context,
                       tokenizer):
    """
    Train a language model using mini-batch gradient descent.

    This function iterates through the training data, computes loss, updates
    model parameters, and periodically evaluates the model on both the training
    and validation sets. After each epoch, it generates and prints a sample
    text to monitor progress.

    Parameters
    ----------
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training
                                                    dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation
                                                  dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model
                                           parameters.
        device (torch.device): The device (CPU or GPU) on which to train the
                               model.
        num_epochs (int): The number of training epochs.
        eval_freq (int): The number of steps between each evaluation.
        eval_iter (int): The number of evaluation iterations during validation.
        start_context (str): The initial text prompt for generating a sample
                             after each epoch.
        tokenizer: Tokenizer with 'encode' and 'decode' methods for text
                   processing.

    Returns
    -------
        tuple: A tuple containing:
            - train_losses (list of float): List of training losses at each
                                            evaluation step.
            - val_losses (list of float): List of validation losses at each
                                          evaluation step.
            - track_tokens_seen (list of int): Cumulative number of tokens
                                               processed at each evaluation
                                               step.
    """
    # Initialize lists to track losses and tokens seen.
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop.
    for epoch in range(num_epochs):
        # Set model to training mode.
        model.train()

        for input_batch, target_batch in train_loader:

            # Reset loss gradients from previous batch iteration
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Calculate loss gradients
            loss.backward()
            # Update model weights using loss gradients
            optimizer.step()
            # Returns the total number of elements (or tokens) in the
            # input_batch.
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader,
                                                      val_loader, device,
                                                      eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate a model on both training and validation datasets.

    This function temporarily sets the model to evaluation mode, computes
    the average loss over a limited number of batches from the training
    and validation sets, and then restores the model to training mode.

    Parameters
    ----------
        model (torch.nn.Module): The neural network model to be evaluated.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training
                                                    dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation
                                                  dataset.
        device (torch.device): The device (CPU or GPU) on which to run
                               evaluation.
        eval_iter (int): The number of batches to use for computing the average
                         loss.

    Returns
    -------
        tuple: A tuple containing:
            - train_loss (float): The average training loss over 'eval_iter'
                                  batches.
            - val_loss (float): The average validation loss over 'eval_iter'
                                batches.
    """
    model.eval()

    with no_grad():
        train_loss = calc_loss_loader(train_loader, model, device,
                                      num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device,
                                    num_batches=eval_iter)
    model.train()

    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generate and prints a sample text from the model using decoding.

    This function tokenizes the starting context, generates new tokens using
    the model, decodes them back into text, and prints the generated output in
    a compact format.

    Parameters
    ----------
        model (torch.nn.Module): The language model used for text generation.
        tokenizer: A tokenizer object with 'encode' and 'decode' methods for
                   text processing.
        device (torch.device): The device (CPU or GPU) on which the model runs.
        start_context (str): The initial text prompt for generation.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with no_grad():
        token_ids = generate_text(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size,
            temperature=1.4, top_k=25, eos_id=None
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    # Compact print format
    print(decoded_text.replace("\n", " "))

    model.train()
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with no_grad():
        token_ids = generate_text(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size,
            temperature=1.4, top_k=25, eos_id=None
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    # Compact print format
    print(decoded_text.replace("\n", " "))

    model.train()


def plot_losses(num_epochs, tokens_seen, train_losses, val_losses):
    """
    Plot training and validation losses against epochs and tokens seen.

    This function generates a loss curve with two x-axes:
    - The primary x-axis represents epochs.
    - The secondary x-axis represents the number of tokens seen.
    The plot includes both training and validation losses.

    Returns
    -------
        num_epochs (int): Number of training epochs.
        tokens_seen (list of int): A list of cumulative token counts
                                   corresponding to recorded losses.
        train_losses (list of float): Training losses.
        val_losses (list of float): Validation losses.

    Parameters
    ----------
        None: The function saves the plot as 'loss-plot.pdf' and displays it.
    """
    epochs_seen = linspace(0, num_epochs, len(train_losses))

    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')

    # only show integer labels on x-axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create a second x-axis for tokens seen
    # Create a second x-axis that shares the same y-axis
    ax2 = ax1.twiny()
    # Invisible plot for aligning ticks
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()
