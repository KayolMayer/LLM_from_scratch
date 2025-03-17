"""
Created on Mon Mar 17 12:20:42 2025.

@author: kayol
"""

from torch.nn.functional import cross_entropy


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Compute the cross-entropy loss for a single batch of input and target data.

    This function moves the input and target tensors to the specified device,
    passes the input through the model to obtain logits, and calculates the
    cross-entropy loss between the model predictions and the target labels.

    Parameters
    ----------
        input_batch (torch.Tensor): The input batch tensor of shape
                                    (batch_size, seq_len).
        target_batch (torch.Tensor): The target batch tensor of shape
                                     (batch_size, seq_len).
        model (torch.nn.Module): The language model used for predictions.
        device (torch.device): The device (CPU or GPU) on which to perform
                               computations.

    Returns
    -------
        torch.Tensor: The computed cross-entropy loss for the batch.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Compute the average cross-entropy loss over multiple batches from a loader.

    This function iterates through a specified number of batches from the given
    data loader, computes the loss for each batch and returns the average loss.

    Parameters
    ----------
        data_loader (torch.utils.data.DataLoader): The data loader providing
                                                   batches of input and target
                                                   tensors.
        model (torch.nn.Module): The language model used for generating
                                 predictions.
        device (torch.device): The device (CPU or GPU) on which to perform
                               computations.
        num_batches (int, optional): The number of batches to compute the loss
                                     over. If None, the loss is computed over
                                     the entire data loader.

    Returns
    -------
        float: The average loss over the specified number of batches.
               Returns NaN if the data loader is empty.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in
        # the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):

        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches
