"""
Created on Fri Mar 21 17:15:36 2025.

@author: kayol
"""

from torch import no_grad, argmax


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Compute the accuracy of a model over a dataset using a DataLoader.

    This function iterates through the DataLoader, passes inputs through the
    model, and compares predictions with ground truth labels to compute
    accuracy.

    Parameters
    ----------
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches
                                                   of input and target tensors.
        model (torch.nn.Module): The neural network model used for inference.
        device (torch.device): The device (CPU or GPU) on which computations
                               are performed.
        num_batches (int, optional): The number of batches to evaluate.
                                     If None, evaluates the entire dataset.

    Returns
    -------
        float: The accuracy of the model, computed as
               (correct predictions / total samples).
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with no_grad():
                # Logits of last output token
                logits = model(input_batch)[:, -1, :]
            predicted_labels = argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels ==
                                    target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples
