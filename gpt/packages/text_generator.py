"""
Created on Fri Mar 14 16:54:47 2025.

@author: kayol
"""

from torch import no_grad, softmax, argmax, cat


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text autoregressively using a given language model.

    This function takes an initial token sequence and iteratively predicts the
    next token using the model's logits, selecting the most probable token at
    each step. It maintains a rolling context window to respect the model's
    maximum supported context size.

    Parameters
    ----------
        model (torch.nn.Module): A GPT-style language model that takes a tensor
                                 of token indices and returns logits.
        idx (torch.Tensor): Input tensor of shape (batch_size, n_tokens),
                            representing the initial token indices.
        max_new_tokens (int): The number of new tokens to generate.
        context_size (int): The maximum number of tokens the model can consider
                            as context at any step.

    Returns
    -------
        torch.Tensor: A tensor of shape (batch_size,n_tokens + max_new_tokens),
                      containing the generated token indices.
    """
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probs = softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = argmax(probs, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
