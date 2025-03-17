"""
Created on Fri Mar 14 16:54:47 2025.

@author: kayol
"""

from torch import no_grad, softmax, argmax, cat, tensor, topk, where, \
    multinomial


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

        # Get the idx of the vocab entry with the highest probability value.
        # It is called greedy decoding.
        idx_next = argmax(probs, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    """
    Convert a text string into a tensor of token IDs using a tokenizer.

    This function tokenizes the given text into token IDs and returns a tensor
    with an added batch dimension.

    Parameters
    ----------
        text (str): The input text to be tokenized.
        tokenizer: A tokenizer object with an `encode` method for converting
                  text to token IDs.

    Returns
    -------
        torch.Tensor: Tensor of shape (1, seq_len), containing tokenized text.
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = tensor(encoded).unsqueeze(0)  # add batch dimension

    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert a tensor of token IDs back into a text string using a tokenizer.

    This function removes the batch dimension and decodes the token IDs into
    human-readable text.

    Parameters
    ----------
        token_ids (torch.Tensor): A tensor of tokenized text with shape
                                  (1, seq_len) or (seq_len,).
        tokenizer: A tokenizer object with a `decode` method for converting
                   token IDs back to text.

    Returns
    -------
        str: The decoded text.
    """
    flat = token_ids.squeeze(0)  # remove batch dimension

    return tokenizer.decode(flat.tolist())


def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0,
                  top_k=None, eos_id=None):
    """
    Generate text using optional temperature scaling and top-k sampling.

    This function generates new tokens by iteratively passing the current
    token sequence through the model, predicting the next token, and appending
    it to the sequence. It supports:
    - Greedy decoding (argmax)
    - Temperature-scaled sampling for randomness
    - Top-k filtering to restrict sampling to the most probable tokens
    - Early stopping when an end-of-sequence (EOS) token is encountered

    Parameters
    ----------
        model (torch.nn.Module): The language model used for text generation.
        idx (torch.Tensor): Input tensor of shape (batch_size, seq_len),
                            representing token indices in the current context.
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The maximum number of tokens the model can consider
                            as context.
        temperature (float, optional): A scaling factor for logits before
                                       sampling. A value of 0.0 defaults to
                                       greedy decoding. (default: 0.0)
        top_k (int, optional): If specified, restricts sampling to the top-k
                               most probable tokens at each step.
                               (default: None)
        eos_id (int, optional): If specified, generation stops when this token
                                is produced. (default: None)

    Returns
    -------
        torch.Tensor: A tensor of shape (batch_size, seq_len + max_new_tokens),
                      containing the generated token indices.
    """
    # Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = where(logits < min_val,
                           tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest
        # logits value
        else:
            idx_next = argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop generating early if end-of-sequence token is encountered and
        # eos_id is specified
        if idx_next == eos_id:
            break

        # Same as before: append sampled index to the running sequence
        idx = cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
