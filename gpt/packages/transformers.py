"""
Created on Fri Mar 14 12:02:18 2025.

@author: kayol
"""

from torch import tensor, inf, softmax, ones, zeros, triu, cat, sqrt, tanh, \
    pi, pow, arange
from torch.nn import Module, ModuleList, Parameter, Linear, Dropout, \
    Sequential, Embedding


class causal_attention(Module):
    """
    Implement a causal self-attention mechanism for transformer-based models.

    This module computes attention scores using query, key, and value
    projections, while ensuring causality by applying an upper-triangular mask
    to prevent information leakage from future tokens.

    Attributes
    ----------
        d_out (int): The output dimension of the attention mechanism.
        W_query (torch.nn.Linear): Linear transformation for computing query
                                   vectors.
        W_key (torch.nn.Linear): Linear transformation for computing key
                                 vectors.
        W_value (torch.nn.Linear): Linear transformation for computing value
                                   vectors.
        dropout (torch.nn.Dropout): Dropout applied to the attention weights.
        mask (torch.Tensor): Upper-triangular mask to enforce causality.

    Parameters
    ----------
        d_in (int): The input dimension of each token representation.
        d_out (int): The output dimension for the transformed queries, keys,
                     and values.
        context_length (int): The maximum sequence length for the attention
                              mask.
        dropout (float): Dropout probability applied to attention weights.
        qkv_bias (bool, optional): Whether to include a bias term in the query,
        key, and value projections (default: False).

    Methods
    -------
        forward(x):
            Computes the causal self-attention output for an input tensor.
    """

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = Dropout(dropout)
        self.register_buffer('mask', triu(ones(context_length, context_length),
                                          diagonal=1))

    def forward(self, x):
        """
        Compute causal self-attention for the given input tensor.

        Parameters
        ----------
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len,d_in).

        Returns
        -------
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out),
                          containing the attended representations.
        """
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -inf)
        attn_weights = softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        return context_vec


class multi_head_attention_wrapper(Module):
    """
    Implement a multi-head causal attention mechanism.

    It is obtained by wrapping multiple instances of single-head causal
    attention. This module applies multiple independent causal attention heads
    in parallel and concatenates their outputs to increase the model's
    expressiveness.

    Attributes
    ----------
        heads (torch.nn.ModuleList): A list of causal attention heads.

    Parameters
    ----------
        d_in (int): The input dimension of each token representation.
        d_out (int): The output dimension for each attention head.
        context_length (int): The maximum sequence length for the attention
                              mask.
        dropout (float): Dropout probability applied to attention weights.
        num_heads (int): The number of independent attention heads.
        qkv_bias (bool, optional): Whether to include bias terms in query, key,
                                   and value projections (default: False).

    Methods
    -------
        forward(x):
            Applies multi-head causal attention to the input tensor.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False):
        super().__init__()
        self.heads = ModuleList(
            [causal_attention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        """
        Apply multi-head causal attention to the input tensor.

        Parameters
        ----------
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len,d_in).

        Returns
        -------
            torch.Tensor: Output tensor of shape (batch_size, seq_len,
                                                  d_out * num_heads),
                          concatenating the outputs of all attention heads.
        """
        return cat([head(x) for head in self.heads], dim=-1)


class multi_head_attention(Module):
    """
    Implements multi-head self-attention with weights' split.

    This module performs scaled dot-product attention across multiple
    independent attention heads. It projects input tokens into query, key, and
    value spaces, splits them into multiple heads, applies attention within
    each head, and then combines the outputs into a single representation.

    Attributes
    ----------
        d_out (int): The output dimensionality of the attention mechanism.
        num_heads (int): The number of independent attention heads.
        head_dim (int): The dimensionality of each attention head.
        W_query (torch.nn.Linear): Linear layer to project inputs to query
                                   vectors.
        W_key (torch.nn.Linear): Linear layer to project inputs to key vectors.
        W_value (torch.nn.Linear): Linear layer to project inputs to value
                                   vectors.
        out_proj (torch.nn.Linear): Linear layer to combine attention outputs
                                    from all heads.
        dropout (torch.nn.Dropout): Dropout layer applied to attention weights.
        mask (torch.Tensor): Upper-triangular causal mask to prevent
                             information leakage from future tokens.

    Parameters
    ----------
        d_in (int): The input dimension of each token representation.
        d_out (int): The output dimension for the attention mechanism.
        context_length (int): The maximum sequence length for the attention
                              mask.
        dropout (float): Dropout probability applied to attention weights.
        num_heads (int): The number of independent attention heads.
        qkv_bias (bool, optional): Whether to include bias terms in the query,
                                   key, and value projections (default: False).

    Methods
    -------
        forward(x):
            Applies multi-head causal attention to the input tensor.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Reduce the projection dim to match desired output dim
        self.head_dim = d_out // num_heads

        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)
        # Linear layer to combine head outputs
        self.out_proj = Linear(d_out, d_out)
        self.dropout = Dropout(dropout)
        self.register_buffer(
            "mask", triu(ones(context_length, context_length), diagonal=1)
            )

    def forward(self, x):
        """
        Apply multi-head causal self-attention to the input tensor.

        Parameters
        ----------
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len,d_in).

        Returns
        -------
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out),
                          containing attended token representations.
        """
        b, num_tokens, d_in = x.shape

        # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads,
        #                                             head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads,
        #                                                     num_tokens,
        #                                                     head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a
        # causal mask
        # Dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)

        # Original mask truncated to the number of tokens and converted to
        # boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -inf)

        attn_weights = softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # optional projection
        context_vec = self.out_proj(context_vec)

        return context_vec


class layer_norm(Module):
    """
    Implements Layer Normalization for stabilizing neural network training.

    This module normalizes the input across the last dimension (feature
    dimension) by subtracting the mean and dividing by the standard deviation,
    with learnable scale and shift parameters.

    Attributes
    ----------
        eps (float): A small constant added to the variance for numerical
                     stability.
        scale (torch.nn.Parameter): Learnable scaling factor applied after
                                    normalization.
        shift (torch.nn.Parameter): Learnable shift factor applied after
                                    normalization.

    Parameters
    ----------
        emb_dim (int): The number of features (embedding dimension) for
                       normalization.

    Methods
    -------
        forward(x):
            Applies layer normalization to the input tensor.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = Parameter(ones(emb_dim))
        self.shift = Parameter(zeros(emb_dim))

    def forward(self, x):
        """
        Apply layer normalization to the input tensor.

        Parameters
        ----------
            x (torch.Tensor): Input tensor of shape (..., emb_dim), where
                              normalization is applied over the last dimension.

        Returns
        -------
            torch.Tensor: Normalized tensor of the same shape as the input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


class gelu(Module):
    """
    Implement the Gaussian Error Linear Unit (GELU) activation function.

    The GELU activation function is approximated as:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))

    GELU is commonly used in transformer-based models, such as BERT and GPT,
    as it provides smooth, non-linear activation with improved performance
    over ReLU in some deep learning tasks.

    Methods
    -------
        forward(x):
            Applies the GELU activation function to the input tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Apply the GELU activation function to the input tensor.

        Parameters
        ----------
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Tensor with GELU activation applied.
        """
        return 0.5 * x * (1 + tanh(sqrt(tensor(2.0 / pi)) *
                                   (x + 0.044715 * pow(x, 3))))


class feedforward(Module):
    """
    Implement a feedforward neural network module used in transformers.

    This module consists of two linear layers with a GELU activation function
    in between. It expands the input dimension by a factor of 4 before
    projecting it back to the original size. This design is commonly used in
    transformer-based models like GPT and BERT.

    Attributes
    ----------
        layers (torch.nn.Sequential): A sequential container of linear and
                                      activation layers.

    Parameters
    ----------
        cfg (dict): Configuration dictionary containing model hyperparameters.
            - "emb_dim" (int): The embedding dimension of the model.

    Methods
    -------
        forward(x):
            Passes the input through the feedforward network.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = Sequential(
            Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expansion
            gelu(),  # Activation function
            Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # Contraction
        )

    def forward(self, x):
        """
        Pass the input through the feedforward network.

        Parameters
        ----------
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len,
                                                     emb_dim).

        Returns
        -------
            torch.Tensor: Output tensor of the same shape as input.
        """
        return self.layers(x)


class transformer_block(Module):
    """
    Implement a single Transformer block.

    This module follows the standard Transformer architecture where each block
    consists of:
    1. A multi-head self-attention mechanism with layer normalization and a
       residual connection.
    2. A position-wise feedforward network with layer normalization and another
       residual connection.
    3. Skip connections to make the loss smoother. Please, read the paper:
        Visualizing the Loss Landscape of Neural Nets
        https://arxiv.org/pdf/1712.09913

    Attributes
    ----------
        att (multi_head_attention): Multi-head self-attention mechanism.
        ff (feedforward): Position-wise feedforward network.
        norm1 (layer_norm): Layer normalization applied before the attention
                            block.
        norm2 (layer_norm): Layer normalization applied before the feedforward
                            block.
        drop_shortcut (torch.nn.Dropout): Dropout layer applied to residual
                                          connections.

    Parameters
    ----------
        cfg (dict): Configuration dictionary containing model hyperparameters:
            - "emb_dim" (int): The embedding dimension of the model.
            - "context_length" (int): The maximum sequence length.
            - "n_heads" (int): The number of attention heads.
            - "drop_rate" (float): Dropout probability.
            - "qkv_bias" (bool): Whether to include bias terms in query, key,
                                 and value projections.

    Methods
    -------
        forward(x):
            Processes input through the Transformer block, including attention,
            feedforward layers, and residual connections.
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = multi_head_attention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = feedforward(cfg)
        self.norm1 = layer_norm(cfg["emb_dim"])
        self.norm2 = layer_norm(cfg["emb_dim"])
        self.drop_shortcut = Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Process input through the Transformer block.

        Applying self-attention, a feedforward network, normalization, and
        residual connections.

        Parameters
        ----------
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len,
                                                     emb_dim).

        Returns
        -------
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class gpt_model(Module):
    """
    Implement a GPT-style transformer model for autoregressive text generation.

    This model consists of:
    - Token and positional embeddings.
    - A stack of Transformer blocks with self-attention and feedforward layers.
    - Layer normalization before the final output.
    - A linear output projection to predict token logits.

    Attributes
    ----------
        tok_emb (torch.nn.Embedding): Token embedding layer mapping vocab
                                      indices to embeddings.
        pos_emb (torch.nn.Embedding): Positional embedding layer encoding
                                      sequence position.
        drop_emb (torch.nn.Dropout): Dropout applied to the sum of token and
                                     positional embeddings.
        trf_blocks (torch.nn.Sequential): Stack of Transformer blocks for
                                          contextual learning.
        final_norm (layer_norm): Layer normalization applied before final
                                 projection.
        out_head (torch.nn.Linear): Linear output layer mapping embeddings to
                                    vocabulary logits.

    Parameters
    ----------
        cfg (dict): Configuration dictionary containing model hyperparameters:
            - "vocab_size" (int): The size of the vocabulary.
            - "emb_dim" (int): The embedding dimension.
            - "context_length" (int): The maximum sequence length.
            - "drop_rate" (float): Dropout probability.
            - "n_layers" (int): The number of transformer blocks in the model.

    Methods
    -------
        forward(in_idx):
            Computes the token logits for a given input sequence.
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = Dropout(cfg["drop_rate"])

        self.trf_blocks = Sequential(
            *[transformer_block(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = layer_norm(cfg["emb_dim"])
        self.out_head = Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        """
        Compute the token logits for a given input sequence.

        Parameters
        ----------
            in_idx (torch.Tensor): Input tensor of shape (batch_size, seq_len)
                                   containing token indices.

        Returns
        -------
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size),
                          representing the probability distribution over
                          vocabulary tokens.
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
