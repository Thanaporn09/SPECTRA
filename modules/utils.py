import numpy as np
import cv2
import torch
import torch.nn as nn

class CrossAttentionTokenReducer(nn.Module):
    def __init__(self, hidden_dim, target_length, num_heads):
        """
        Initializes the cross-attention token reducer layer.

        Args:
            hidden_dim (int): The dimensionality of the input and output features.
            target_length (int): The desired sequence length after reduction.
            num_heads (int): The number of attention heads.
        """
        super(CrossAttentionTokenReducer, self).__init__()
        self.hidden_dim = hidden_dim
        self.target_length = target_length
        self.num_heads = num_heads

        # Initialize query as a learnable parameter
        self.query = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1, target_length, hidden_dim), 0., 0.2))

        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, x):
        """
        Forward pass for the token reduction layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, hidden_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, target_length, hidden_dim).
        """
        # Reshape input to match the shape expected by nn.MultiheadAttention
        # Shape becomes (seq_length, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)

        # Repeat the query for each item in the batch and permute
        query = self.query.repeat(x.size(1), 1, 1).permute(1, 0, 2)

        # Apply multi-head attention
        # Output shape: (target_length, batch_size, hidden_dim)
        attn_output, _ = self.multihead_attn(query=query, key=x, value=x)

        # Reshape output to (batch_size, target_length, hidden_dim)
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, weights):
    image = image.transpose(1, 2, 0)
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
    weights = weights - np.min(weights)
    weights = weights / np.max(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result