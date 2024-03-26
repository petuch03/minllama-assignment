from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # TODO
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    freqs = theta ** (-2 * torch.arange(0, head_dim / 2, device=device).float() / head_dim)
    m = torch.arange(seqlen, device=device)
    angles = torch.outer(m, freqs).float()
    cos, sin = torch.cos(angles), torch.sin(angles)
    cos, sin = reshape_for_broadcast(cos, query_real), reshape_for_broadcast(sin, query_real)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    query_out_1_real, query_out_1_imag = query_real * cos, query_imag * cos
    query_out_2_real, query_out_2_imag = query_real * sin, -query_imag * sin
    query_out_1 = query_out_1_real + query_out_2_imag
    query_out_2 = query_out_1_imag + query_out_2_real

    key_out_1_real, key_out_1_imag = key_real * cos, key_imag * cos
    key_out_2_real, key_out_2_imag = key_real * sin, -key_imag * sin
    key_out_1 = key_out_1_real + key_out_2_imag
    key_out_2 = key_out_1_imag + key_out_2_real

    # merging into single query_out and key_out
    target = list(query_out_1.shape)
    target[-1] = -1

    query_stacked = torch.stack((query_out_1, query_out_2), dim=-1)
    query_out = query_stacked.reshape(target)

    key_stacked = torch.stack((key_out_1, key_out_2), dim=-1)
    key_out = key_stacked.reshape(target)
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out
