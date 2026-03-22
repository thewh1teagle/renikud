# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

import torch
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.
    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore[arg-type]
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Match upstream LLaMA for 2D (seq, head_dim/2); also support 3D (batch, seq, head_dim/2) from position_ids."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.ndim == 2:
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"freqs_cis.shape: {freqs_cis.shape}, x.shape: {x.shape}"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    assert freqs_cis.ndim == 3, f"freqs_cis.ndim: {freqs_cis.ndim}"
    assert freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1]), f"freqs_cis.shape: {freqs_cis.shape}, x.shape: {x.shape}"
    shape = [d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.
    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    cos = freqs_cis.real.to(xq.dtype)
    sin = freqs_cis.imag.to(xq.dtype)
    
    xq_ = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 2)

    cos = reshape_for_broadcast(cos, xq_[..., 0])
    sin = reshape_for_broadcast(sin, xq_[..., 0])
    
    xq_out = torch.empty_like(xq_)
    xq_out[..., 0] = xq_[..., 0] * cos - xq_[..., 1] * sin
    xq_out[..., 1] = xq_[..., 0] * sin + xq_[..., 1] * cos
    
    xk_out = torch.empty_like(xk_)
    xk_out[..., 0] = xk_[..., 0] * cos - xk_[..., 1] * sin
    xk_out[..., 1] = xk_[..., 0] * sin + xk_[..., 1] * cos
    
    return xq_out.flatten(3), xk_out.flatten(3)
