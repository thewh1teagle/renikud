# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

import os
import torch
from typing import Tuple

_ONNX_EXPORT = os.getenv("NEOBERT_ONNX_EXPORT") == "1"


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
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    if _ONNX_EXPORT:
        # ONNX does not support complex64; store as real [..., dim//2, 2] (cos, sin)
        return torch.stack([freqs.cos(), freqs.sin()], dim=-1)
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    if not _ONNX_EXPORT:  # assert fires during ONNX symbolic tracing where shapes are not concrete
        assert freqs_cis.shape[1:] == (x.shape[1], x.shape[-1])
    return freqs_cis.contiguous().unsqueeze(2)


def apply_rotary_emb_pure(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real-valued rotation equivalent to apply_rotary_emb, without complex tensors.
    Used when NEOBERT_ONNX_EXPORT=1 since ONNX does not support complex64.
    freqs_cis must be [..., dim//2, 2] (cos, sin) as returned by precompute_freqs_cis in that mode.
    (x_r + i*x_i) * (cos + i*sin) = (x_r*cos - x_i*sin) + i*(x_r*sin + x_i*cos)
    """
    def rotate(x):
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        cos = freqs_cis[..., 0].unsqueeze(2)
        sin = freqs_cis[..., 1].unsqueeze(2)
        x_r, x_i = x_[..., 0], x_[..., 1]
        return torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1).flatten(-2)
    return rotate(xq).type_as(xq), rotate(xk).type_as(xk)


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
    if _ONNX_EXPORT:
        return apply_rotary_emb_pure(xq, xk, freqs_cis)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
