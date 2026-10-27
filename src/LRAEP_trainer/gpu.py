"""Custom PyTorch GPU transcript search, derived from PEARL's reference math.

Reference: official miner/miner-base/src/miner_base/noisy_gemm.py and
zk-pow/src/ffi/mine.rs, revision 0798903bced203de8cfcb49441f44fd9b51b18eb.
The GPU performs the matrix arithmetic and transcript accumulation; callers
check the resulting 64-byte messages with keyed BLAKE3.
"""

from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
import torch


RANK = 128
_ROTATE_RIGHT_MASK = (1 << 13) - 1
# Several output-sized intermediates coexist during the batched reduction.
# Keep their aggregate size bounded, rather than materializing all k slabs.
_TRANSCRIPT_SCRATCH_BYTES = 256 * 1024 * 1024


class TrainingCancelled(RuntimeError):
    """The current training job was cancelled between bounded matrix stages."""


def _as_int8_tensor(value: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        if value.dtype != np.int8:
            raise ValueError(f"{name} must have dtype int8")
        tensor = torch.from_numpy(np.ascontiguousarray(value))
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        raise TypeError(f"{name} must be a NumPy array or PyTorch tensor")
    if tensor.dtype != torch.int8:
        raise ValueError(f"{name} must have dtype int8")
    if tensor.ndim != 2 or 0 in tensor.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    return tensor


def _xor_tiles(values: torch.Tensor) -> torch.Tensor:
    """Reduce the last dimension using an XOR tree, including odd sizes."""
    while values.shape[-1] > 1:
        pairs = values.shape[-1] // 2
        reduced = values[..., : 2 * pairs : 2] ^ values[..., 1 : 2 * pairs : 2]
        if values.shape[-1] % 2:
            reduced = torch.cat((reduced, values[..., -1:]), dim=-1)
        values = reduced
    return values[..., 0]


@contextmanager
def _exact_matmul():
    """Use exact float32 integer products and restore the caller's settings."""
    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _destination(device):
    destination = torch.device(device)
    if destination.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable GPU and CUDA-enabled PyTorch are required")
    return destination


@torch.inference_mode()
def gpu_add_noise(a, bt, al, ar, bl, br, device="cuda:0"):
    """Apply official rank-128 noise factors on the selected device.

    Signals and all four factors are int8 matrices. Rank-128 signed int8 dot
    products, including full-range inputs, have absolute sums <= 2**21 and
    are therefore exact in float32. Convert their results to int32 and wrap
    to int8 *before* adding signals, preserving the official integer rules.
    Return contiguous int8 ``[m,k]`` and ``[k,n]`` matrices on ``device``.
    """
    a, bt, al, ar, bl, br = (
        _as_int8_tensor(value, name)
        for value, name in ((a, "a"), (bt, "bt"), (al, "al"),
                            (ar, "ar"), (bl, "bl"), (br, "br"))
    )
    m, k = a.shape
    n, other_k = bt.shape
    if other_k != k:
        raise ValueError("a and bt have incompatible common dimensions")
    if (al.shape != (m, RANK) or ar.shape != (RANK, k)
            or bl.shape != (k, RANK) or br.shape != (RANK, n)):
        raise ValueError("noise factors must match the signal dimensions and rank 128")
    destination = _destination(device)
    with _exact_matmul():
        noise_a = (al.to(destination, dtype=torch.float32)
                   @ ar.to(destination, dtype=torch.float32)).to(torch.int32).to(torch.int8)
        noise_b = (bl.to(destination, dtype=torch.float32)
                   @ br.to(destination, dtype=torch.float32)).to(torch.int32).to(torch.int8)
        noise_a.add_(a.to(destination))
        noise_b.add_(bt.to(destination).T)
    return noise_a.contiguous(), noise_b.contiguous()


def _as_rank_indices(value, name, k):
    if isinstance(value, np.ndarray):
        if value.dtype != np.int64:
            raise ValueError(f"{name} must have dtype int64")
        indices = torch.from_numpy(np.ascontiguousarray(value))
    elif isinstance(value, torch.Tensor):
        indices = value
    else:
        raise TypeError(f"{name} must be a NumPy array or PyTorch tensor")
    if indices.dtype != torch.int64:
        raise ValueError(f"{name} must have dtype int64")
    if indices.ndim != 1 or indices.shape[0] != k:
        raise ValueError(f"{name} must be a vector with length k")
    # Official factors arrive on the CPU. Validate there before any GPU
    # allocations; existing GPU indices are also supported via a small copy.
    values = indices.cpu().numpy()
    if values.min() < 0 or values.max() >= RANK:
        raise ValueError(f"{name} indices must lie between 0 and 127")
    return indices


@torch.inference_mode()
def gpu_add_sparse_noise(a, bt, al, a_first, a_second, b_first, b_second,
                         br, device="cuda:0"):
    """Apply the official two-entry permutation factors with integer gathers.

    Every permutation column/row contains a +1 and a -1. Its rank-128
    matrix product is therefore exactly a difference of two selected uniform
    noise entries. Int8 subtraction and addition preserve the official wrap
    semantics, including full-range int8 signals and uniform factors. This
    avoids dense permutation matrices, float intermediates, and matmuls.

    ``al`` is ``[m,128]`` and ``br`` is ``[128,n]``. All four index vectors
    have dtype int64, length ``k``, and values in ``[0,127]``. Return
    contiguous int8 ``[m,k]`` and ``[k,n]`` tensors on the selected device.
    """
    a, bt, al, br = (
        _as_int8_tensor(value, name)
        for value, name in ((a, "a"), (bt, "bt"), (al, "al"), (br, "br"))
    )
    m, k = a.shape
    n, other_k = bt.shape
    if other_k != k:
        raise ValueError("a and bt have incompatible common dimensions")
    if al.shape != (m, RANK) or br.shape != (RANK, n):
        raise ValueError("noise factors must match the signal dimensions and rank 128")
    a_first, a_second, b_first, b_second = (
        _as_rank_indices(value, name, k)
        for value, name in ((a_first, "a_first"), (a_second, "a_second"),
                            (b_first, "b_first"), (b_second, "b_second"))
    )
    destination = _destination(device)
    al = al.to(destination)
    noise_a = al.index_select(1, a_first.to(destination))
    noise_a.sub_(al.index_select(1, a_second.to(destination)))
    noise_a.add_(a.to(destination))
    br = br.to(destination)
    noise_b = br.index_select(0, b_first.to(destination))
    noise_b.sub_(br.index_select(0, b_second.to(destination)))
    noise_b.add_(bt.to(destination).T)
    return noise_a.contiguous(), noise_b.contiguous()


def _slabs_per_chunk(m, n):
    # A power-of-two chunk divides the 16 transcript slots, so one vectorized
    # update never wraps across the slot boundary. At least one slab fits even
    # when a single large output matrix exceeds the preferred scratch budget.
    affordable = max(1, _TRANSCRIPT_SCRATCH_BYTES // (m * n * 24))
    return min(16, 1 << (affordable.bit_length() - 1))


@torch.inference_mode()
def gpu_transcripts(
    a_noised: np.ndarray | torch.Tensor,
    b_noised: np.ndarray | torch.Tensor,
    tile_h: int = 8,
    tile_w: int = 8,
    device: str | torch.device = "cuda:0",
    cancelled: Callable[[], bool] = lambda: False,
) -> np.ndarray:
    """Compute PEARL transcripts for every contiguous output tile.

    ``a_noised`` has shape [m, k] and ``b_noised`` has shape [k, n].
    Inputs must be int8. The rank is fixed at 128 and k must be a positive
    multiple of it. Return uint32 [m // tile_h, n // tile_w, 16] on the CPU.

    Every rank-128 product is evaluated in float32 with TF32 disabled. A
    product of signed int8 values and any intermediate sum of 128 such
    products is bounded by 2**21, below float32's exact-integer limit of
    2**24. Each slab is converted to int32 *before* cumulative addition.
    Products are batched in bounded chunks, accumulated by integer cumsum,
    and XOR-reduced across all slabs in each chunk together. This preserves
    PEARL's integer arithmetic while avoiding a Python launch sequence for
    every slab, including on the GTX 1650 without int32 matrix multiplication.
    """
    a = _as_int8_tensor(a_noised, "a_noised")
    b = _as_int8_tensor(b_noised, "b_noised")
    m, k = a.shape
    other_k, n = b.shape
    if other_k != k:
        raise ValueError("a_noised and b_noised have incompatible common dimensions")
    if k % RANK:
        raise ValueError("the common dimension must be a multiple of 128")
    if k > 65536:
        raise ValueError("the common dimension must be at most 65536")
    if not isinstance(tile_h, int) or not isinstance(tile_w, int):
        raise ValueError("tile dimensions must be integers")
    if tile_h <= 0 or tile_w <= 0 or tile_h % 2 or tile_w % 2:
        raise ValueError("tile dimensions must be positive even integers")
    if not 32 <= tile_h * tile_w <= 256:
        raise ValueError("a transcript tile must contain between 32 and 256 cells")
    if m % tile_h or n % tile_w:
        raise ValueError("matrix dimensions must be divisible by transcript tile dimensions")
    if cancelled():
        raise TrainingCancelled("training job cancelled")

    destination = _destination(device)
    rows, cols = m // tile_h, n // tile_w
    chunk_size = _slabs_per_chunk(m, n)
    with _exact_matmul():
        a = a.to(destination).contiguous()
        b = b.to(destination).contiguous()
        cumulative = None
        transcript = torch.zeros((16, rows, cols), dtype=torch.int32, device=destination)
        for slab in range(0, k // RANK, chunk_size):
            if cancelled():
                raise TrainingCancelled("training job cancelled")
            count = min(chunk_size, k // RANK - slab)
            start, end = slab * RANK, (slab + count) * RANK
            left = a[:, start:end].reshape(m, count, RANK).permute(1, 0, 2)
            right = b[start:end].reshape(count, RANK, n)
            partial = torch.bmm(left.to(torch.float32), right.to(torch.float32)).to(torch.int32)
            accumulated = partial.cumsum(dim=0, dtype=torch.int32)
            if cumulative is not None:
                accumulated.add_(cumulative)
            cumulative = accumulated[-1]
            tiles = accumulated.reshape(count, rows, tile_h, cols, tile_w)
            tiles = tiles.permute(0, 1, 3, 2, 4).reshape(count, rows, cols, tile_h * tile_w)
            reduction = _xor_tiles(tiles)
            slot = slab % 16
            previous = transcript[slot:slot + count]
            if slab < 16:
                previous.copy_(reduction)
            else:
                rotated = (previous << 13) | ((previous >> 19) & _ROTATE_RIGHT_MASK)
                previous.copy_(rotated ^ reduction)
            if cancelled():
                raise TrainingCancelled("training job cancelled")
        # A signed int32 transcript has exactly the required uint32 bit pattern.
        return transcript.permute(1, 2, 0).contiguous().cpu().numpy().view(np.uint32)
