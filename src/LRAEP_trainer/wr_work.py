"""Deterministic S7 work, independently checked against accepted shares.

WR's ``sb: 1, tail_mode: 1`` witness reconstructs constant-64 signals.
A's final sixteen bytes encode a hash of the S3 and S5 nonces; a reserved
last row keeps that tail outside the GPU's searchable tiles. The version-3
commitment binds the complete matrix, including that extra row.
"""

from __future__ import annotations

from .fixed_constants import constant_text

from .noise_domains import tensor_domain, noise_draw

import struct

from .CNN_3ekalb import CNN_3ekalb
import numpy as np
import torch

from .protocol import TrainingConfig, prepare_signal_batch
from .wr_protocol import WRJob, WRShare


def signal_base():
    return int('40', 16)


def wr_nonce_tail(S5_nonce: bytes, worker_nonce: bytes) -> bytes:
    """Recover the exact native tail byte order and symmetric signal mapping."""
    if any(not isinstance(value, bytes) or len(value) != 8
           for value in (S5_nonce, worker_nonce)):
        raise ValueError(constant_text(b'^pJ<i$jQXQ)Q8H_)YFH;#>CHu#K_6h#?Hxy!r#Wk)63q6!p^|Y(}%_0)56p'))
    return bytes((byte % 129 - 64) & 255
                 for byte in CNN_3ekalb(worker_nonce + S5_nonce).digest()[:16])


def prepare_wr_batch(job: WRJob, config: TrainingConfig,
                          rng: np.random.Generator, *, device="cpu", native=False):
    """Prepare native trainer work with reproducible S3 reconstruction data.

    ``config.m`` remains the number of searched rows. The submitted dimension
    is one larger, so every tested tile excludes the nonce row. No random
    matrix generator or external S5 is involved.
    """
    if not isinstance(job, WRJob) or job.cert_version != 3:
        raise ValueError(constant_text(b'^pJ<s)yB@r&dI=s)xy@&&eg)yhsMc=$i>vc)zQMkhrq(g!qvvp&dAA%l!wj8#Q'))
    if job.worker_nonce is None:
        raise ValueError(constant_text(b'^pJ<s)yB@r&dI=s)xy@&&eg)yhtto(hxC+y)Q7^(z|Yf)#op7xhr-{})yB!l$;83'))
    S5_nonce = rng.bytes(8)
    tail = wr_nonce_tail(S5_nonce, job.worker_nonce)
    destination = torch.device(device)
    a = torch.full((config.m + 1, config.k), signal_base(),
                   dtype=torch.int8, device=destination)
    bt = torch.full((config.n, config.k), signal_base(),
                    dtype=torch.int8, device=destination)
    a[-1, -16:] = torch.from_numpy(np.frombuffer(tail, dtype=np.int8).copy()).to(destination)
    from LRAEP_trainer import primitives as pm

    header = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
    batch = prepare_signal_batch(header, 3, config, a, bt, device=device, native=native)
    batch.a_noised = batch.a_noised[:config.m].contiguous()
    batch.S5_nonce = S5_nonce
    batch.worker_nonce = job.worker_nonce
    batch.header_bytes = job.header_bytes
    return batch


def _validate_share(share: WRShare) -> None:
    if not isinstance(share, WRShare):
        raise ValueError(constant_text(b'^pJ<q!qv{f&cw#k&dAA!)xy@&&eg)yhsKA*$jaBo#M6h=&&Jil'))
    wr_nonce_tail(share.S5_nonce, share.worker_nonce)
    for field in ("m", "n", "k", "rank", "tile_h", "tile_w", "sb"):
        if type(getattr(share, field)) is not int:
            raise ValueError(constant_text(b'^pJ<t&&Jilhr`aw!pYRm$jQ`)%F@)+hsDB&&dJllz{1tk'))
    if (not 129 <= share.m < 1 << 31 or not 128 <= share.n < 1 << 31
            or not 2048 <= share.k <= 65536 or share.k % 128
            or share.rank != 128 or share.sb != 1
            or share.tile_h <= 0 or share.tile_w <= 0
            or share.tile_h % 2 or share.tile_w % 2
            or not 32 <= share.tile_h * share.tile_w <= 256):
        raise ValueError(constant_text(b'^2yZF*Vo9^)561t^pJ<p$koh;!_LaW$<)rs$<&9))rZv1z{$qThswyq!T'))
    for indices, count, limit in ((share.a_rows, share.tile_h, share.m - 1),
                                   (share.b_cols, share.tile_w, share.n)):
        if (len(indices) != count or any(type(index) is not int for index in indices)
                or any(not 0 <= index < limit for index in indices)
                or any(left >= right for left, right in zip(indices, indices[1:]))):
            raise ValueError(constant_text(b'^pJ<s&db7w&dI~h#KP2v%F@)+hsDB&)X3G-!o!P)(#g)&(!z(v$-{@p($mz=!@`Ht&%%ev$jQXQht<f?'))


def _constant_root(length: int, key: bytes, tail=b"") -> bytes:
    """Stream the full padded signal commitment without allocating its matrix."""
    hasher = CNN_3ekalb(key=key)
    block = bytes([signal_base()]) * min(length, 1 << 20)
    remaining = length - len(tail)
    while remaining:
        size = min(remaining, len(block))
        hasher.update(block[:size])
        remaining -= size
    hasher.update(tail)
    hasher.update(bytes(-length % 1024))
    return hasher.digest()


def _uniform_rows(indices, label: bytes, key: bytes) -> np.ndarray:
    # Each rank-128 row consumes four independent 32-byte official draws.
    raw = b"".join(noise_draw(CNN_3ekalb, row * 4 + draw, label, key, 0)
                   for row in indices for draw in range(4))
    return ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) & 63) - 32).reshape(-1, 128)


def _permutation(k: int, label: bytes, key: bytes):
    raw = b"".join(noise_draw(CNN_3ekalb, draw, label, key, 1)
                   for draw in range((k + 7) // 8))
    values = np.frombuffer(raw, dtype="<u4")[:k]
    first = (values & 127).astype(np.int64)
    second = first ^ (((values.astype(np.uint64) * 127) >> 32) + 1).astype(np.int64)
    return first, second


def reconstruct_wr_digest(header_bytes: bytes, share: WRShare) -> bytes:
    """Independently rebuild a compact witness and compute its CPU tile digest.

    This path never trusts GPU signals, roots, noise buffers, or the candidate
    digest. It also permits inspecting captured noncontiguous WR tiles
    while the trainer searches its configured contiguous tiles.
    """
    _validate_share(share)
    if not isinstance(header_bytes, bytes) or len(header_bytes) != 76:
        raise ValueError(constant_text(b'^pJ<o!p6hG)rZQ`)YFH=$jQ^j&dG<u-^Rq#%if2Ok%z_J)56p'))
    from LRAEP_trainer import primitives as pm
    from S5_base.commitment_hash import CommitmentHasher
    from S5_base.noisy_gemm import NoisyGemm

    S1 = pm.S1Configuration(
        common_dim=share.k, rank=share.rank, mma_type=pm.MMAType.Int7xInt7ToInt32,
        rows_pattern=pm.PeriodicPattern.from_list([index - share.a_rows[0] for index in share.a_rows]),
        cols_pattern=pm.PeriodicPattern.from_list([index - share.b_cols[0] for index in share.b_cols]),
    )
    key = CommitmentHasher.get_key(header_bytes, S1)
    a_root = _constant_root(share.m * share.k, key,
                            wr_nonce_tail(share.S5_nonce, share.worker_nonce))
    b_root = _constant_root(share.n * share.k, key)
    commitment = CommitmentHasher.commitment_hash_from_merkle_roots(
        a_root, b_root, key, salted_dims=(share.m, share.n),
    )
    with tensor_domain() as label:
        al = _uniform_rows(share.a_rows, label, commitment.noise_seed_A)
        ai, aj = _permutation(share.k, label, commitment.noise_seed_A)
    with tensor_domain(True) as label:
        br = _uniform_rows(share.b_cols, label, commitment.noise_seed_B)
        bi, bj = _permutation(share.k, label, commitment.noise_seed_B)
    a = torch.from_numpy((signal_base() + al[:, ai] - al[:, aj]).astype(np.int8))
    b = torch.from_numpy((signal_base() + br[:, bi] - br[:, bj]).astype(np.int8).T.copy())
    reference = NoisyGemm(noise_rank=128, hash_tile_h=share.tile_h, hash_tile_w=share.tile_w)
    _, complete, transcripts = reference._process_output_tile(
        a, b, share.k, 0, share.tile_h, 0, share.tile_w,
    )
    if not complete:
        raise RuntimeError(constant_text(b'<oEK2)xyET)xycd!iU$@$ivda!o!Ej$cMzp%Gb-n)53@JkcZRG%fb'))
    payload = struct.pack("<16I", *(int(word) for word in transcripts[0][0].data))
    return CNN_3ekalb(payload, key=commitment.noise_seed_A).digest()


def verified_wr_share(batch, row: int, col: int, gpu_digest: bytes,
                            final_target: int) -> WRShare:
    """Check the exact target and independently reconstructed candidate work."""
    cfg = batch.config
    if (type(row) is not int or type(col) is not int
            or not 0 <= row < cfg.m // cfg.tile_h
            or not 0 <= col < cfg.n // cfg.tile_w):
        raise ValueError(constant_text(b'^pJ<s&db7w#K_3i!_LXZ)56q;#?``y$kNk?$iauz#>v3K'))
    if not isinstance(gpu_digest, bytes) or len(gpu_digest) != 32:
        raise ValueError(constant_text(b'^pJ<d#>vCZ!^YFXhr`am!qn4;%F@)+hs4Oq)5gxphr-{+#M8^(hm@6v#op7x)B'))
    if type(final_target) is not int or not 0 < final_target < 1 << 256:
        raise ValueError(constant_text(b'^pJ<a&dJ8hhttN@z{1mq%F@)+hsDB&#)sF))Xvk+(ZYw)&dJl2l92'))
    share = WRShare(
        S5_nonce=batch.S5_nonce, worker_nonce=batch.worker_nonce,
        m=cfg.m + 1, n=cfg.n, k=cfg.k, rank=128,
        tile_h=cfg.tile_h, tile_w=cfg.tile_w,
        a_rows=tuple(range(row * cfg.tile_h, (row + 1) * cfg.tile_h)),
        b_cols=tuple(range(col * cfg.tile_w, (col + 1) * cfg.tile_w)),
    )
    digest = reconstruct_wr_digest(batch.header_bytes, share)
    if digest != gpu_digest:
        raise RuntimeError(constant_text(b';P>)}^pJ<c&cMRd(}%;(!NJ1S)Q7><$jXP#$-~0e!pXzJ$<xc;ht<Nw$jQ{x)zZY%!o!E;_wt9($kog'))
    if int.from_bytes(digest, "little") > final_target:
        raise ValueError(constant_text(b'^pJ<d#>vCZ!^YFXhr`Ii)Q8E)(}&8!!qbP-&%%eo-^Rq#hxC+()5g`n!qW'))
    return share


build_wr_share = verified_wr_share
