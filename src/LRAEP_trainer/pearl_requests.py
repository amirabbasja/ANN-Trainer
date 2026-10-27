"""Pearlhash request fields verified with WR 0.51.8's offline serializer.

This module needs no miner, native library, or network connection. It builds the
JSON value only; framing/encoding and mining-result verification are separate.
Adapted from this workspace's independent wr-checker/tools/pearl_requests.py.
Three actual accepted shares subsequently confirmed this serializer byte for
byte. Serialization alone does not establish proof validity.
"""
from __future__ import annotations

from collections.abc import Iterable


def _i32(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not -(1 << 31) <= value < (1 << 31):
        raise ValueError(f"{name} is outside the signed 32-bit serializer range")
    return value


def _nonce(value: bytes | str, name: str) -> str:
    if isinstance(value, str):
        if len(value) != 16:
            raise ValueError(f"{name} must be exactly 16 hex characters")
        try:
            value = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be hex") from exc
    if not isinstance(value, bytes) or len(value) != 8:
        raise ValueError(f"{name} must be exactly eight bytes")
    return value.hex()


def build_submit_request(
    request_id: int, *, miner_nonce: bytes | str, worker_nonce: bytes | str,
    m: int, n: int, k: int, rank: int, tile_h: int, tile_w: int, sb: int,
    a_rows: Iterable[int], b_cols: Iterable[int],
) -> dict:
    """Reproduce the observed share serializer's exact field names and types.

    Nonces are byte-order-preserving 8-byte values or 16-character hex strings.
    The native serializer emits ``nonce`` as an alias of ``miner_nonce``, and
    always emits ``tail_mode: 1`` and ``sb_p: []`` for this work structure. Its
    scalar and index storage is signed int32. ``sb`` is retained by its native
    name; no unverified interpretation is imposed here.

    This checks representability, not whether dimensions, tile indices, or the
    resulting PoUW solution satisfy the current pool job. No job ID, wallet, or
    base64 proof is added because none appears in the verified serializer.
    """
    if (isinstance(request_id, bool) or not isinstance(request_id, int)
            or not 0 <= request_id < (1 << 64)):
        raise ValueError("request_id must be an unsigned 64-bit integer")
    miner = _nonce(miner_nonce, "miner_nonce")
    worker = _nonce(worker_nonce, "worker_nonce")
    params = {
        "a_rows": [_i32(value, "a_rows index") for value in a_rows],
        "b_cols": [_i32(value, "b_cols index") for value in b_cols],
        "k": _i32(k, "k"),
        "m": _i32(m, "m"),
        "miner_nonce": miner,
        "n": _i32(n, "n"),
        "nonce": miner,
        "rank": _i32(rank, "rank"),
        "sb": _i32(sb, "sb"),
        "sb_p": [],
        "tail_mode": 1,
        "tile_h": _i32(tile_h, "tile_h"),
        "tile_w": _i32(tile_w, "tile_w"),
        "worker_nonce": worker,
    }
    return {"id": request_id, "method": "submit", "params": [params]}
