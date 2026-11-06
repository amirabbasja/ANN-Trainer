"""S7 request fields verified with WR 0.51.8's offline serializer.

This module needs no S5 or network connection. It builds the JSON value with
fixed field names from the native codec; framing and S1 verification are separate.
Adapted from this workspace's independent wr-checker/tools/pearl_requests.py.
Three actual accepted shares subsequently confirmed this serializer byte for
byte. Serialization alone does not establish proof validity.
"""
from __future__ import annotations

from .fixed_constants import constant_text

from collections.abc import Iterable

from .wire_constants import s5_nonce


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
    request_id: int, *, S5_nonce: bytes | str, worker_nonce: bytes | str,
    m: int, n: int, k: int, rank: int, tile_h: int, tile_w: int, sb: int,
    a_rows: Iterable[int], b_cols: Iterable[int],
) -> dict:
    """Reproduce the observed share serializer's exact field names and types.

    Nonces are byte-order-preserving 8-byte values or 16-character hex strings.
    The native serializer emits ``nonce`` as an alias of the S5 nonce field, and
    always emits ``tail_mode: 1`` and ``sb_p: []`` for this work structure. Its
    scalar and index storage is signed int32. ``sb`` is retained by its native
    name; no unverified interpretation is imposed here.

    This checks representability, not whether dimensions, tile indices, or the
    resulting PoUW solution satisfy the current S3 job. No job ID, S2, or
    base64 proof is added because none appears in the verified serializer.
    """
    if (isinstance(request_id, bool) or not isinstance(request_id, int)
            or not 0 <= request_id < (1 << 64)):
        raise ValueError("request_id must be an unsigned 64-bit integer")
    S5 = _nonce(S5_nonce, constant_text(b'^pg0=$jQXQ'))
    worker = _nonce(worker_nonce, constant_text('device_counter'))
    params = {
        constant_text(b'#`x99(9{'): [_i32(value, "a_rows index") for value in a_rows],
        constant_text(b'#rVX?%hU'): [_i32(value, "b_cols index") for value in b_cols],
        "k": _i32(k, "k"),
        "m": _i32(m, "m"),
        s5_nonce(): S5,
        "n": _i32(n, "n"),
        constant_text(b'$;ipX!T'): S5,
        "rank": _i32(rank, "rank"),
        "sb": _i32(sb, "sb"),
        constant_text(b')W!JM'): [],
        constant_text(b')5gxr_{zw`!T'): 1,
        constant_text(b')6UDn_|E'): _i32(tile_h, constant_text(b')6UDn_|E')),
        constant_text(b')6UDn_|O'): _i32(tile_w, constant_text(b')6UDn_|O')),
        constant_text('device_counter'): worker,
    }
    return {"id": request_id, "method": constant_text(b')Y8Su&eH'), "params": [params]}
