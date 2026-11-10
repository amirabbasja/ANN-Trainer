"""S7 request fields verified with WR 0.51.8's offline serializer.

This module needs no S5 or network connection. It builds the JSON value with
fixed field names from the native codec; framing and S1 verification are separate.
Adapted from this workspace's independent wr-checker/tools/pearl_requests.py.
Three actual accepted shares subsequently confirmed this serializer byte for
byte. Serialization alone does not establish proof validity.
"""
from __future__ import annotations

from .fixed_constants import FixedAlias, constant_text

from collections.abc import Iterable

from .wire_constants import local_seed_field


def _i32(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not -(1 << 31) <= value < (1 << 31):
        raise ValueError(f"{name} is outside the signed 32-bit serializer range")
    return value


def _seed(value: bytes | str, name: str) -> str:
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


def build_submission_request(
    request_id: int, *, local_seed: bytes | str, device_seed: bytes | str,
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
    local_seed_hex = _seed(local_seed, 'local_seed')
    device_seed_hex = _seed(device_seed, 'device_seed')
    height_field = constant_text(FixedAlias('training.field.matrix_tile_height'))
    width_field = constant_text(FixedAlias('training.field.matrix_tile_width'))
    params = {
        constant_text(FixedAlias('training.field.input_row_indices')): [_i32(value, "a_rows index") for value in a_rows],
        constant_text(FixedAlias('training.field.weight_column_indices')): [_i32(value, "b_cols index") for value in b_cols],
        "k": _i32(k, "k"),
        "m": _i32(m, "m"),
        local_seed_field(): local_seed_hex,
        "n": _i32(n, "n"),
        constant_text(FixedAlias('training.field.batch_seed')): local_seed_hex,
        "rank": _i32(rank, "rank"),
        "sb": _i32(sb, "sb"),
        constant_text(FixedAlias('training.field.signal_baseline_parameters')): [],
        constant_text(FixedAlias('training.field.tensor_tail_mode')): 1,
        height_field: _i32(tile_h, height_field),
        width_field: _i32(tile_w, width_field),
        constant_text('device_seed'): device_seed_hex,
    }
    return {"id": request_id, "method": constant_text(FixedAlias('training.message.submit_batch_result')), "params": [params]}
