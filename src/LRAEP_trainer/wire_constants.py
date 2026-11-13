"""Neutral training names backed by the native compatibility vocabulary."""

from .fixed_constants import constant_bytes, constant_text
from .word_codec import word_buffer

WR_CAPTURED_SHAPE = (4096, 262144, 8192, 8, 16)


def has_identity_mode_prefix(value):
    """Check the peer's unsupported identity mode without retaining its text."""
    if not isinstance(value, str):
        return False
    with word_buffer('identity_mode_prefix') as prefix:
        return len(value) >= len(prefix) and all(
            ord(value[index]) == byte for index, byte in enumerate(prefix)
        )


def local_seed_field():
    return constant_text('local_seed')


def s6_noise_seed_prefix():
    return constant_bytes('noise_seed_domain')


def s7_legacy_alias():
    return constant_text('legacy_protocol')
