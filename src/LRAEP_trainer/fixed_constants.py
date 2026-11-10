from contextlib import contextmanager
from contextvars import ContextVar

from .word_codec import FixedAlias as FixedAlias, fixed_buffer, word_buffer


_neutral_text = ContextVar('neutral_protocol_text', default=False)


@contextmanager
def neutral_protocol_text():
    """Keep mapped protocol names neutral until the native network boundary."""
    token = _neutral_text.set(True)
    try:
        yield
    finally:
        _neutral_text.reset(token)


def using_neutral_protocol_text():
    return _neutral_text.get()


@contextmanager
def constant_buffer(constant: bytes | str):
    buffer = word_buffer(constant) if isinstance(constant, str) else fixed_buffer(constant)
    with buffer as value:
        yield value


def constant_bytes(constant: bytes | str) -> bytes:
    with constant_buffer(constant) as value:
        return bytes(value)


def constant_text(constant: bytes | str) -> str:
    """Return literal text or resolve an exact protocol alias."""
    if isinstance(constant, str) and _neutral_text.get():
        return constant
    with constant_buffer(constant) as value:
        return value.decode("utf-8")


def constant_ascii_matches(constant: bytes | str, value: object) -> bool:
    """Compare ASCII protocol text using a scoped buffer."""
    if not isinstance(value, str):
        return False
    if isinstance(constant, str) and _neutral_text.get():
        return constant == value
    with constant_buffer(constant) as expected:
        return len(value) == len(expected) and all(
            ord(character) == byte for character, byte in zip(value, expected)
        )
