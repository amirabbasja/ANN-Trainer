from contextlib import contextmanager
import hashlib
import unittest
from unittest.mock import patch

from LRAEP_trainer import fixed_constants
from LRAEP_trainer.fixed_constants import FixedAlias, constant_ascii_matches, constant_buffer, constant_bytes, constant_text


class FixedConstantTests(unittest.TestCase):
    def test_exact_binary_values_and_immutable_api(self):
        encoded = FixedAlias('training.fixture.byte_sweep')
        expected = bytes(range(256))
        self.assertEqual(constant_bytes(encoded), expected)
        with constant_buffer(encoded) as value:
            self.assertIsInstance(value, bytearray)
            self.assertEqual(value, expected)
            digest = hashlib.sha256(value).digest()
        self.assertEqual(digest, hashlib.sha256(expected).digest())
        self.assertEqual(value, bytearray(256))

    def test_cleanup_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with constant_buffer(FixedAlias('training.fixture.text_sample')) as value:
                self.assertEqual(value, b'abc')
                raise RuntimeError("consumer failed")
        self.assertEqual(value, bytearray(3))

    def test_text_and_empty_values(self):
        alias = FixedAlias('training.fixture.text_sample')
        self.assertEqual(constant_text(alias), 'abc')
        self.assertEqual(constant_bytes(b''), b'')
        self.assertEqual(constant_text(b''), '')
        with fixed_constants.neutral_protocol_text():
            self.assertEqual(constant_text('account'), 'account')
            self.assertEqual(constant_text(alias), 'abc')
            self.assertEqual(constant_bytes(bytes(alias)), bytes(alias))

    def test_native_word_buffer_is_cleared_after_success_and_exception(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                try:
                    with constant_buffer('account') as value:
                        self.assertEqual(value, b'wallet')
                        if fail:
                            raise RuntimeError('consumer failed')
                except RuntimeError:
                    if not fail:
                        raise
                self.assertEqual(value, bytearray(6))

    def test_ascii_comparisons_clear_temporary_native_buffers(self):
        buffers = []

        @contextmanager
        def observe(constant):
            with constant_buffer(constant) as buffer:
                buffers.append(buffer)
                yield buffer

        class BrokenText(str):
            def __iter__(self):
                raise RuntimeError('comparison failed')

        with patch.object(fixed_constants, 'constant_buffer', observe):
            self.assertTrue(constant_ascii_matches('account', 'wallet'))
            for candidate in ('wallEt', 'walle', 'wall\u00e9t', None, [], b'wallet'):
                self.assertFalse(constant_ascii_matches('account', candidate))
            self.assertTrue(constant_ascii_matches(FixedAlias('training.fixture.text_sample'), 'abc'))
            with self.assertRaisesRegex(RuntimeError, 'comparison failed'):
                constant_ascii_matches('account', BrokenText('wallet'))
        self.assertTrue(buffers)
        self.assertTrue(all(not any(buffer) for buffer in buffers))
