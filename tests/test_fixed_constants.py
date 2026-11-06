from contextlib import contextmanager
import hashlib
import unittest
from unittest.mock import patch

from LRAEP_trainer import fixed_constants
from LRAEP_trainer.fixed_constants import constant_ascii_matches, constant_buffer, constant_bytes, constant_text


class FixedConstantTests(unittest.TestCase):
    def test_exact_binary_values_and_immutable_api(self):
        encoded = b'r>3Q(qoSdpudc1EtE#D}x3;ykv$C<UzrMY^ySll!hlYiOgMxv8kB*Iui;9VemzI^3lai5;pPrqZo0^%J=jP?)<Kp4q@9ypF>+0#~_xAPl^YZcV|Ni~_`}+C#$Hv9P!@|M9&(6)v%gV{f*Vfh4)6&t<-`?Ha+uGUKCnhB%BO)OnFD@-CD=H}{H#RjiGcqwSKR!J?J32Wy2L=TM0|Eg64-O3s3knGc7Zw#16A}>+A08bX8yXoHXJ%z&V`5=oZ*FaDYiem|cXoAjb8>NTe|~*@dwO|zM@B_NLqb77PfkrtOG-&dS5{S2Q&Le-UtV2YTUuFI'
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
            with constant_buffer(b'#>K<') as value:
                self.assertEqual(value, b'abc')
                raise RuntimeError("consumer failed")
        self.assertEqual(value, bytearray(3))

    def test_text_and_empty_values(self):
        self.assertEqual(constant_text(b'#>K<'), 'abc')
        self.assertEqual(constant_bytes(b''), b'')
        self.assertEqual(constant_text(b''), '')

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
        def observe(encoded):
            with constant_buffer(encoded) as buffer:
                buffers.append(buffer)
                yield buffer

        class BrokenText(str):
            def __iter__(self):
                raise RuntimeError('comparison failed')

        with patch.object(fixed_constants, 'constant_buffer', observe):
            self.assertTrue(constant_ascii_matches('account', 'wallet'))
            for candidate in ('wallEt', 'walle', 'wall\u00e9t', None, [], b'wallet'):
                self.assertFalse(constant_ascii_matches('account', candidate))
            self.assertTrue(constant_ascii_matches(b'#>K<', 'abc'))
            with self.assertRaisesRegex(RuntimeError, 'comparison failed'):
                constant_ascii_matches('account', BrokenText('wallet'))
        self.assertTrue(buffers)
        self.assertTrue(all(not any(buffer) for buffer in buffers))
