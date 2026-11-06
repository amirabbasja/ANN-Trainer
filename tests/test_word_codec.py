import base64
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import unittest

from LRAEP_trainer import word_codec


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    'training': 'mining',
    'train': 'mine',
    'account': 'wallet',
    'account_id': 'wallet_address',
    'connectionRule': 'stratum',
    'p_goal': 'pearl',
    'legacyMethod': 'pearlhash',
    'newerMethod': 'kryptex',
    'prochPerSecond': 'hashrate',
    'epoch': 'hash',
    'extracounter': 'extranonce',
    'walletnonce': 'w_counter',
    'trainer_counter': 'miner_nonce',
    'device_counter': 'worker_nonce',
    'training.subscribe': 'mining.subscribe',
    'training.authorize': 'mining.authorize',
    'training.submit': 'mining.submit',
    'training.notify': 'mining.notify',
    'training.reconnect': 'mining.reconnect',
    'noise_seed_domain': 'pearl/cert-v3/noise-seed/',
    'account_prefix': 'prl1',
    'device_default': 'rig-01',
    'record_id': 'txid',
    'parent_digest': 'previousblockhash',
    'size_limit': 'vsizelimit',
    'records': 'transactions',
    'poll_id': 'longpollid',
    'counter_range': 'noncerange',
    'reward_metadata': 'coinbaseaux',
    'reward_amount': 'coinbasevalue',
    'default_attestation': 'default_witness_commitment',
    'required_version': 'requiredcertversion',
    'template_request': 'getblocktemplate',
    'tip_request': 'getbestblockhash',
    'submit_record': 'submitblock',
}


class NativeWordCodecTests(unittest.TestCase):
    def test_required_mappings_in_both_directions(self):
        for placeholder, value in REQUIRED.items():
            with self.subTest(placeholder=placeholder):
                self.assertEqual(word_codec.transform_word(placeholder, 'decode'), value)
                self.assertEqual(word_codec.transform_word(value, 'encode'), placeholder)
                self.assertEqual(word_codec.decode_word(placeholder), value)
                self.assertEqual(word_codec.encode_word(value), placeholder)
                self.assertTrue(word_codec.word_matches(placeholder, value))
                self.assertFalse(word_codec.word_matches(placeholder, value + '!'))

    def test_unknown_words_are_exact_case_sensitive_pass_through(self):
        for value in ('', 'unknown', 'TRAINING', 'account.extra', 'training_extra', 'abc\0def', '\u0645\u0642\u062f\u0627\u0631', '\U0001f512', '\ud800'):
            for method in ('encode', 'decode'):
                with self.subTest(value=repr(value), method=method):
                    self.assertEqual(word_codec.transform_word(value, method), value)

    def test_invalid_method_and_non_string_word_are_rejected(self):
        for method in ('', 'Encode', 'other', None, [], 1):
            with self.subTest(method=method), self.assertRaises(ValueError):
                word_codec.transform_word('account', method)
        for value in (None, b'account', [], 1):
            with self.subTest(value=value), self.assertRaises(TypeError):
                word_codec.decode_word(value)
        self.assertFalse(word_codec.word_matches('account', None))

    def test_word_buffer_is_resizable_and_wiped_after_use_and_exception(self):
        for fail in (False, True):
            try:
                with word_codec.word_buffer('account') as value:
                    self.assertEqual(value, b'wallet')
                    value.extend(b'-additional')
                    size = len(value)
                    if fail:
                        raise RuntimeError('consumer failed')
            except RuntimeError as error:
                self.assertEqual(str(error), 'consumer failed')
            self.assertEqual(value, bytearray(size))
        with word_codec.word_buffer('') as empty:
            self.assertEqual(empty, bytearray())
            empty.extend(b'added')
        self.assertEqual(empty, bytearray(5))

    def test_fixed_buffers_match_existing_binary_transform_and_allow_resize(self):
        rng = random.Random(0)
        for length in range(261):
            original = rng.randbytes(length)
            encoded = base64.b85encode(bytes(byte ^ 0xA7 for byte in original))
            with self.subTest(length=length), word_codec.fixed_buffer(encoded) as value:
                self.assertEqual(value, original)
                value.extend(b'!')
                size = len(value)
            self.assertEqual(value, bytearray(size))
        with word_codec.fixed_buffer(b'0') as value:
            self.assertEqual(value, b'')
        try:
            with word_codec.fixed_buffer(b'#>K<') as value:
                value.extend(b'-more')
                raise RuntimeError('consumer failed')
        except RuntimeError:
            pass
        self.assertEqual(value, bytearray(8))

    def test_invalid_fixed_encoding_is_rejected(self):
        for encoded in (b' ', b'~~~~~', b'\x00', b'abcde '):
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                with word_codec.fixed_buffer(encoded):
                    self.fail('invalid encoding accepted')

    def test_native_capacity_and_mode_errors_do_not_overwrite_output(self):
        library = word_codec._library()
        output = ctypes.create_string_buffer(b'guard')
        written = ctypes.c_size_t()
        status = library.wr_codec_transform(ctypes.c_char_p(b'account'), 7, 0, output, 1, ctypes.byref(written))
        self.assertEqual(status, 2)
        self.assertEqual(written.value, 6)
        self.assertEqual(output.value, b'guard')
        self.assertEqual(library.wr_codec_transform(ctypes.c_char_p(b'account'), 7, 2, output, 6, ctypes.byref(written)), 3)
        self.assertEqual(output.value, b'guard')

    def test_concurrent_transformations_do_not_share_mutable_results(self):
        pairs = list(REQUIRED.items()) * 10

        def check(pair):
            placeholder, value = pair
            return word_codec.decode_word(placeholder), word_codec.encode_word(value)

        with ThreadPoolExecutor(max_workers=8) as executor:
            actual = list(executor.map(check, pairs))
        self.assertEqual(actual, [(value, placeholder) for placeholder, value in pairs])

    def test_compiled_source_fingerprint_and_deployment_check(self):
        expected = hashlib.sha256((ROOT / 'native/word_codec.cpp').read_bytes()).hexdigest()
        self.assertEqual(word_codec._library().wr_codec_source_id().decode(), expected)
        result = subprocess.run([sys.executable, str(ROOT / 'tools/build_word_codec.py'), '--check'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
