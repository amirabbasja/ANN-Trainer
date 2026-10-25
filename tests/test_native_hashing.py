"""Native GPU keyed BLAKE3 parity and exact target/capacity boundaries."""

import ctypes
import unittest

from blake3 import blake3
import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class NativeHashingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pearl_trainer.native_cuda import load_native

        cls.library = load_native()
        if cls.library is None:
            raise unittest.SkipTest("native CUDA helper unavailable")
        cls.hash_candidates = cls.library.pearl_hash_candidates
        cls.hash_candidates.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p,
        ]
        cls.hash_candidates.restype = ctypes.c_int
        cls.device = torch.device("cuda:0")

    def hash(self, transcripts, key, target, capacity=None, stream=None):
        count = transcripts.shape[0]
        if capacity is None:
            capacity = count
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        with torch.cuda.stream(stream):
            messages = torch.from_numpy(np.ascontiguousarray(transcripts, dtype=np.uint32)).to(self.device)
            key_words = torch.from_numpy(np.frombuffer(key, dtype="<u4").copy()).to(self.device)
            target_words = torch.from_numpy(
                np.frombuffer(target.to_bytes(32, "little"), dtype="<u4").copy()
            ).to(self.device)
            # Guard rows remain untouched even when every tile matches and
            # the returned total exceeds the available output capacity.
            indices = torch.full((capacity + 4,), -1, dtype=torch.int32, device=self.device)
            digests = torch.full((capacity + 4, 32), 0xA5, dtype=torch.uint8, device=self.device)
            total = torch.tensor([12345], dtype=torch.int32, device=self.device)
            error = self.hash_candidates(
                messages.data_ptr(), count, key_words.data_ptr(), target_words.data_ptr(),
                indices.data_ptr(), digests.data_ptr(), total.data_ptr(), capacity,
                stream.cuda_stream,
            )
            self.assertEqual(error, 0)
        stream.synchronize()
        found = int(total.cpu()[0])
        saved = min(capacity, found)
        observed = {
            int(index): bytes(digest)
            for index, digest in zip(indices[:saved].cpu().tolist(), digests[:saved].cpu().tolist())
        }
        self.assertEqual(len(observed), saved, "matching tile indices must be unique")
        self.assertTrue(bool(torch.all(indices[capacity:] == -1)))
        self.assertTrue(bool(torch.all(digests[capacity:] == 0xA5)))
        return found, observed

    @staticmethod
    def reference(transcripts, key, target):
        result = {}
        for index, words in enumerate(transcripts):
            digest = blake3(words.astype("<u4").tobytes(), key=key).digest()
            if int.from_bytes(digest, "little") <= target:
                result[index] = digest
        return result

    def test_all_digests_match_python_across_random_keys_messages_and_block_edges(self):
        rng = np.random.default_rng(71921)
        for count in (1, 31, 128, 129, 513):
            with self.subTest(count=count):
                messages = rng.integers(0, 1 << 32, (count, 16), dtype=np.uint32)
                key = rng.bytes(32)
                target = (1 << 256) - 1
                total, actual = self.hash(messages, key, target)
                self.assertEqual(total, count)
                self.assertEqual(actual, self.reference(messages, key, target))

    def test_target_filters_all_digest_words_in_little_endian_order(self):
        rng = np.random.default_rng(538)
        messages = rng.integers(0, 1 << 32, (257, 16), dtype=np.uint32)
        key = rng.bytes(32)
        for target in (0, (1 << 128) - 1, (1 << 255) + 12345):
            with self.subTest(target_bits=target.bit_length()):
                total, actual = self.hash(messages, key, target)
                expected = self.reference(messages, key, target)
                self.assertEqual(total, len(expected))
                self.assertEqual(actual, expected)

    def test_exact_digest_target_is_inclusive_and_one_less_is_rejected(self):
        messages = np.arange(16, dtype=np.uint32).reshape(1, 16)
        key = bytes(range(32))
        digest = blake3(messages.astype("<u4").tobytes(), key=key).digest()
        boundary = int.from_bytes(digest, "little")
        self.assertEqual(self.hash(messages, key, boundary), (1, {0: digest}))
        self.assertEqual(self.hash(messages, key, boundary - 1), (0, {}))

    def test_overflow_counts_every_match_and_preserves_output_bounds(self):
        messages = np.arange(513 * 16, dtype=np.uint32).reshape(513, 16)
        key = bytes([0xD3]) * 32
        expected = self.reference(messages, key, (1 << 256) - 1)
        for capacity in (0, 1, 7, 128):
            with self.subTest(capacity=capacity):
                total, actual = self.hash(messages, key, (1 << 256) - 1, capacity)
                self.assertEqual(total, 513)
                self.assertEqual(len(actual), capacity)
                for index, digest in actual.items():
                    self.assertEqual(digest, expected[index])

    def test_empty_input_resets_counter_without_launching_or_writing_outputs(self):
        self.assertEqual(self.hash(np.empty((0, 16), dtype=np.uint32), bytes(32), 0), (0, {}))

    def test_nondefault_stream_obeys_input_output_order(self):
        messages = np.arange(129 * 16, dtype=np.uint32).reshape(129, 16)
        key = bytes(reversed(range(32)))
        stream = torch.cuda.Stream(device=self.device)
        total, actual = self.hash(messages, key, (1 << 256) - 1, stream=stream)
        self.assertEqual(total, 129)
        self.assertEqual(actual, self.reference(messages, key, (1 << 256) - 1))

    def test_invalid_counts_and_required_null_pointers_fail_before_launch(self):
        buffer = torch.empty(32, dtype=torch.int32, device=self.device)
        pointer = buffer.data_ptr()
        stream = torch.cuda.current_stream(self.device).cuda_stream
        valid = [pointer, 1, pointer, pointer, pointer, pointer, pointer, 1, stream]
        for index, value in ((1, -1), (7, -1), (0, None), (2, None),
                             (3, None), (4, None), (5, None), (6, None)):
            with self.subTest(argument=index):
                arguments = valid.copy()
                arguments[index] = value
                self.assertNotEqual(self.hash_candidates(*arguments), 0)
        torch.cuda.current_stream(self.device).synchronize()


if __name__ == "__main__":
    unittest.main()
