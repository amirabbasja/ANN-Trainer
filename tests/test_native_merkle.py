"""CUDA commitment roots against PEARL's unmodified CPU Merkle helper."""

from pathlib import Path
import unittest

import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class NativeMerkleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from LRAEP_trainer.protocol import load_official_helpers

        root = Path(__file__).resolve().parents[1]
        try:
            load_official_helpers(root / "src" / "LRAEP_helpers")
            from miner_base.matrix_merkle_tree import MatrixMerkleTree
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"official helper unavailable: {exc}") from None
        from LRAEP_trainer import native_cuda

        native_cuda.load_native("cuda:0")
        cls.native = native_cuda
        cls.reference = MatrixMerkleTree
        cls.device = torch.device("cuda:0")

    def assert_root_matches(self, signal, key):
        expected = self.reference(signal.cpu(), key).root
        actual = self.native.merkle_root(signal.to(self.device), key)
        self.assertIsInstance(actual, bytes)
        self.assertEqual(len(actual), 32)
        self.assertEqual(actual, expected)

    def test_single_chunk_roots_match_official_zero_padding_at_block_boundaries(self):
        rng = np.random.default_rng(26319)
        for size in (1, 63, 64, 65, 1023, 1024):
            with self.subTest(size=size):
                signal = torch.from_numpy(rng.integers(-128, 128, (1, size), dtype=np.int8))
                signal[0, -1] = 101
                self.assert_root_matches(signal, bytes(range(32)))

    def test_odd_chunk_counts_and_partial_final_chunk_match_official_tree(self):
        rng = np.random.default_rng(6831)
        for size in (1025, 3 * 1024, 3 * 1024 - 1,
                     4 * 1024 + 13, 5 * 1024, 8 * 1024 - 17):
            with self.subTest(size=size):
                signal = torch.from_numpy(rng.integers(-128, 128, (1, size), dtype=np.int8))
                self.assert_root_matches(signal, bytes(reversed(range(32))))

    def test_training_matrix_roots_match_across_signal_seeds_and_keys(self):
        keys = (bytes(32), bytes([255]) * 32, bytes(range(32)))
        for seed, key in enumerate(keys, 280):
            with self.subTest(seed=seed, key=key[:2]):
                rng = np.random.default_rng(seed)
                signal = torch.from_numpy(rng.integers(-64, 64, (128, 2048), dtype=np.int8))
                self.assert_root_matches(signal, key)

    def test_noncontiguous_matrix_uses_logical_row_major_bytes(self):
        rng = np.random.default_rng(913)
        signal = torch.from_numpy(rng.integers(-128, 128, (129, 65), dtype=np.int8)).T
        self.assertFalse(signal.is_contiguous())
        self.assert_root_matches(signal, bytes([0xA7]) * 32)

    def test_contiguous_unaligned_storage_offset_matches_without_misaligned_loads(self):
        storage = torch.arange(2050, dtype=torch.int32).to(torch.int8).to(self.device)
        signal = storage[1:2050].view(1, -1)
        self.assertTrue(signal.is_contiguous())
        self.assertEqual(signal.data_ptr() % 4, 1)
        self.assert_root_matches(signal, bytes([0x3D]) * 32)

    def test_nondefault_stream_orders_commitment_work_after_signal_generation(self):
        stream = torch.cuda.Stream(device=self.device)
        signal = torch.arange(1537, dtype=torch.int32).to(torch.int8).reshape(1, -1)
        expected = self.reference(signal, bytes([0x6B]) * 32).root
        with torch.cuda.stream(stream):
            device_signal = signal.to(self.device)
            actual = self.native.merkle_root(device_signal, bytes([0x6B]) * 32)
        self.assertEqual(actual, expected)

    def test_invalid_signal_and_key_inputs_are_rejected(self):
        valid = torch.zeros((1, 1024), dtype=torch.int8, device=self.device)
        invalid = (
            valid.cpu(),
            valid.to(torch.float32),
            valid.flatten(),
            torch.empty((0, 1024), dtype=torch.int8, device=self.device),
        )
        for signal in invalid:
            with self.subTest(shape=tuple(signal.shape), dtype=signal.dtype, device=signal.device):
                with self.assertRaises(ValueError):
                    self.native.merkle_root(signal, bytes(32))
        for length in (0, 31, 33):
            with self.subTest(key_length=length), self.assertRaises(ValueError):
                self.native.merkle_root(valid, bytes(length))


if __name__ == "__main__":
    unittest.main()
