"""Compare optimized preparation with the unchanged official CPU noise path."""

from pathlib import Path
import unittest

import numpy as np
import torch

from LRAEP_trainer.protocol import TrainingConfig, load_official_helpers, prepare_batch


class FullRangeSignals:
    """Arithmetic stress inputs; ordinary mining signals remain int7."""

    def __init__(self):
        self.calls = 0

    def integers(self, low, high, size, dtype):
        assert (low, high, dtype) == (-64, 64, np.int8)
        values = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
        if self.calls:
            values = values[::-1]
        self.calls += 1
        return np.resize(values, size).copy()


class PreparationReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        try:
            load_official_helpers(root / "src" / "LRAEP_helpers")
            from LRAEP_trainer import primitives as pm
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"official helper unavailable: {exc}") from None
        cls.pm = pm
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)

    def setUp(self):
        # Asymmetric dimensions expose incorrect B transposition or V3 salting.
        self.cfg = TrainingConfig(m=128, n=256, k=2048)
        self.header = self.pm.IncompleteBlockHeader(
            version=1, prev_block=bytes(range(32)), merkle_root=bytes(reversed(range(32))),
            timestamp=1720000000, nbits=0x1D00FFFF,
        )

    def assert_signal_sequence(self, batch, seed):
        rng = np.random.default_rng(seed)
        expected_a = rng.integers(-64, 64, (self.cfg.m, self.cfg.k), dtype=np.int8)
        expected_bt = rng.integers(-64, 64, (self.cfg.n, self.cfg.k), dtype=np.int8)
        np.testing.assert_array_equal(batch.a.numpy(), expected_a)
        np.testing.assert_array_equal(batch.bt.numpy(), expected_bt)

    def assert_official_reference(self, batch, version, device_type):
        from miner_base.commitment_hash import CommitmentHasher
        from miner_base.matrix_merkle_tree import MatrixMerkleTree
        from miner_base.noise_generation import NoiseGenerator

        self.assertEqual(batch.a.device.type, "cpu")
        self.assertEqual(batch.bt.device.type, "cpu")
        self.assertEqual(batch.a.dtype, torch.int8)
        self.assertEqual(batch.bt.dtype, torch.int8)
        key = CommitmentHasher.get_key(bytes(self.header.to_bytes()), self.cfg.official_config())
        a_tree = MatrixMerkleTree(batch.a, key)
        b_tree = MatrixMerkleTree(batch.bt, key)
        self.assertEqual(batch.a_tree.root, a_tree.root)
        self.assertEqual(batch.b_tree.root, b_tree.root)
        commitment = CommitmentHasher.commitment_hash_from_merkle_roots(
            a_tree.root, b_tree.root, key,
            salted_dims=(self.cfg.m, self.cfg.n) if version == 3 else None,
        )
        self.assertEqual(batch.commitment.noise_seed_A, commitment.noise_seed_A)
        self.assertEqual(batch.commitment.noise_seed_B, commitment.noise_seed_B)

        # This is the pinned official implementation, not the new adapter.
        al, ar, bl, br = NoiseGenerator(noise_rank=128).generate_noise_metrices(
            commitment.noise_seed_A, commitment.noise_seed_B,
            self.cfg.m, self.cfg.k, self.cfg.n,
        )
        raw_a = batch.a.to(torch.int32) + al.to(torch.int32) @ ar.to(torch.int32)
        raw_b = batch.bt.T.to(torch.int32) + bl.to(torch.int32) @ br.to(torch.int32)
        expected_a, expected_b = raw_a.to(torch.int8), raw_b.to(torch.int8)
        for actual, expected in ((batch.a_noised, expected_a), (batch.b_noised, expected_b)):
            self.assertEqual(actual.device.type, device_type)
            self.assertEqual(actual.dtype, torch.int8)
            self.assertTrue(actual.is_contiguous())
            self.assertTrue(torch.equal(actual.cpu(), expected))
        return raw_a, raw_b

    def test_default_cpu_path_matches_official_reference_for_all_versions(self):
        for version in (1, 2, 3):
            with self.subTest(version=version):
                seed = 1000 + version
                batch = prepare_batch(self.header, version, self.cfg, np.random.default_rng(seed))
                self.assert_signal_sequence(batch, seed)
                self.assert_official_reference(batch, version, "cpu")

    def test_explicit_cpu_device_preserves_default_call_behavior(self):
        seed = 20261025
        default = prepare_batch(self.header, 3, self.cfg, np.random.default_rng(seed))
        explicit = prepare_batch(self.header, 3, self.cfg, np.random.default_rng(seed), device="cpu")
        for field in ("a", "bt", "a_noised", "b_noised"):
            self.assertTrue(torch.equal(getattr(default, field), getattr(explicit, field)))
        self.assertEqual(default.a_tree.root, explicit.a_tree.root)
        self.assertEqual(default.b_tree.root, explicit.b_tree.root)
        self.assertEqual(default.commitment.noise_seed_A, explicit.commitment.noise_seed_A)
        self.assertEqual(default.commitment.noise_seed_B, explicit.commitment.noise_seed_B)

    def test_cuda_noise_matches_official_cpu_reference_for_all_versions(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        for version in (1, 2, 3):
            with self.subTest(version=version):
                seed = 2000 + version
                batch = prepare_batch(
                    self.header, version, self.cfg, np.random.default_rng(seed), device="cuda:0",
                )
                self.assert_signal_sequence(batch, seed)
                self.assert_official_reference(batch, version, "cuda")

    def test_full_range_noise_addition_preserves_int8_wrap(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda:0")
        for device in devices:
            with self.subTest(device=device):
                batch = prepare_batch(self.header, 3, self.cfg, FullRangeSignals(), device=device)
                self.assertEqual(int(batch.a.min()), -128)
                self.assertEqual(int(batch.a.max()), 127)
                raw_a, raw_b = self.assert_official_reference(
                    batch, 3, torch.device(device).type,
                )
                self.assertTrue(bool(((raw_a < -128) | (raw_a > 127)).any()))
                self.assertTrue(bool(((raw_b < -128) | (raw_b > 127)).any()))

    def test_gpu_generated_signals_match_official_noise_for_all_versions(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        for version in (1, 2, 3):
            with self.subTest(version=version):
                batch = prepare_batch(
                    self.header, version, self.cfg, np.random.default_rng(3000 + version),
                    device="cuda:0", gpu_signals=True,
                )
                for signal in (batch.a, batch.bt):
                    self.assertGreaterEqual(int(signal.min()), -64)
                    self.assertLessEqual(int(signal.max()), 63)
                self.assert_official_reference(batch, version, "cuda")

    def test_gpu_signal_generation_is_reproducible_and_preserves_global_rng(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")
        before = torch.cuda.get_rng_state(0).clone()
        first = prepare_batch(self.header, 3, self.cfg, np.random.default_rng(123),
                              device="cuda:0", gpu_signals=True)
        second = prepare_batch(self.header, 3, self.cfg, np.random.default_rng(123),
                               device="cuda:0", gpu_signals=True)
        for field in ("a", "bt", "a_noised", "b_noised"):
            self.assertTrue(torch.equal(getattr(first, field), getattr(second, field)))
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(0), before))

    def test_gpu_signal_generation_rejects_cpu_destination(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            prepare_batch(self.header, 3, self.cfg, np.random.default_rng(123),
                          device="cpu", gpu_signals=True)


if __name__ == "__main__":
    unittest.main()
