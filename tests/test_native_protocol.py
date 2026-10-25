"""End-to-end native batches against independent official arithmetic/proofs."""

from pathlib import Path
import unittest
from unittest.mock import patch

from blake3 import blake3
import numpy as np
import torch

from pearl_trainer import protocol
from pearl_trainer.gpu import gpu_transcripts


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class NativeProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        try:
            protocol.load_official_helpers(root / ".official" / "pearl")
            import pearl_mining as pm
            from miner_base.commitment_hash import CommitmentHasher
            from miner_base.matrix_merkle_tree import MatrixMerkleTree
            from miner_base.noise_generation import NoiseGenerator
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"official helper unavailable: {exc}") from None
        from pearl_trainer import native_cuda

        native_cuda.load_native("cuda:0")
        cls.native = native_cuda
        cls.pm = pm
        cls.hasher = CommitmentHasher
        cls.tree_type = MatrixMerkleTree
        cls.noise_type = NoiseGenerator
        cls.header = pm.IncompleteBlockHeader(
            version=1, prev_block=bytes(32), merkle_root=bytes(32),
            timestamp=1234, nbits=0x1D00FFFF,
        )

    def prepare(self, version=3, *, config=None, seed=709):
        return protocol.prepare_batch(
            self.header, version,
            protocol.TrainingConfig(m=128, n=128, k=2048) if config is None else config,
            np.random.default_rng(seed), device="cuda:0", gpu_signals=True, native=True,
        )

    def official_noise(self, batch):
        cfg = batch.config
        al, ar, bl, br = self.noise_type(noise_rank=128).generate_noise_metrices(
            batch.commitment.noise_seed_A, batch.commitment.noise_seed_B,
            cfg.m, cfg.k, cfg.n,
        )
        a = batch.a.cpu() + (al.int() @ ar.int()).to(torch.int8)
        b = batch.bt.cpu().T + (bl.int() @ br.int()).to(torch.int8)
        return a.contiguous(), b.contiguous()

    @staticmethod
    def python_matches(transcripts, key, target):
        matches = []
        for row in range(transcripts.shape[0]):
            for col in range(transcripts.shape[1]):
                digest = blake3(transcripts[row, col].astype("<u4").tobytes(), key=key).digest()
                if int.from_bytes(digest, "little") <= target:
                    matches.append((row, col, digest))
        return matches

    def test_native_commitments_and_noise_match_official_helpers_for_every_certificate(self):
        for version in (1, 2, 3):
            with self.subTest(certificate_version=version):
                batch = self.prepare(version)
                self.assertTrue(batch.native)
                self.assertTrue(batch.a.is_cuda)
                self.assertTrue(batch.bt.is_cuda)
                self.assertIsInstance(batch.a_tree, protocol.DeferredMerkleTree)
                self.assertIsNone(batch.a_tree._tree)
                self.assertIsNone(batch.b_tree._tree)
                key = self.hasher.get_key(bytes(self.header.to_bytes()), batch.config.official_config())
                a_root = self.tree_type(batch.a.cpu(), key).root
                b_root = self.tree_type(batch.bt.cpu(), key).root
                self.assertEqual(batch.a_tree.root, a_root)
                self.assertEqual(batch.b_tree.root, b_root)
                expected = self.hasher.commitment_hash_from_merkle_roots(
                    a_root, b_root, key,
                    salted_dims=(batch.config.m, batch.config.n) if version == 3 else None,
                )
                self.assertEqual(batch.commitment.noise_seed_A, expected.noise_seed_A)
                self.assertEqual(batch.commitment.noise_seed_B, expected.noise_seed_B)
                expected_a, expected_b = self.official_noise(batch)
                self.assertTrue(torch.equal(batch.a_noised.cpu(), expected_a))
                self.assertTrue(torch.equal(batch.b_noised.cpu(), expected_b))
                self.assertTrue(torch.equal(batch.b_noised_t.cpu(), expected_b.T))
                # Independent checks above must not populate the batch's trees.
                self.assertIsNone(batch.a_tree._tree)
                self.assertIsNone(batch.b_tree._tree)

    def test_native_search_matches_cpu_integer_transcripts_and_python_blake3(self):
        cfg = protocol.TrainingConfig(m=128, n=256, k=2176)
        batch = self.prepare(config=cfg, seed=913)
        a, b = self.official_noise(batch)
        expected_transcripts = gpu_transcripts(a, b, device="cpu")
        actual_transcripts = self.native.transcripts_device(batch.a_noised, batch.b_noised_t)
        np.testing.assert_array_equal(actual_transcripts.cpu().numpy().view(np.uint32),
                                      expected_transcripts)
        target = (1 << 255) - 1
        expected = self.python_matches(expected_transcripts, batch.commitment.noise_seed_A, target)
        actual = list(protocol.search_batch(batch, target, "cuda:0"))
        self.assertEqual(actual, expected)
        self.assertIsNone(batch.a_tree._tree)
        self.assertIsNone(batch.b_tree._tree)

    def test_native_candidates_materialize_only_the_needed_official_proof_trees(self):
        maximum_target = (1 << 256) - 1
        for version in (1, 2, 3):
            with self.subTest(certificate_version=version):
                with patch("miner_base.matrix_merkle_tree.MatrixMerkleTree",
                           wraps=self.tree_type) as construct:
                    batch = self.prepare(version, seed=117 + version)
                    matches = list(protocol.search_batch(batch, maximum_target, "cuda:0"))
                    self.assertEqual(len(matches), 256)
                    construct.assert_not_called()
                    self.assertIsNone(batch.a_tree._tree)
                    self.assertIsNone(batch.b_tree._tree)
                    row, col, digest = matches[0]
                    proof, verified_digest = protocol.verified_pool_share(
                        batch, self.header, version, row, col, digest, maximum_target,
                    )
                    self.assertEqual(verified_digest, digest)
                    self.assertEqual(construct.call_count, 2)
                    self.assertIsNotNone(batch.a_tree._tree)
                    self.assertIsNotNone(batch.b_tree._tree)
                    self.assertEqual(batch.a_tree._tree.root, batch.a_tree.root)
                    self.assertEqual(batch.b_tree._tree.root, batch.b_tree.root)
                    valid, error = self.pm.verify_plain_proof_for_cert_version(
                        version, self.header, proof, nbits_override=0x207FFFFF,
                    )
                    self.assertTrue(valid, error)
                    batch.plain_proof(row, col)
                    self.assertEqual(construct.call_count, 2, "proof trees must be reused")

    def test_corrupted_gpu_commitment_is_rejected_before_official_proof_creation(self):
        batch = self.prepare(seed=441)
        maximum_target = (1 << 256) - 1
        row, col, digest = next(protocol.search_batch(batch, maximum_target, "cuda:0"))
        root = batch.a_tree.root
        batch.a_tree.root = bytes([root[0] ^ 1]) + root[1:]
        with self.assertRaisesRegex(RuntimeError, "GPU Merkle root differs"):
            protocol.verified_pool_share(batch, self.header, 3, row, col, digest, maximum_target)
        self.assertIsNone(batch.a_tree._tree)
        self.assertIsNone(batch.b_tree._tree)

    def test_hash_wrapper_retries_full_buffer_without_losing_matching_tickets(self):
        rng = np.random.default_rng(5493)
        words = rng.integers(0, 1 << 32, (5, 17, 16), dtype=np.uint32)
        transcripts = torch.from_numpy(words.view(np.int32)).to("cuda:0")
        key = bytes(range(32))
        target = (1 << 256) - 1
        actual = self.native.hash_candidates(transcripts, key, target, capacity=1)
        self.assertEqual(len(actual), 85)
        self.assertEqual(actual, self.python_matches(words, key, target))

    def test_hash_wrapper_rejects_out_of_range_targets_before_kernel_launch(self):
        transcripts = torch.zeros((1, 1, 16), dtype=torch.int32, device="cuda:0")
        for target in (-1, 1 << 256):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.native.hash_candidates(transcripts, bytes(32), target)


if __name__ == "__main__":
    unittest.main()
