"""Pool share checks using the real GPU and pinned official CPU/native helpers."""

import os
from pathlib import Path
import unittest

import numpy as np
import torch

from pearl_trainer.gpu import gpu_transcripts
from pearl_trainer.protocol import (
    TrainingConfig, load_official_helpers, matching_tiles, pool_share_target,
    prepare_batch, verified_pool_share,
)

try:
    import pearl_mining as pm
except ImportError:
    pm = None


class PoolTargetTests(unittest.TestCase):
    def test_configuration_rejects_proof_input_larger_than_consensus_limit(self):
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            TrainingConfig(k=32384, tile_h=128, tile_w=2)
        self.assertEqual(TrainingConfig(k=32256, tile_h=128, tile_w=2).k, 32256)

    def test_final_target_is_exact_and_does_not_require_compact_representation(self):
        # Keep low significant bits that Bitcoin compact encoding cannot store.
        target = (1 << 225) + 123456789
        self.assertEqual(pool_share_target(target, TrainingConfig(), "final"), target)

    def test_invalid_targets_and_target_kinds_are_rejected(self):
        for target in (True, False, 0, -1, 2**256, 1.5, "ff"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                pool_share_target(target, TrainingConfig(), "final")
        with self.assertRaisesRegex(ValueError, "kind"):
            pool_share_target(123, TrainingConfig(), "automatic")

    @unittest.skipIf(pm is None, "official native helper unavailable")
    def test_base_target_scaling_preserves_existing_overflow_rejection(self):
        config = TrainingConfig()
        factor = config.tile_h * config.tile_w * config.k
        self.assertEqual(pool_share_target(1234567, config), 1234567 * factor)
        maximum_base = (2**256 - 1) // factor
        with self.assertRaisesRegex(ValueError, "overflows"):
            pool_share_target(maximum_base + 1, config, "base")


class OfficialPoolShareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get(
            "PEARL_OFFICIAL_SOURCE",
            str(Path(__file__).resolve().parents[1] / ".official" / "pearl"),
        ))
        if not source.exists():
            raise unittest.SkipTest("pinned official PEARL helper checkout unavailable")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA device unavailable")
        try:
            load_official_helpers(source)
            import pearl_mining as native
        except ImportError as exc:
            raise unittest.SkipTest(f"official native/helper dependency unavailable: {exc}") from None
        cls.native = native
        old_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(2)
        cls.config = TrainingConfig(m=128, n=128)
        cls.maximum = 2**256 - 1
        cls.examples = {}
        block_target = (0xFFFF << (8 * (0x1D - 3))) * (
            cls.config.tile_h * cls.config.tile_w * cls.config.k
        )
        for version in (1, 2, 3):
            header = native.IncompleteBlockHeader(
                version=1, prev_block=bytes(32), merkle_root=bytes(32),
                timestamp=123, nbits=0x1D00FFFF,
            )
            batch = prepare_batch(header, version, cls.config,
                                  np.random.default_rng(20261100 + version))
            transcripts = gpu_transcripts(batch.a_noised, batch.b_noised,
                                          cls.config.tile_h, cls.config.tile_w)
            # Select a valid pool share which deliberately misses the harder
            # block target, making override-vs-header behavior observable.
            for row, col, digest in matching_tiles(
                transcripts, batch.commitment.noise_seed_A, cls.maximum,
            ):
                if int.from_bytes(digest, "little") > block_target:
                    cls.examples[version] = (batch, header, row, col, digest)
                    break
            else:
                raise AssertionError("fixture contains no share above the block target")

    def test_v1_v2_v3_accept_pool_shares_without_altering_block_headers(self):
        for version, (batch, header, row, col, digest) in self.examples.items():
            with self.subTest(certificate_version=version):
                before = bytes(header.to_bytes())
                proof, verified_digest = verified_pool_share(
                    batch, header, version, row, col, digest, self.maximum,
                )
                self.assertEqual(verified_digest, digest)
                self.assertEqual(bytes(header.to_bytes()), before)
                self.assertEqual(header.nbits, 0x1D00FFFF)
                valid, _ = self.native.verify_plain_proof_for_cert_version(version, header, proof)
                self.assertFalse(valid, "fixture should fail the original block difficulty")

    def test_tampered_gpu_digest_is_rejected_by_official_cpu_reference(self):
        batch, header, row, col, digest = self.examples[3]
        tampered = bytes([digest[0] ^ 1]) + digest[1:]
        with self.assertRaisesRegex(RuntimeError, "CPU reference"):
            verified_pool_share(batch, header, 3, row, col, tampered, self.maximum)

    def test_exact_target_boundary_is_inclusive_and_one_below_rejects(self):
        batch, header, row, col, digest = self.examples[3]
        target = int.from_bytes(digest, "little")
        _, verified = verified_pool_share(batch, header, 3, row, col, digest, target)
        self.assertEqual(verified, digest)
        with self.assertRaisesRegex(ValueError, "exact pool target"):
            verified_pool_share(batch, header, 3, row, col, digest, target - 1)

    def test_unsupported_certificate_and_malformed_digests_are_rejected(self):
        batch, header, row, col, digest = self.examples[3]
        for version in (0, 4, True):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "certificate"):
                verified_pool_share(batch, header, version, row, col, digest, self.maximum)
        for invalid in (b"short", digest + b"extra", digest.hex()):
            with self.subTest(digest_type=type(invalid)), self.assertRaisesRegex(ValueError, "32 bytes"):
                verified_pool_share(batch, header, 3, row, col, invalid, self.maximum)

    def test_different_header_fails_native_commitment_verification(self):
        batch, _, row, col, digest = self.examples[3]
        wrong_header = self.native.IncompleteBlockHeader(
            version=1, prev_block=bytes(32), merkle_root=bytes(32),
            timestamp=123, nbits=0x207FFFFF,
        )
        with self.assertRaisesRegex(RuntimeError, "official verification"):
            verified_pool_share(batch, wrong_header, 3, row, col, digest, self.maximum)


if __name__ == "__main__":
    unittest.main()
