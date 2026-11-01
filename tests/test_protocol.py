"""Protocol boundary checks independent of the GPU implementation."""

import unittest

import numpy as np

from LRAEP_trainer.protocol import TrainingConfig, adjusted_target, matching_tiles


class TranscriptTargetTests(unittest.TestCase):
    def test_little_endian_target_is_inclusive(self):
        words = np.arange(16, dtype=np.uint32).reshape(1, 1, 16)
        key = bytes(range(32))
        # Fixed keyed digest captured from the independent upstream backend.
        digest = bytes.fromhex("40770fb51e85ef4fdf271c7172bb9e4775a235e4f0e9d7e01d074b525301ca26")
        boundary = int.from_bytes(digest, "little")
        self.assertEqual(list(matching_tiles(words, key, boundary)), [(0, 0, digest)])
        self.assertEqual(list(matching_tiles(words, key, boundary - 1)), [])

    def test_big_endian_numpy_storage_serializes_to_protocol_order(self):
        native = np.arange(16, dtype=np.uint32).reshape(1, 1, 16)
        swapped = native.astype(">u4")
        self.assertEqual(list(matching_tiles(native, bytes(32), 2**256 - 1)),
                         list(matching_tiles(swapped, bytes(32), 2**256 - 1)))

    def test_consensus_invalid_configurations_fail_before_work(self):
        invalid = [dict(k=1024), dict(k=2112), dict(k=65664), dict(m=64),
                   dict(n=130), dict(tile_h=3), dict(tile_h=2, tile_w=8),
                   dict(tile_h=32, tile_w=32)]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                TrainingConfig(**overrides)

    def test_target_scaling_and_overflow(self):
        config = TrainingConfig()
        factor = config.tile_h * config.tile_w * config.k
        self.assertEqual(adjusted_target(123456, config), 123456 * factor)
        maximum_base = (2**256 - 1) // factor
        self.assertEqual(adjusted_target(maximum_base, config), maximum_base * factor)
        with self.assertRaises(ValueError):
            adjusted_target(maximum_base + 1, config)


if __name__ == "__main__":
    unittest.main()
