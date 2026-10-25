"""Protocol boundary checks independent of the GPU implementation."""

import unittest

import numpy as np
from blake3 import blake3

from pearl_trainer.protocol import TrainingConfig, adjusted_target, matching_tiles

try:
    import pearl_mining
except ImportError:
    pearl_mining = None


class TranscriptTargetTests(unittest.TestCase):
    def test_little_endian_target_is_inclusive(self):
        words = np.arange(16, dtype=np.uint32).reshape(1, 1, 16)
        key = bytes(range(32))
        message = b"".join(int(x).to_bytes(4, "little") for x in range(16))
        digest = blake3(message, key=key).digest()
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

    @unittest.skipIf(pearl_mining is None, "build official helper to test target adjustment")
    def test_official_target_scaling_and_overflow(self):
        config = TrainingConfig()
        factor = config.tile_h * config.tile_w * config.k
        self.assertEqual(adjusted_target(123456, config), 123456 * factor)
        maximum_base = (2**256 - 1) // factor
        self.assertEqual(adjusted_target(maximum_base, config), maximum_base * factor)
        with self.assertRaises(ValueError):
            adjusted_target(maximum_base + 1, config)


if __name__ == "__main__":
    unittest.main()
