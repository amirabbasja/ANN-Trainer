"""Native signed-int8 DP4A transcripts against independent int64 products."""

import unittest

import numpy as np
import torch


def integer_transcripts(a, bt):
    """Keep cumulative cells in CPU int64, independently of packed GPU work."""
    m, k = a.shape
    n = bt.shape[0]
    cumulative = np.zeros((m, n), dtype=np.int64)
    output = np.zeros((m // 8, n // 8, 16), dtype=np.uint32)
    for slab, start in enumerate(range(0, k, 128)):
        cumulative += a[:, start:start + 128].astype(np.int64) @ bt[:, start:start + 128].astype(np.int64).T
        tiles = cumulative.reshape(m // 8, 8, n // 8, 8).transpose(0, 2, 1, 3)
        reductions = np.bitwise_xor.reduce(tiles.reshape(m // 8, n // 8, 64), axis=-1).astype(np.uint32)
        previous = output[..., slab % 16]
        output[..., slab % 16] = ((previous << np.uint32(13)) | (previous >> np.uint32(19))) ^ reductions
    return output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class NativeTranscriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import lazily: systems without CUDA or the optional compiler should
        # still be able to discover and run the CPU test suite.
        from pearl_trainer.native_cuda import NativeUnavailable, load_native, transcripts_device

        try:
            load_native("cuda:0")
        except NativeUnavailable as exc:
            raise unittest.SkipTest(str(exc)) from None
        cls.run_transcripts = staticmethod(transcripts_device)

    def check_shape(self, m, n, k, variants, seed):
        rng = np.random.default_rng(seed)
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        bt = rng.integers(-128, 128, (n, k), dtype=np.int8)
        expected = integer_transcripts(a, bt)
        a_gpu, bt_gpu = torch.from_numpy(a).cuda(), torch.from_numpy(bt).cuda()
        for variant in variants:
            with self.subTest(shape=(m, n, k), variant=variant):
                result = self.run_transcripts(a_gpu, bt_gpu, variant=variant)
                self.assertEqual(result.dtype, torch.int32)
                self.assertTrue(result.is_cuda)
                np.testing.assert_array_equal(result.cpu().numpy().view(np.uint32), expected)

    def test_all_variants_preserve_full_range_and_repeated_slots(self):
        self.check_shape(16, 24, 2176, range(8), 198)

    def test_partial_blocks_are_masked_without_affecting_valid_tiles(self):
        self.check_shape(24, 40, 256, range(8), 491)

    def test_single_rank_and_single_tile(self):
        self.check_shape(8, 8, 128, (0, 4, 7), 812)

    def test_maximum_k_preserves_every_slot_rotation(self):
        self.check_shape(8, 8, 65536, (0, 4, 7), 519)

    def test_nondefault_stream_orders_inputs_and_output(self):
        rng = np.random.default_rng(182)
        a = rng.integers(-128, 128, (8, 256), dtype=np.int8)
        bt = rng.integers(-128, 128, (8, 256), dtype=np.int8)
        expected = integer_transcripts(a, bt)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = self.run_transcripts(torch.from_numpy(a).cuda(), torch.from_numpy(bt).cuda())
        stream.synchronize()
        np.testing.assert_array_equal(result.cpu().numpy().view(np.uint32), expected)


if __name__ == "__main__":
    unittest.main()
