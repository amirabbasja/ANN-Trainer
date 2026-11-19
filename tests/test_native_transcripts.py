"""Native signed-int8 DP4A transcripts against independent int64 products."""

import unittest

import numpy as np
import torch


def integer_transcripts(a, bt, tile_w=8):
    """Keep cumulative cells in CPU int64, independently of packed GPU work."""
    m, k = a.shape
    n = bt.shape[0]
    cumulative = np.zeros((m, n), dtype=np.int64)
    output = np.zeros((m // 8, n // tile_w, 16), dtype=np.uint32)
    for slab, start in enumerate(range(0, k, 128)):
        cumulative += a[:, start:start + 128].astype(np.int64) @ bt[:, start:start + 128].astype(np.int64).T
        tiles = cumulative.reshape(m // 8, 8, n // tile_w, tile_w).transpose(0, 2, 1, 3)
        reductions = np.bitwise_xor.reduce(tiles.reshape(m // 8, n // tile_w, 8 * tile_w), axis=-1).astype(np.uint32)
        previous = output[..., slab % 16]
        output[..., slab % 16] = ((previous << np.uint32(13)) | (previous >> np.uint32(19))) ^ reductions
    return output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class NativeTranscriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import lazily: systems without CUDA or the optional compiler should
        # still be able to discover and run the CPU test suite.
        from LRAEP_trainer.native_cuda import NativeUnavailable, load_native, transcripts_device

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

    def test_wide_aligned_and_unaligned_inputs_match_integer_reference(self):
        rng = np.random.default_rng(71)
        a = rng.integers(-128, 128, size=(72, 2176), dtype=np.int8)
        bt = rng.integers(-128, 128, size=(144, 2176), dtype=np.int8)
        expected = integer_transcripts(a, bt, tile_w=16)
        a_gpu, bt_gpu = torch.from_numpy(a).cuda(), torch.from_numpy(bt).cuda()
        storage = torch.empty(bt.size + 4, dtype=torch.int8, device=bt_gpu.device)
        unaligned_bt = storage[4:].view(bt.shape)
        unaligned_bt.copy_(bt_gpu)
        for operand in (bt_gpu, unaligned_bt):
            actual = self.run_transcripts(a_gpu, operand, tile_w=16)
            np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint32), expected)

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


    def test_large_optimized_dispatch_and_long_transcripts(self):
        # This enters the large-matrix dispatch and updates each transcript slot
        # repeatedly. Compare every 8x8 word with the retained scalar kernel,
        # and independently reconstruct corner tiles for both transcript widths.
        m, n, k = 776, 784, 16384
        rng = np.random.default_rng(979)
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        bt = rng.integers(-128, 128, (n, k), dtype=np.int8)
        ag, bg = torch.from_numpy(a).cuda(), torch.from_numpy(bt).cuda()
        for tile_w in (8, 16):
            with self.subTest(tile_w=tile_w):
                actual = self.run_transcripts(ag, bg, tile_w=tile_w)
                actual_cpu = actual.cpu().numpy().view(np.uint32)
                if tile_w == 8:
                    legacy = self.run_transcripts(ag, bg, tile_w=8, variant=7)
                    np.testing.assert_array_equal(actual_cpu, legacy.cpu().numpy().view(np.uint32))
                for row in (0, m // 8 - 1):
                    for col in (0, n // tile_w - 1):
                        expected = integer_transcripts(a[row * 8:(row + 1) * 8],
                                                         bt[col * tile_w:(col + 1) * tile_w],
                                                         tile_w=tile_w)
                        np.testing.assert_array_equal(actual_cpu[row, col], expected[0, 0])


if __name__ == "__main__":
    unittest.main()
