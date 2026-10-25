"""Compare the custom GPU arithmetic with an independent scalar oracle."""

import unittest

import numpy as np
import torch

from pearl_trainer.gpu import (TrainingCancelled, gpu_add_noise,
                               gpu_add_sparse_noise, gpu_transcripts)


def scalar_transcripts(a, b, tile_h=8, tile_w=8):
    """Deliberately scalar implementation of the official transcript rules."""
    m, k = a.shape
    n = b.shape[1]
    answer = np.zeros((m // tile_h, n // tile_w, 16), dtype=np.uint32)
    for row in range(0, m, tile_h):
        for col in range(0, n, tile_w):
            cells = [[0] * tile_w for _ in range(tile_h)]
            words = [0] * 16
            for slab, start in enumerate(range(0, k, 128)):
                combined = 0
                for i in range(tile_h):
                    for j in range(tile_w):
                        for inner in range(start, start + 128):
                            cells[i][j] += int(a[row + i, inner]) * int(b[inner, col + j])
                        combined ^= cells[i][j] & 0xFFFFFFFF
                slot = slab % 16
                previous = words[slot]
                words[slot] = (((previous << 13) | (previous >> 19)) & 0xFFFFFFFF) ^ combined
            answer[row // tile_h, col // tile_w] = words
    return answer


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((8, 256), dtype=np.int8)
        self.b = np.zeros((256, 8), dtype=np.int8)

    def test_rejects_wrong_dtype(self):
        with self.assertRaisesRegex(ValueError, "dtype int8"):
            gpu_transcripts(self.a.astype(np.int32), self.b)

    def test_rejects_incompatible_dimensions(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            gpu_transcripts(self.a, self.b[:128])

    def test_rejects_incomplete_slab(self):
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            gpu_transcripts(self.a[:, :255], self.b[:255])

    def test_rejects_partial_tile(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            gpu_transcripts(self.a[:6], self.b)

    def test_rejects_invalid_tile_dimensions(self):
        for h, w in ((0, 8), (3, 8), (2, 2), (32, 32)):
            with self.subTest(tile=(h, w)), self.assertRaises(ValueError):
                gpu_transcripts(self.a, self.b, tile_h=h, tile_w=w)

    def test_cancellation_before_allocation(self):
        with self.assertRaises(TrainingCancelled):
            gpu_transcripts(self.a, self.b, cancelled=lambda: True)


class TestNoisePreparation(unittest.TestCase):
    def check_noise(self, device):
        rng = np.random.default_rng(247)
        a = rng.integers(-128, 128, (6, 256), dtype=np.int8)
        bt = rng.integers(-128, 128, (8, 256), dtype=np.int8)
        factors = [rng.integers(-128, 128, shape, dtype=np.int8)
                   for shape in ((6, 128), (128, 256), (256, 128), (128, 8))]
        al, ar, bl, br = factors
        expected_a = a + (al.astype(np.int32) @ ar.astype(np.int32)).astype(np.int8)
        expected_b = bt.T + (bl.astype(np.int32) @ br.astype(np.int32)).astype(np.int8)
        actual_a, actual_b = gpu_add_noise(a, bt, *factors, device=device)
        self.assertEqual(actual_a.device.type, device)
        self.assertEqual(actual_b.device.type, device)
        self.assertEqual(actual_a.dtype, torch.int8)
        self.assertTrue(actual_a.is_contiguous())
        self.assertTrue(actual_b.is_contiguous())
        np.testing.assert_array_equal(actual_a.cpu().numpy(), expected_a)
        np.testing.assert_array_equal(actual_b.cpu().numpy(), expected_b)

    def test_cpu_noise_matches_independent_integer_products_and_wrapping(self):
        self.check_noise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_gpu_noise_matches_independent_integer_products_and_wrapping(self):
        self.check_noise("cuda")

    def test_noise_rejects_wrong_rank_before_computation(self):
        a = torch.zeros((8, 256), dtype=torch.int8)
        al = torch.zeros((8, 127), dtype=torch.int8)
        ar = torch.zeros((127, 256), dtype=torch.int8)
        bl = torch.zeros((256, 128), dtype=torch.int8)
        br = torch.zeros((128, 8), dtype=torch.int8)
        with self.assertRaisesRegex(ValueError, "rank 128"):
            gpu_add_noise(a, a, al, ar, bl, br, device="cpu")


class TestSparseNoisePreparation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(394)
        self.a = rng.integers(-128, 128, (8, 256), dtype=np.int8)
        self.bt = rng.integers(-128, 128, (6, 256), dtype=np.int8)
        self.al = rng.integers(-128, 128, (8, 128), dtype=np.int8)
        self.br = rng.integers(-128, 128, (128, 6), dtype=np.int8)
        self.a_first = rng.integers(0, 128, 256, dtype=np.int64)
        self.a_second = (self.a_first + rng.integers(1, 128, 256)) % 128
        self.b_first = rng.integers(0, 128, 256, dtype=np.int64)
        self.b_second = (self.b_first + rng.integers(1, 128, 256)) % 128

    def arguments(self):
        return [self.a, self.bt, self.al, self.a_first, self.a_second,
                self.b_first, self.b_second, self.br]

    def check_noise(self, device):
        # Reconstruct integer permutation matrices independently. Comparing
        # their dense products catches gather-axis and +1/-1 ordering errors.
        ar = np.zeros((128, 256), dtype=np.int32)
        bl = np.zeros((256, 128), dtype=np.int32)
        ar[self.a_first, np.arange(256)] = 1
        ar[self.a_second, np.arange(256)] = -1
        bl[np.arange(256), self.b_first] = 1
        bl[np.arange(256), self.b_second] = -1
        expected_a = self.a + (self.al.astype(np.int32) @ ar).astype(np.int8)
        expected_b = self.bt.T + (bl @ self.br.astype(np.int32)).astype(np.int8)
        actual_a, actual_b = gpu_add_sparse_noise(*self.arguments(), device=device)
        self.assertEqual(actual_a.device.type, device)
        self.assertEqual(actual_b.device.type, device)
        self.assertEqual(actual_a.dtype, torch.int8)
        self.assertEqual(actual_b.dtype, torch.int8)
        self.assertTrue(actual_a.is_contiguous())
        self.assertTrue(actual_b.is_contiguous())
        np.testing.assert_array_equal(actual_a.cpu().numpy(), expected_a)
        np.testing.assert_array_equal(actual_b.cpu().numpy(), expected_b)

    def test_cpu_sparse_noise_matches_dense_integer_oracle_with_wrapping(self):
        self.check_noise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_gpu_sparse_noise_matches_dense_integer_oracle_with_wrapping(self):
        self.check_noise("cuda")

    def test_torch_inputs_and_noncontiguous_uniform_factors(self):
        arguments = [torch.from_numpy(value) for value in self.arguments()]
        arguments[2] = arguments[2].T.contiguous().T
        arguments[7] = arguments[7].T.contiguous().T
        actual = gpu_add_sparse_noise(*arguments, device="cpu")
        expected = gpu_add_sparse_noise(*self.arguments(), device="cpu")
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)

    def test_sparse_noise_rejects_wrong_factor_shape(self):
        arguments = self.arguments()
        arguments[2] = self.al[:, :127]
        with self.assertRaisesRegex(ValueError, "rank 128"):
            gpu_add_sparse_noise(*arguments, device="cpu")

    def test_sparse_noise_rejects_wrong_signal_dimensions(self):
        arguments = self.arguments()
        arguments[1] = self.bt[:, :128]
        with self.assertRaisesRegex(ValueError, "incompatible"):
            gpu_add_sparse_noise(*arguments, device="cpu")

    def test_sparse_noise_rejects_non_int64_or_wrong_length_indices(self):
        for replacement, message in ((self.a_first.astype(np.int32), "dtype int64"),
                                     (self.a_first[:128], "length k"),
                                     (self.a_first.reshape(2, 128), "length k")):
            arguments = self.arguments()
            arguments[3] = replacement
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                gpu_add_sparse_noise(*arguments, device="cpu")

    def test_sparse_noise_rejects_out_of_rank_indices_before_allocation(self):
        for index in (-1, 128):
            arguments = self.arguments()
            arguments[5] = self.b_first.copy()
            arguments[5][10] = index
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "between 0 and 127"):
                gpu_add_sparse_noise(*arguments, device="cpu")


class TestCPUTranscripts(unittest.TestCase):
    def test_batched_cpu_path_matches_scalar_oracle(self):
        rng = np.random.default_rng(910)
        a = rng.integers(-128, 128, (8, 2176), dtype=np.int8)
        b = rng.integers(-128, 128, (2176, 8), dtype=np.int8)
        np.testing.assert_array_equal(gpu_transcripts(a, b, device="cpu"), scalar_transcripts(a, b))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class TestGPUTranscripts(unittest.TestCase):
    def test_negative_full_range_and_multiple_tiles(self):
        rng = np.random.default_rng(123456)
        a = rng.integers(-128, 128, (16, 256), dtype=np.int8)
        b = rng.integers(-128, 128, (256, 16), dtype=np.int8)
        expected = scalar_transcripts(a, b)
        actual = gpu_transcripts(a, b)
        self.assertEqual(actual.dtype, np.uint32)
        np.testing.assert_array_equal(actual, expected)

    def test_repeated_transcript_slot_rotates(self):
        rng = np.random.default_rng(987)
        a = rng.integers(-128, 128, (8, 2176), dtype=np.int8)
        b = rng.integers(-128, 128, (2176, 8), dtype=np.int8)
        np.testing.assert_array_equal(gpu_transcripts(a, b), scalar_transcripts(a, b))

    def test_maximum_k_preserves_repeated_slot_rotations(self):
        rng = np.random.default_rng(654)
        a = rng.integers(-128, 128, (8, 65536), dtype=np.int8)
        b = rng.integers(-128, 128, (65536, 8), dtype=np.int8)
        np.testing.assert_array_equal(gpu_transcripts(a, b), scalar_transcripts(a, b))

    def test_small_scratch_chunks_match_scalar_oracle(self):
        from unittest.mock import patch

        rng = np.random.default_rng(422)
        a = rng.integers(-128, 128, (8, 2176), dtype=np.int8)
        b = rng.integers(-128, 128, (2176, 8), dtype=np.int8)
        with patch("pearl_trainer.gpu._TRANSCRIPT_SCRATCH_BYTES", 1):
            np.testing.assert_array_equal(gpu_transcripts(a, b), scalar_transcripts(a, b))

    def test_non_power_of_two_tile_area(self):
        rng = np.random.default_rng(55)
        a = rng.integers(-128, 128, (6, 256), dtype=np.int8)
        b = rng.integers(-128, 128, (256, 8), dtype=np.int8)
        np.testing.assert_array_equal(
            gpu_transcripts(a, b, tile_h=6), scalar_transcripts(a, b, tile_h=6)
        )

    def test_torch_inputs_and_precision_restored(self):
        a = torch.full((8, 256), -128, dtype=torch.int8)
        b = torch.full((256, 8), -128, dtype=torch.int8)
        old_precision = torch.get_float32_matmul_precision()
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        np.testing.assert_array_equal(gpu_transcripts(a, b), np.zeros((1, 1, 16), np.uint32))
        self.assertEqual(torch.get_float32_matmul_precision(), old_precision)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, old_tf32)

    def test_cancellation_between_slabs(self):
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls >= 3

        a = np.zeros((8, 256), dtype=np.int8)
        b = np.zeros((256, 8), dtype=np.int8)
        with self.assertRaises(TrainingCancelled):
            gpu_transcripts(a, b, cancelled=cancelled)


if __name__ == "__main__":
    unittest.main()
