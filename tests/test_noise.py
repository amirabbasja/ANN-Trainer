"""Noise-factor parity with the unmodified pinned official generator."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pearl_trainer import noise
from pearl_trainer.noise import generate_noise_factors, generate_sparse_noise_factors
from pearl_trainer.protocol import load_official_helpers


class NoiseFactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        try:
            load_official_helpers(root / ".official" / "pearl")
            from miner_base.noise_generation import NoiseGenerator
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"official helper unavailable: {exc}") from None
        cls.reference = NoiseGenerator(noise_rank=128)

    def assert_factors_match(self, seed_a, seed_b, m, k, n):
        expected = self.reference.generate_noise_metrices(seed_a, seed_b, m, k, n)
        actual = generate_noise_factors(seed_a, seed_b, m, k, n)
        self.assertEqual(len(actual), 4)
        for label, factor, reference, shape in zip(
            ("AL", "AR", "BL", "BR"), actual, expected,
            ((m, 128), (128, k), (k, 128), (128, n)),
        ):
            with self.subTest(factor=label):
                self.assertEqual(factor.device.type, "cpu")
                self.assertEqual(factor.dtype, torch.int8)
                self.assertEqual(tuple(factor.shape), shape)
                self.assertTrue(torch.equal(factor, reference))
        return actual

    def test_all_factors_match_across_seeds_and_dimensions(self):
        cases = (
            (bytes(32), bytes([255]) * 32, 128, 128, 128),
            (bytes(range(32)), bytes(reversed(range(32))), 256, 2048, 256),
            (bytes([0x55]) * 32, bytes([0xAA]) * 32, 384, 4096, 512),
        )
        for seed_a, seed_b, m, k, n in cases:
            with self.subTest(m=m, k=k, n=n):
                self.assert_factors_match(seed_a, seed_b, m, k, n)

    def test_partial_final_hash_draw_matches_without_extra_rows_or_columns(self):
        # The helper permits non-aligned dimensions even though training's k
        # uses rank-sized slabs. Exercise every partial eight-u32 digest tail.
        for tail in range(1, 8):
            with self.subTest(tail=tail):
                self.assert_factors_match(
                    bytes(range(32)), bytes([tail]) * 32,
                    128 + tail, 128 + tail, 136 - tail,
                )

    def test_permutation_factors_have_exactly_one_positive_and_negative_entry(self):
        _, ar, bl, _ = generate_noise_factors(bytes(32), bytes([255]) * 32,
                                              128, 2051, 128)
        for factor in (ar.T, bl):
            self.assertTrue(np.all((factor == 1).sum(dim=1).numpy() == 1))
            self.assertTrue(np.all((factor == -1).sum(dim=1).numpy() == 1))
            self.assertTrue(np.all((factor != 0).sum(dim=1).numpy() == 2))

    def test_repeated_calls_are_deterministic_and_return_independent_tensors(self):
        args = (bytes(range(32)), bytes(reversed(range(32))), 128, 131, 129)
        first = generate_noise_factors(*args)
        second = generate_noise_factors(*args)
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        first[1].zero_()
        third = generate_noise_factors(*args)
        self.assertTrue(torch.equal(second[1], third[1]))

    def test_official_minimum_dimension_validation_is_preserved(self):
        for shape in ((127, 128, 128), (128, 127, 128), (128, 128, 127)):
            for generate in (generate_noise_factors, generate_sparse_noise_factors):
                with self.subTest(shape=shape, generate=generate.__name__), self.assertRaises(ValueError):
                    generate(bytes(32), bytes(32), *shape)

    def test_sparse_factors_and_indexed_products_match_official_dense_noise(self):
        cases = (
            (bytes(32), bytes([255]) * 32, 128, 128, 128),
            (bytes(range(32)), bytes(reversed(range(32))), 129, 131, 133),
            (bytes([0x55]) * 32, bytes([0xAA]) * 32, 256, 2048, 384),
        )
        for seed_a, seed_b, m, k, n in cases:
            with self.subTest(m=m, k=k, n=n):
                expected_al, expected_ar, expected_bl, expected_br = (
                    self.reference.generate_noise_metrices(seed_a, seed_b, m, k, n)
                )
                al, a_first, a_second, b_first, b_second, br = (
                    generate_sparse_noise_factors(seed_a, seed_b, m, k, n)
                )
                self.assertTrue(torch.equal(al, expected_al))
                self.assertTrue(torch.equal(br, expected_br))
                for indices in (a_first, a_second, b_first, b_second):
                    self.assertEqual(indices.dtype, torch.int64)
                    self.assertEqual(indices.device.type, "cpu")
                    self.assertEqual(tuple(indices.shape), (k,))
                    self.assertTrue(bool(torch.all((0 <= indices) & (indices < 128))))

                self.assertTrue(torch.equal(a_first, expected_ar.argmax(dim=0)))
                self.assertTrue(torch.equal(a_second, expected_ar.argmin(dim=0)))
                self.assertTrue(torch.equal(b_first, expected_bl.argmax(dim=1)))
                self.assertTrue(torch.equal(b_second, expected_bl.argmin(dim=1)))
                self.assertTrue(torch.equal(
                    al[:, a_first] - al[:, a_second],
                    (expected_al.int() @ expected_ar.int()).to(torch.int8),
                ))
                self.assertTrue(torch.equal(
                    br[b_first, :] - br[b_second, :],
                    (expected_bl.int() @ expected_br.int()).to(torch.int8),
                ))

    def test_sparse_path_does_not_materialize_dense_permutations(self):
        with patch.object(noise._generator(), "_NoiseGenerator__generate_permutation_matrix",
                          side_effect=AssertionError("dense permutation was allocated")):
            factors = generate_sparse_noise_factors(bytes(32), bytes([255]) * 32,
                                                    128, 65536, 128)
        self.assertEqual(len(factors), 6)
        self.assertEqual(tuple(factors[0].shape), (128, 128))
        self.assertEqual(tuple(factors[-1].shape), (128, 128))
        self.assertTrue(all(tuple(vector.shape) == (65536,) for vector in factors[1:-1]))

    def test_sparse_permutation_tail_indices_match_every_partial_digest(self):
        for tail in range(1, 8):
            with self.subTest(tail=tail):
                args = (bytes([tail]) * 32, bytes([255 - tail]) * 32,
                        128, 128 + tail, 128)
                _, expected_ar, expected_bl, _ = self.reference.generate_noise_metrices(*args)
                _, a_first, a_second, b_first, b_second, _ = generate_sparse_noise_factors(*args)
                self.assertTrue(torch.equal(a_first, expected_ar.argmax(dim=0)))
                self.assertTrue(torch.equal(a_second, expected_ar.argmin(dim=0)))
                self.assertTrue(torch.equal(b_first, expected_bl.argmax(dim=1)))
                self.assertTrue(torch.equal(b_second, expected_bl.argmin(dim=1)))


if __name__ == "__main__":
    unittest.main()
