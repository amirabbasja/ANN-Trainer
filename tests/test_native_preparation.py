"""Native CUDA noise bytes and fused addition against the official CPU path."""

import ctypes
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


class NativePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        from LRAEP_trainer.native_cuda import load_native
        from LRAEP_trainer.protocol import load_official_helpers

        cls.lib = load_native()
        root = Path(__file__).resolve().parents[1]
        try:
            load_official_helpers(root / "src" / "LRAEP_helpers")
            from miner_base.noise_generation import NoiseGenerator
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"official helper unavailable: {exc}") from None
        cls.reference = NoiseGenerator(noise_rank=128)
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        cls.lib.LRAEP_noise_factors.argtypes = [integer, integer, integer] + [pointer] * 7
        cls.lib.LRAEP_noise_factors.restype = integer
        cls.lib.LRAEP_apply_noise.argtypes = [pointer, pointer, integer, integer, integer] + [pointer] * 7
        cls.lib.LRAEP_apply_noise.restype = integer
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)

    @staticmethod
    def pointer(tensor):
        return ctypes.c_void_p(tensor.data_ptr())

    def factors(self, seed_a, seed_b, m, n, k):
        key_a = torch.from_numpy(np.frombuffer(seed_a, dtype="<u4").copy()).cuda()
        key_b = torch.from_numpy(np.frombuffer(seed_b, dtype="<u4").copy()).cuda()
        al = torch.empty((m, 128), dtype=torch.int8, device="cuda")
        brt = torch.empty((n, 128), dtype=torch.int8, device="cuda")
        ai = torch.empty((k, 2), dtype=torch.int32, device="cuda")
        bi = torch.empty((k, 2), dtype=torch.int32, device="cuda")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        code = self.lib.LRAEP_noise_factors(
            m, n, k, self.pointer(key_a), self.pointer(key_b),
            self.pointer(al), self.pointer(brt), self.pointer(ai), self.pointer(bi), stream,
        )
        self.assertEqual(code, 0, f"noise launch failed with CUDA error {code}")
        # Retain keys until callers synchronize the stream.
        return SimpleNamespace(al=al, brt=brt, ai=ai, bi=bi, keys=(key_a, key_b))

    def apply(self, a, bt, factors):
        m, k = a.shape
        n = bt.shape[0]
        out_a = torch.empty_like(a)
        out_bt = torch.empty_like(bt)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        code = self.lib.LRAEP_apply_noise(
            self.pointer(a), self.pointer(bt), m, n, k,
            self.pointer(factors.al), self.pointer(factors.brt),
            self.pointer(factors.ai), self.pointer(factors.bi),
            self.pointer(out_a), self.pointer(out_bt), stream,
        )
        self.assertEqual(code, 0, f"noise application failed with CUDA error {code}")
        return out_a, out_bt

    def assert_factors(self, actual, expected):
        al, ar, bl, br = expected
        self.assertTrue(torch.equal(actual.al.cpu(), al))
        self.assertTrue(torch.equal(actual.brt.cpu(), br.T))
        expected_ai = torch.stack(((ar.T == 1).to(torch.int32).argmax(1),
                                   (ar.T == -1).to(torch.int32).argmax(1)), dim=1)
        expected_bi = torch.stack(((bl == 1).to(torch.int32).argmax(1),
                                   (bl == -1).to(torch.int32).argmax(1)), dim=1)
        self.assertTrue(torch.equal(actual.ai.cpu().to(torch.int64), expected_ai))
        self.assertTrue(torch.equal(actual.bi.cpu().to(torch.int64), expected_bi))

    def test_uniform_bytes_and_all_permutation_positions_match_official(self):
        cases = (
            (bytes(32), bytes([255]) * 32, 128, 128, 128),
            (bytes(range(32)), bytes(reversed(range(32))), 129, 135, 131),
            (bytes([0x55]) * 32, bytes([0xAA]) * 32, 256, 384, 2049),
        )
        for seed_a, seed_b, m, n, k in cases:
            with self.subTest(m=m, n=n, k=k):
                expected = self.reference.generate_noise_metrices(seed_a, seed_b, m, k, n)
                actual = self.factors(seed_a, seed_b, m, n, k)
                torch.cuda.current_stream().synchronize()
                self.assert_factors(actual, expected)

    def test_full_range_fused_addition_matches_dense_integer_reference(self):
        m, n, k = 129, 137, 2049
        seed_a, seed_b = bytes(range(32)), bytes([0xA5]) * 32
        al, ar, bl, br = self.reference.generate_noise_metrices(seed_a, seed_b, m, k, n)
        rng = np.random.default_rng(20261101)
        a_cpu = torch.from_numpy(rng.integers(-128, 128, (m, k), dtype=np.int8))
        bt_cpu = torch.from_numpy(rng.integers(-128, 128, (n, k), dtype=np.int8))
        raw_a = a_cpu.to(torch.int32) + al.to(torch.int32) @ ar.to(torch.int32)
        raw_bt = bt_cpu.to(torch.int32) + (bl.to(torch.int32) @ br.to(torch.int32)).T
        self.assertTrue(bool(((raw_a < -128) | (raw_a > 127)).any()))
        self.assertTrue(bool(((raw_bt < -128) | (raw_bt > 127)).any()))
        factors = self.factors(seed_a, seed_b, m, n, k)
        out_a, out_bt = self.apply(a_cpu.cuda(), bt_cpu.cuda(), factors)
        torch.cuda.current_stream().synchronize()
        self.assertTrue(torch.equal(out_a.cpu(), raw_a.to(torch.int8)))
        self.assertTrue(torch.equal(out_bt.cpu(), raw_bt.to(torch.int8)))
        self.assertTrue(out_a.is_contiguous())
        self.assertTrue(out_bt.is_contiguous())

    def test_nondefault_stream_keeps_factor_generation_and_application_ordered(self):
        m, n, k = 128, 129, 257
        seed_a, seed_b = bytes([0x31]) * 32, bytes([0x97]) * 32
        al, ar, bl, br = self.reference.generate_noise_metrices(seed_a, seed_b, m, k, n)
        a_cpu = torch.full((m, k), -64, dtype=torch.int8)
        bt_cpu = torch.full((n, k), 63, dtype=torch.int8)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            factors = self.factors(seed_a, seed_b, m, n, k)
            out_a, out_bt = self.apply(a_cpu.cuda(), bt_cpu.cuda(), factors)
        stream.synchronize()
        expected_a = a_cpu + (al.to(torch.int32) @ ar.to(torch.int32)).to(torch.int8)
        expected_bt = bt_cpu + (bl.to(torch.int32) @ br.to(torch.int32)).to(torch.int8).T
        self.assertTrue(torch.equal(out_a.cpu(), expected_a))
        self.assertTrue(torch.equal(out_bt.cpu(), expected_bt))

    def test_invalid_dimensions_fail_before_a_kernel_launch(self):
        null = ctypes.c_void_p()
        self.assertNotEqual(self.lib.LRAEP_noise_factors(0, 128, 128, *([null] * 7)), 0)
        self.assertNotEqual(self.lib.LRAEP_apply_noise(null, null, 128, 0, 128,
                                                     *([null] * 7)), 0)


if __name__ == "__main__":
    unittest.main()
