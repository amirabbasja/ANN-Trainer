import gc
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from LRAEP_trainer import native_cuda


class _TrackedBytes(bytes):
    def __del__(self):
        self.released.append(True)


class _DeviceWords:
    def __init__(self, values, destination):
        self.values = values.copy()
        self.destination = destination
        self.streams = []

    def record_stream(self, stream):
        self.streams.append(stream)


class _HostWords:
    def __init__(self, values):
        self.values = values

    def to(self, destination):
        return _DeviceWords(self.values, destination)


class NativeWordLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = [("shared", native_cuda)]
        source = (Path(__file__).resolve().parents[2] / "LRAEP-WR-Trainer"
                  / "src" / "LRAEP_trainer" / "native_cuda.py")
        if source.exists():
            spec = importlib.util.spec_from_file_location(
                "LRAEP_trainer._standalone_native_memory", source,
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.modules.append(("standalone", module))

    def test_repeated_values_are_released_after_the_caller_drops_them(self):
        for label, module in self.modules:
            with self.subTest(copy=label), \
                    patch.object(module.torch, "from_numpy", side_effect=_HostWords), \
                    patch.object(module.torch.cuda, "current_stream", return_value=object()):
                released = []
                value = _TrackedBytes(range(32))
                value.released = released
                first = module._words(value, "cuda:2")
                second = module._words(value, "cuda:2")
                self.assertIsNot(first, second)
                first_ref, second_ref = weakref.ref(first), weakref.ref(second)
                del first, second, value
                gc.collect()
                self.assertIsNone(first_ref())
                self.assertIsNone(second_ref())
                self.assertEqual(released, [True])

    def test_word_order_and_consuming_stream_are_preserved(self):
        values = np.array([0, 1, 0xFFFFFFFF, 0x80000000, 0x12345678, 9, 11, 13], dtype="<u4")
        for label, module in self.modules:
            stream = object()
            with self.subTest(copy=label), \
                    patch.object(module.torch, "from_numpy", side_effect=_HostWords), \
                    patch.object(module.torch.cuda, "current_stream", return_value=stream) as current:
                result = module._words(values.tobytes(), "cuda:2")
                np.testing.assert_array_equal(result.values, values.view(np.int32))
                self.assertEqual(result.destination, module.torch.device("cuda:2"))
                self.assertEqual(result.streams, [stream])
                current.assert_called_once_with(module.torch.device("cuda:2"))

    def test_invalid_lengths_fail_before_allocation(self):
        for label, module in self.modules:
            for length in (0, 31, 33):
                with self.subTest(copy=label, length=length), \
                        patch.object(module.torch, "from_numpy") as allocate:
                    with self.assertRaises(ValueError):
                        module._words(bytes(length), "cuda:0")
                    allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
