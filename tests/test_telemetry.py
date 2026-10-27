"""GPU identity mapping, optional sensor handling, and sampler lifecycle."""

import os
import subprocess
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from LRAEP_trainer import telemetry


UUID_ONE = "356f34b2-260b-075e-a59b-34fd9402b132"
UUID_TWO = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class DeviceIdentityTests(unittest.TestCase):
    def test_selected_logical_device_uses_its_uuid_despite_visible_reordering(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,1"}), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid=UUID_TWO)) as properties:
            self.assertEqual(telemetry._device_identifier("cuda:1"), f"GPU-{UUID_TWO}")
        properties.assert_called_once_with(1)

    def test_implicit_cuda_device_uses_current_device_and_preserves_uuid_prefix(self):
        with patch("torch.cuda.current_device", return_value=2), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid=f"GPU-{UUID_ONE}")) as properties:
            self.assertEqual(telemetry._device_identifier("cuda"), f"GPU-{UUID_ONE}")
        properties.assert_called_once_with(2)

    def test_pci_identity_is_used_when_torch_has_no_uuid(self):
        properties = SimpleNamespace(pci_domain_id=0, pci_bus_id=0x88, pci_device_id=0)
        with patch("torch.cuda.get_device_properties", return_value=properties):
            self.assertEqual(telemetry._device_identifier("cuda:0"), "00000000:88:00.0")

    def test_visible_uuid_is_fallback_when_cuda_properties_unavailable(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": f"GPU-{UUID_ONE},GPU-{UUID_TWO}"}), \
                patch("torch.cuda.get_device_properties", side_effect=RuntimeError("unavailable")):
            self.assertEqual(telemetry._device_identifier("cuda:1"), f"GPU-{UUID_TWO}")

    def test_ambiguous_numeric_ordinal_does_not_report_another_gpu(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,1"}), \
                patch("torch.cuda.get_device_properties", side_effect=RuntimeError("unavailable")):
            self.assertIsNone(telemetry._device_identifier("cuda:1"))

    def test_invalid_or_cpu_device_is_unavailable(self):
        self.assertIsNone(telemetry._device_identifier("cpu"))
        self.assertIsNone(telemetry._device_identifier("invalid-device"))


class SensorTests(unittest.TestCase):
    def test_queries_exact_gpu_with_bounded_subprocess(self):
        result = subprocess.CompletedProcess([], 0, "57\n", "")
        with patch.object(telemetry.subprocess, "run", return_value=result) as run:
            self.assertEqual(telemetry._query_gpu_utilization(f"GPU-{UUID_ONE}"), 57)
        self.assertIn(f"--id=GPU-{UUID_ONE}", run.call_args.args[0])
        self.assertLessEqual(run.call_args.kwargs["timeout"], 1)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_missing_driver_tool_or_timeout_is_unavailable(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 0.75)):
            with self.subTest(error=type(error).__name__), \
                    patch.object(telemetry.subprocess, "run", side_effect=error):
                self.assertIsNone(telemetry._query_gpu_utilization(f"GPU-{UUID_ONE}"))

    def test_unsupported_malformed_multiple_or_out_of_range_readings_are_unavailable(self):
        for code, output in ((0, "N/A\n"), (0, ""), (1, "0\n"),
                             (0, "30\n40\n"), (0, "101\n"), (0, "-1\n")):
            with self.subTest(code=code, output=output), \
                    patch.object(telemetry.subprocess, "run", return_value=subprocess.CompletedProcess([], code, output, "")):
                self.assertIsNone(telemetry._query_gpu_utilization(f"GPU-{UUID_ONE}"))


class SamplerTests(unittest.TestCase):
    def test_reads_are_cached_and_close_interrupts_sampling_wait(self):
        sampled = threading.Event()

        def query(_):
            sampled.set()
            return 42

        sensor = telemetry.GpuUtilization("cuda:1")
        self.addCleanup(sensor.close)
        self.assertIsNone(sensor.read())
        with patch.object(telemetry, "_device_identifier", return_value=f"GPU-{UUID_TWO}"), \
                patch.object(telemetry, "_query_gpu_utilization", side_effect=query) as poll:
            self.assertIs(sensor.start(), sensor)
            self.assertTrue(sampled.wait(1))
            # Wait for the background thread to publish its sampled value.
            deadline = time.monotonic() + 1
            while sensor.read() is None and time.monotonic() < deadline:
                time.sleep(0.001)
            for _ in range(20):
                self.assertEqual(sensor.read(), 42)
            time.sleep(0.02)
            sensor.start()  # Repeated start must not create another sampler.
            self.assertEqual(poll.call_count, 1)
            started = time.monotonic()
            sensor.close()
            self.assertLess(time.monotonic() - started, 0.25)
            self.assertFalse(sensor._thread.is_alive())
        sensor.close()

    def test_no_device_identity_never_invokes_sensor_command(self):
        with patch.object(telemetry, "_device_identifier", return_value=None), \
                patch.object(telemetry, "_query_gpu_utilization") as poll:
            with telemetry.GpuUtilization("cuda:0") as sensor:
                sensor._thread.join(timeout=1)
                self.assertIsNone(sensor.read())
            poll.assert_not_called()


if __name__ == "__main__":
    unittest.main()
