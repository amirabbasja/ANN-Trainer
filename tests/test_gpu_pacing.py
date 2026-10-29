"""Deterministic checks for miner GPU duty-cycle pacing."""

import threading
import unittest
from unittest.mock import Mock, patch

from LRAEP_trainer import gpu_pacing


class GpuPacingTests(unittest.TestCase):
    def test_rest_scales_with_completed_work_and_synchronizes_cuda(self):
        stopped = Mock()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12, 12]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:2", 80)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        synchronize.assert_called_once_with("cuda:2")
        stopped.wait.assert_called_once_with(0.5)

    def test_time_spent_submitting_shares_counts_toward_rest(self):
        stopped = Mock()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12, 13]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize"):
            pacer = gpu_pacing.GpuPacer("cuda:0", 50)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        stopped.wait.assert_called_once_with(1)

    def test_time_limit_bounds_rest(self):
        stopped = Mock()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12, 12]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize"):
            pacer = gpu_pacing.GpuPacer("cuda:0", 25)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped, deadline=12.2)
        self.assertAlmostEqual(stopped.wait.call_args.args[0], 0.2)

    def test_stop_event_interrupts_a_long_rest(self):
        stopped = threading.Event()
        stopped.set()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12, 12]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize"):
            pacer = gpu_pacing.GpuPacer("cuda:0", 1)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        self.assertTrue(stopped.is_set())

    def test_100_disables_pacing_and_cuda_synchronization(self):
        stopped = Mock()
        with patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:0", 100)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        synchronize.assert_not_called()
        stopped.wait.assert_not_called()

    def test_cancelled_100_percent_batch_synchronizes_before_pause_or_exit(self):
        stopped = Mock()
        with patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:1", 100)
            pacer.begin()
            pacer.finish(cancelled=True)
            pacer.wait(stopped)
        synchronize.assert_called_once_with("cuda:1")
        stopped.wait.assert_not_called()

    def test_cancelled_paced_batch_still_synchronizes_and_computes_rest(self):
        stopped = Mock()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12, 12]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:0", 80)
            pacer.begin()
            pacer.finish(cancelled=True)
            pacer.wait(stopped)
        synchronize.assert_called_once_with("cuda:0")
        stopped.wait.assert_called_once_with(0.5)

    def test_interval_change_discards_old_rest_and_uses_new_percentage(self):
        stopped = Mock()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12, 20, 22, 22]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:0", 80)
            pacer.begin()
            pacer.finish()
            pacer.set_utilization(50)
            pacer.wait(stopped)
            stopped.wait.assert_not_called()
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        self.assertEqual(synchronize.call_count, 2)
        stopped.wait.assert_called_once_with(2)

    def test_reapplying_percentage_preserves_pending_rest(self):
        stopped = Mock()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12, 12]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize"):
            pacer = gpu_pacing.GpuPacer("cuda:0", 80)
            pacer.begin()
            pacer.finish()
            pacer.set_utilization(80)
            pacer.wait(stopped)
        stopped.wait.assert_called_once_with(0.5)

    def test_switching_to_100_drops_pending_rest(self):
        stopped = Mock()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:0", 50)
            pacer.begin()
            pacer.finish()
            pacer.set_utilization(100)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        self.assertEqual(synchronize.call_count, 1)
        stopped.wait.assert_not_called()

    def test_rejects_invalid_percentages(self):
        for value in (0, 101, -1, 80.5, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gpu_pacing.GpuPacer("cuda:0", value)


if __name__ == "__main__":
    unittest.main()
