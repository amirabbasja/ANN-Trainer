"""Decoy intervals select CNN batches without initializing or submitting mining work."""

import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from LRAEP_trainer import effort
from LRAEP_trainer.__main__ import TipWatcher


class ScheduledDecoyTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.clock = patch.object(effort.time, "monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        module = ModuleType("LRAEP_trainer.decoy")
        self.factory = Mock()
        module.DecoyJob = self.factory
        self.module_patch = patch.dict(sys.modules, {module.__name__: module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.job = self.factory.return_value
        self.job.step.return_value = True
        self.config = {"device": "cuda:0", "decoy_job": {"vpn": {"enabled": False}}}
        self.adapter = effort.ScheduledDecoy(self.config)
        self.addCleanup(self.adapter.close)
        self.event = Mock()
        self.event.is_set.return_value = False
        self.pacer = Mock(max_utilization=100)
        self.pacer.set_utilization.side_effect = lambda p: setattr(self.pacer, "max_utilization", p)
        self.schedule = effort.JobSequence([
            ("main", 10, 80), ("decoy", 20, 70), ("decoy", 5, 0),
        ], started=0)

    def test_main_pause_and_stopped_intervals_never_initialize_cnn(self):
        self.assertFalse(self.adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.now = 30
        self.assertFalse(self.adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.now = 10
        self.event.is_set.return_value = True
        self.assertFalse(self.adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.factory.assert_not_called()
        self.pacer.begin.assert_not_called()

    def test_decoy_job_and_partial_epoch_survive_main_intervals(self):
        self.now = 10
        self.assertTrue(self.adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.now = 35
        self.assertFalse(self.adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.now = 45
        self.assertTrue(self.adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.factory.assert_called_once_with(self.config)
        self.assertEqual(self.job.step.call_count, 2)
        self.pacer.set_utilization.assert_called_once_with(70)
        self.pacer.wait.assert_called_with(self.event, deadline=65)
        self.adapter.close()
        self.adapter.close()
        self.job.close.assert_called_once()

    def test_epoch_limit_waits_through_interval_without_busy_spin(self):
        self.now = 12
        self.job.step.return_value = False
        self.assertTrue(self.adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.event.wait.assert_called_once_with(18)
        self.event.wait.reset_mock()
        self.now = 13
        self.adapter.run_if_active(self.schedule, self.event, self.pacer, deadline=15)
        self.event.wait.assert_called_once_with(2)
        self.job.step.assert_called_once()

    def test_boundary_crossing_cancels_batch_before_gpu_work_after_download(self):
        self.now = 10

        def download_then_check(*, cancelled):
            self.assertFalse(cancelled())
            self.now = 30
            self.assertTrue(cancelled())
            return True

        self.job.step.side_effect = download_then_check
        self.assertTrue(self.adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.pacer.finish.assert_called_once_with(cancelled=True)
        self.pacer.wait.assert_not_called()

    def test_exception_drains_gpu_and_job_remains_available_for_cleanup(self):
        self.now = 10
        self.job.step.side_effect = RuntimeError("training failed")
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.adapter.run_if_active(self.schedule, self.event, self.pacer)
        self.pacer.finish.assert_called_once_with(cancelled=True)
        self.adapter.close()
        self.job.close.assert_called_once()


class DecoyWatcherTests(unittest.TestCase):
    def test_tip_watcher_waits_through_decoy_without_rpc(self):
        event = Mock()
        event.is_set.side_effect = [False, True]
        schedule = Mock()
        schedule.wait.return_value = True
        schedule.current_job.return_value = ("decoy", 70, 30)
        node = Mock(timeout=5)
        watcher = TipWatcher(node, 1, schedule=schedule, stopped=event, deadline=25)
        with patch.object(effort.time, "monotonic", return_value=10):
            watcher._run()
        node.best_hash.assert_not_called()
        event.wait.assert_called_once_with(15)


if __name__ == "__main__":
    unittest.main()
