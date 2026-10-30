"""Side-job intervals select CNN batches without initializing or submitting mining work."""

import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from LRAEP_trainer import effort


class ScheduledSideJobTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.clock = patch.object(effort.time, "monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        module = ModuleType("LRAEP_trainer.side_job")
        self.factory = Mock()
        module.SideJob = self.factory
        self.module_patch = patch.dict(sys.modules, {module.__name__: module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.job = self.factory.return_value
        self.job.step.return_value = True
        self.config = {"device": "cuda:0", "side_job": {"vpn": {"enabled": False}}}
        self.adapter = effort.ScheduledSideJob(self.config)
        self.addCleanup(self.adapter.close)
        self.event = Mock()
        self.event.is_set.return_value = False
        self.pacer = Mock(max_utilization=100)
        self.pacer.set_utilization.side_effect = lambda p: setattr(self.pacer, "max_utilization", p)
        self.schedule = effort.JobSequence([
            ("main", 10, 80), ("side-job", 20, 70), ("side-job", 5, 0),
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

    def test_side_job_and_partial_epoch_survive_main_intervals(self):
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

    def test_progress_is_published_before_work_and_again_before_pacing(self):
        clear = Mock()
        updates = []
        adapter = effort.ScheduledSideJob(self.config, before_log=clear,
                                        on_progress=lambda job: updates.append(job.step.call_count))
        self.addCleanup(adapter.close)
        self.now = 10

        def paced(*_args, **_kwargs):
            self.assertEqual(updates, [0, 1])

        self.pacer.wait.side_effect = paced
        self.assertTrue(adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.factory.assert_called_once_with(self.config, before_log=clear)
        clear.assert_called()
        self.assertEqual(updates, [0, 1])
        self.pacer.wait.assert_called_once()

    def test_completed_epoch_progress_is_published_before_waiting_for_interval(self):
        updates = []
        adapter = effort.ScheduledSideJob(self.config,
                                        on_progress=lambda job: updates.append(job.step.call_count))
        self.addCleanup(adapter.close)
        self.now = 10
        self.job.step.return_value = False

        def wait(_seconds):
            self.assertEqual(updates, [0, 1])

        self.event.wait.side_effect = wait
        self.assertTrue(adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.assertEqual(updates, [0, 1])
        self.event.wait.assert_called_once_with(20)
        self.now = 11
        self.event.wait.side_effect = lambda _seconds: self.assertEqual(updates, [0, 1, 1])
        self.assertTrue(adapter.run_if_active(self.schedule, self.event, self.pacer))
        self.assertEqual(updates, [0, 1, 1], "exhausted intervals must restore the completed epoch display")
        self.job.step.assert_called_once()

    def test_boundary_crossing_cancels_batch_before_gpu_work_after_download(self):
        self.now = 10

        def download_then_check(*, cancelled, **_options):
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


class SideJobWatcherTests(unittest.TestCase):
    def test_tip_watcher_waits_through_side_job_without_rpc(self):
        from LRAEP_trainer.__main__ import TipWatcher
        event = Mock()
        event.is_set.side_effect = [False, True]
        schedule = Mock()
        schedule.wait.return_value = True
        schedule.current_job.return_value = ("side-job", 70, 30)
        node = Mock(timeout=5)
        watcher = TipWatcher(node, 1, schedule=schedule, stopped=event, deadline=25)
        with patch.object(effort.time, "monotonic", return_value=10):
            watcher._run()
        node.best_hash.assert_not_called()
        event.wait.assert_called_once_with(15)


if __name__ == "__main__":
    unittest.main()
