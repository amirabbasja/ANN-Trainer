"""Configuration and duty-cycle checks for the standalone WR trainer."""

import copy
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, call, patch

import LRAEP_trainer
from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, load_config, save_config
from LRAEP_trainer.config_export import write_open_text
from LRAEP_trainer import gpu_pacing, pool_training as worker


class GpuLimitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "config.json"
        write_open_text(self.path.parent / ".env", "unit test text key")
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["vpn"]["link"] = "vless://dummy@example.invalid:443"

    def test_encrypted_config_without_effort_fields_receives_default_schedule(self):
        self.config["training"].pop("main_effort")
        self.config["training"].pop("decoy_effort")
        save_config(self.path, self.config)
        training = load_config(self.path)["training"]
        self.assertEqual(training["main_effort"], [(60, 80), (120, 70), (180, 10)])
        self.assertEqual(training["decoy_effort"], [])
        self.assertNotIn("max_gpu_utilization", training)

    def test_encrypted_config_preserves_main_and_decoy_schedules(self):
        self.config["training"].update(main_effort=[[0.5, 100], [3, 0], [0, -1]],
                                       decoy_effort=[[2, 10]])
        save_config(self.path, self.config)
        training = load_config(self.path)["training"]
        self.assertEqual(training["main_effort"], [(0.5, 100), (3, 0), (0, -1)])
        self.assertEqual(training["decoy_effort"], [(2, 10)])

    def test_legacy_gpu_limit_becomes_constant_schedule(self):
        self.config["training"].pop("main_effort")
        self.config["training"]["max_gpu_utilization"] = 70
        save_config(self.path, self.config)
        training = load_config(self.path)["training"]
        self.assertEqual(training["main_effort"], [(60, 70)])
        self.assertNotIn("max_gpu_utilization", training)

    def test_explicit_main_effort_takes_precedence_over_legacy_limit(self):
        self.config["training"].update(main_effort=[[30, 40]], max_gpu_utilization=70)
        save_config(self.path, self.config)
        self.assertEqual(load_config(self.path)["training"]["main_effort"], [(30, 40)])

    def test_invalid_legacy_limits_are_rejected_even_with_main_schedule(self):
        for value in (0, 101, -1, True, 70.5, "70", None):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "training.max_gpu_utilization"):
                self.config["training"]["max_gpu_utilization"] = value
                save_config(self.path, self.config)

    def test_effort_schedule_validation_applies_to_both_fields(self):
        malformed = (None, True, "PRIVATE_EFFORT", {}, [1, 80], [[1]], [[1, 80, 2]],
                     [[0, 80]], [[-1, -1]], [[True, 80]], [["1", 80]],
                     [[float("inf"), 80]], [[float("nan"), 80]],
                     [[1, -2]], [[1, 101]], [[1, True]], [[1, 70.5]], [[1, "PRIVATE_EFFORT"]])
        for field in ("main_effort", "decoy_effort"):
            for value in malformed:
                config = copy.deepcopy(self.config)
                config["training"][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ConfigError, field) as caught:
                    save_config(self.path, config)
                self.assertNotIn("PRIVATE_EFFORT", str(caught.exception))
        self.config["training"]["main_effort"] = []
        with self.assertRaisesRegex(ConfigError, "main_effort"):
            save_config(self.path, self.config)

    def test_effort_defaults_are_independent(self):
        save_config(self.path, self.config)
        training = load_config(self.path)["training"]
        training["main_effort"].append((1, 100))
        training["decoy_effort"].append((1, 0))
        again = load_config(self.path)["training"]
        self.assertEqual(again["main_effort"], [(60, 80), (120, 70), (180, 10)])
        self.assertEqual(again["decoy_effort"], [])
        exported = copy.deepcopy(DEFAULT_CONFIG["training"])
        try:
            DEFAULT_CONFIG["training"]["main_effort"][0][1] = 99
            self.config.pop("training")
            save_config(self.path, self.config)
            self.assertEqual(load_config(self.path)["training"]["main_effort"][0], (60, 80))
        finally:
            DEFAULT_CONFIG["training"] = exported

    def test_pacer_waits_after_cuda_completion(self):
        stopped = Mock()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12, 12]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:0", 50)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        synchronize.assert_called_once_with("cuda:0")
        stopped.wait.assert_called_once_with(2)

    def test_pacer_100_percent_skips_sync_and_sleep(self):
        stopped = Mock()
        with patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:0", 100)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        synchronize.assert_not_called()
        stopped.wait.assert_not_called()


class WRPoolEffortTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["pool"].update(host="pool.example", WR_wallet="test-wallet", reconnect_delay=0)
        self.config["training"].update(m=128, n=128, max_batches=1)
        self.session = MagicMock()
        self.session.__enter__.return_value = self.session
        self.session.report_info.return_value = 1
        self.session.negotiated_gzip = False
        self.session.is_current.return_value = True
        self.session.submit.return_value = SimpleNamespace(accepted=True, error=None)
        self.session.wait_job.return_value = SimpleNamespace(
            job=SimpleNamespace(header_bytes=bytes(76), target=1, job_id="test", cert_version=3),
            generation=1,
        )
        self.factory = self.stack.enter_context(
            patch("LRAEP_trainer.wr_client.WRSession", return_value=self.session))
        self.stack.enter_context(patch.object(worker, "load_official_helpers"))
        self.stack.enter_context(patch.object(worker.torch.cuda, "is_available", return_value=True))
        self.stack.enter_context(patch.object(worker.torch.cuda, "get_device_name", return_value="test GPU"))
        self.stack.enter_context(patch.object(worker.torch.cuda, "synchronize"))
        self.stack.enter_context(patch.object(worker.native_cuda, "available", return_value=False))
        self.monitor = self.stack.enter_context(patch.object(worker, "GpuUtilization"))
        self.monitor.return_value.start.return_value.read.return_value = 42
        self.progress = self.stack.enter_context(patch.object(worker, "LiveProgress"))
        self.stack.enter_context(patch.object(worker, "pool_share_target", return_value=1))
        self.prepare = self.stack.enter_context(patch("LRAEP_trainer.wr_work.prepare_wr_batch"))
        self.verify = self.stack.enter_context(
            patch("LRAEP_trainer.wr_work.verified_wr_share", return_value="verified-share"))
        self.search = self.stack.enter_context(patch.object(
            worker, "search_batch", side_effect=lambda *_: iter([(0, 0, bytes(32))])))
        primitives = ModuleType("LRAEP_trainer.primitives")
        primitives.IncompleteBlockHeader = SimpleNamespace(from_bytes=Mock(return_value="header"))
        self.stack.enter_context(patch.dict(sys.modules, {"LRAEP_trainer.primitives": primitives}))
        self.stack.enter_context(patch.object(LRAEP_trainer, "primitives", primitives, create=True))

    def test_wr_pool_paces_after_submitting_and_before_the_next_batch(self):
        self.config["training"].update(max_batches=2, main_effort=[(60, 60)])
        order = []
        self.search.side_effect = lambda *_: (order.append("search"), iter([(0, 0, bytes(32))]))[1]
        self.session.submit.side_effect = lambda *_: (
            order.append("submit"), SimpleNamespace(accepted=True, error=None))[1]
        with patch.object(worker, "GpuPacer") as factory:
            pacer = factory.return_value
            pacer.max_utilization = 100
            pacer.set_utilization.side_effect = lambda value: setattr(pacer, "max_utilization", value)
            pacer.wait.side_effect = lambda *_a, **_kw: order.append("wait")
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(order, ["search", "submit", "wait", "search", "submit"])
        factory.assert_called_once_with(worker.torch.device("cuda:0"), 100)
        pacer.set_utilization.assert_called_once_with(60)
        self.assertEqual(pacer.begin.call_count, 2)
        self.assertEqual(pacer.finish.call_count, 2)
        pacer.wait.assert_called_once()

    def test_cancelled_search_is_paced_before_retrying(self):
        self.search.side_effect = [worker.TrainingCancelled("stale"), iter([])]
        with patch.object(worker, "GpuPacer") as factory:
            pacer = factory.return_value
            pacer.max_utilization = 80
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(pacer.finish.call_count, 2)
        pacer.wait.assert_called_once()
        self.assertIn("deadline", pacer.wait.call_args.kwargs)

    @contextmanager
    def clocked_schedule(self, schedule, *, batches=1, max_seconds=0):
        """Drive the real loop and scheduler without sleeping or touching CUDA."""
        clock = SimpleNamespace(now=0.0, pauses=[], on_wait=lambda: None)
        outer = self

        class ClockEvent:
            stopped = False

            def is_set(self):
                return self.stopped

            def set(self):
                self.stopped = True

            def wait(self, timeout=None):
                if self.stopped:
                    return True
                outer.assertIsNotNone(timeout, "test schedule waited indefinitely")
                clock.on_wait()
                clock.pauses.append((clock.now, timeout, outer.prepare.call_count,
                                     outer.search.call_count, outer.verify.call_count))
                clock.now += timeout
                return self.stopped

        stopped = ClockEvent()
        self.config["training"].update(main_effort=schedule, max_batches=batches,
                                                max_seconds=max_seconds)
        self.search.side_effect = lambda *_: iter([(0, 0, bytes(32))])
        with patch.object(worker.time, "monotonic", side_effect=lambda: clock.now), \
                patch.object(worker.threading, "Event", return_value=stopped), \
                patch.object(worker.threading, "Timer"), \
                patch.object(worker, "GpuPacer") as factory:
            pacer = factory.return_value
            pacer.max_utilization = 100
            pacer.set_utilization.side_effect = lambda value: setattr(pacer, "max_utilization", value)
            yield clock, pacer

    def test_100_percent_prepare_cancellation_drains_cuda_before_pause(self):
        with self.clocked_schedule([(10, 100), (20, 0), (10, 100)]) as (clock, _):
            order = []
            clock.on_wait = lambda: order.append("pause")

            def prepare(*_a, **_kw):
                order.append("prepare")
                if self.prepare.call_count == 1:
                    clock.now = 10

            self.prepare.side_effect = prepare
            worker.torch.cuda.synchronize.side_effect = lambda _device: order.append("synchronize")
            with patch.object(worker, "GpuPacer", gpu_pacing.GpuPacer):
                self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(order, ["prepare", "synchronize", "pause", "prepare"])
        worker.torch.cuda.synchronize.assert_called_once_with(worker.torch.device("cuda:0"))
        self.search.assert_called_once()
        self.verify.assert_called_once()
        self.session.submit.assert_called_once()

    def test_100_percent_search_cancellation_drains_cuda_before_exit_cleanup(self):
        with self.clocked_schedule([(10, 100), (0, -1)]) as (clock, _):
            order = []

            def search(*_):
                order.append("search")
                clock.now = 10
                raise worker.TrainingCancelled("schedule interval ended")

            self.search.side_effect = search
            worker.torch.cuda.synchronize.side_effect = lambda _device: order.append("synchronize")
            self.monitor.return_value.start.return_value.close.side_effect = lambda: order.append("close")
            with patch.object(worker, "GpuPacer", gpu_pacing.GpuPacer):
                self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(order, ["search", "synchronize", "close"])
        worker.torch.cuda.synchronize.assert_called_once_with(worker.torch.device("cuda:0"))
        self.session.__exit__.assert_called_once()
        self.verify.assert_not_called()
        self.session.submit.assert_not_called()

    def test_initial_pause_delays_mining_until_the_positive_interval(self):
        with self.clocked_schedule([(30, 0), (60, 70)]) as (clock, pacer):
            prepared_at = []
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(prepared_at, [30])
        self.assertEqual(clock.pauses, [(0, 30, 0, 0, 0)])
        pacer.set_utilization.assert_called_once_with(70)
        self.session.submit.assert_called_once()

    def test_pause_between_positive_intervals_performs_no_mining(self):
        with self.clocked_schedule([(10, 80), (20, 0), (10, 30)], batches=2) as (clock, pacer):
            prepared_at = []
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)
            pacer.wait.side_effect = lambda *_a, **kw: setattr(clock, "now", kw["deadline"])
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(prepared_at, [0, 30])
        self.assertEqual(clock.pauses, [(10, 20, 1, 1, 1)])
        self.assertEqual(pacer.set_utilization.call_args_list, [call(80), call(30)])
        self.assertEqual(self.session.submit.call_count, 2)

    def test_schedule_switches_effort_and_repeats_from_the_first_interval(self):
        with self.clocked_schedule([(10, 80), (20, 70)], batches=5) as (clock, pacer):
            prepared_at = []
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)
            pacer.wait.side_effect = lambda *_a, **kw: setattr(clock, "now", kw["deadline"])
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(prepared_at, [0, 10, 30, 40, 60])
        self.assertEqual(pacer.set_utilization.call_args_list,
                         [call(80), call(70), call(80), call(70), call(80)])
        self.assertEqual(self.search.call_count, 5)
        self.assertEqual(self.session.submit.call_count, 5)

    def test_zero_effort_until_time_limit_does_not_initialize_or_mine(self):
        with self.clocked_schedule([(30, 0)], max_seconds=10) as (clock, pacer):
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(clock.now, 10)
        worker.load_official_helpers.assert_not_called()
        self.factory.assert_not_called()
        self.prepare.assert_not_called()
        self.search.assert_not_called()
        self.verify.assert_not_called()
        self.session.submit.assert_not_called()
        self.monitor.assert_not_called()
        pacer.begin.assert_not_called()

    def test_terminal_first_interval_exits_before_helpers_cuda_or_sessions(self):
        self.config["training"]["main_effort"] = [(0, -1), (60, 80)]
        with patch.object(worker.torch.cuda, "is_available") as cuda:
            self.assertEqual(worker.run_pool_training(self.config), 0)
        cuda.assert_not_called()
        worker.load_official_helpers.assert_not_called()
        self.factory.assert_not_called()
        self.prepare.assert_not_called()
        self.search.assert_not_called()
        self.verify.assert_not_called()
        self.monitor.assert_not_called()

    def test_terminal_interval_interrupts_blocked_authentication_and_cleans_up(self):
        self.config["training"]["main_effort"] = [(0.3, 80), (0, -1)]

        def wait_for_exit():
            stopped = self.factory.call_args.kwargs["stop_event"]
            self.assertTrue(stopped.wait(2), "schedule did not stop authentication")
            return self.session

        self.session.__enter__.side_effect = wait_for_exit
        self.assertEqual(worker.run_pool_training(self.config), 0)
        self.factory.assert_called_once()
        self.assertLessEqual(self.factory.call_args.args[0]["connect_timeout"], 0.3)
        self.session.__exit__.assert_called_once()
        self.monitor.return_value.start.return_value.close.assert_called_once()
        self.progress.return_value.__exit__.assert_called_once()
        self.prepare.assert_not_called()
        self.search.assert_not_called()
        self.session.submit.assert_not_called()

    def test_terminal_interval_interrupts_blocked_share_submission_and_cleans_up(self):
        from LRAEP_trainer.pool_transport import PoolConnectionError

        self.config["training"]["main_effort"] = [(0.3, 80), (0, -1)]

        def wait_for_exit(*_):
            stopped = self.factory.call_args.kwargs["stop_event"]
            self.assertTrue(stopped.wait(2), "schedule did not stop share submission")
            raise PoolConnectionError("scheduled exit")

        self.session.submit.side_effect = wait_for_exit
        self.assertEqual(worker.run_pool_training(self.config), 0)
        self.factory.assert_called_once()
        self.session.submit.assert_called_once()
        self.session.__exit__.assert_called_once()
        self.monitor.return_value.start.return_value.close.assert_called_once()
        self.progress.return_value.__exit__.assert_called_once()

    def test_empty_decoy_effort_does_not_create_or_run_a_second_schedule(self):
        self.config["training"].update(main_effort=[(60, 70)], decoy_effort=[])
        with patch.object(worker, "EffortSchedule", wraps=worker.EffortSchedule) as schedules:
            self.assertEqual(worker.run_pool_training(self.config), 0)
        schedules.assert_called_once()
        self.assertEqual(schedules.call_args.args[0], [(60, 70)])
        self.prepare.assert_called_once()
        self.search.assert_called_once()
        self.session.submit.assert_called_once()



if __name__ == "__main__":
    unittest.main()
