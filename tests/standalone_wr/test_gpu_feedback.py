import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from LRAEP_trainer import gpu_pacing, telemetry
from LRAEP_trainer.side_job import OrientationCNN


class FeedbackPacingTests(unittest.TestCase):
    def setUp(self):
        self.clock = SimpleNamespace(now=10.0)
        self.waits = []
        self.stopped = Mock()
        self.stopped.is_set.return_value = False

        def wait(seconds):
            self.waits.append(seconds)
            self.clock.now += seconds
            return False

        self.stopped.wait.side_effect = wait
        timer = patch.object(gpu_pacing.time, 'monotonic', side_effect=lambda: self.clock.now)
        timer.start()
        self.addCleanup(timer.stop)
        self.pacer = gpu_pacing.GpuPacer('cuda:0', 70)

    def test_high_measured_load_waits_until_the_sample_is_below_target(self):
        self.pacer.set_monitor(Mock(side_effect=[(100, 10.0), (95, 10.05), (65, 10.1)]))
        self.pacer.wait(self.stopped)
        self.assertEqual(len(self.waits), 2)
        self.assertAlmostEqual(sum(self.waits), 0.1)
        self.stopped.set.assert_not_called()

    def test_feedback_wait_ends_at_interval_boundary_without_stopping_the_sequence(self):
        self.pacer.set_monitor(lambda: (100, self.clock.now))
        self.pacer.wait(self.stopped, deadline=10.07)
        self.assertAlmostEqual(self.clock.now, 10.07)
        self.assertAlmostEqual(sum(self.waits), 0.07)
        self.stopped.set.assert_not_called()

    def test_feedback_allows_measurements_within_the_configured_tolerance(self):
        self.pacer.set_ceiling(80)
        self.pacer.set_monitor(Mock(side_effect=[(85, 10.0), (79, 10.05)]))
        self.pacer.wait(self.stopped)
        self.assertEqual(len(self.waits), 1)
        self.assertAlmostEqual(sum(self.waits), 0.05)

    def test_schedule_ceiling_tracks_the_active_interval_including_wraparound(self):
        from LRAEP_trainer.effort import JobSequence
        sequence = JobSequence([('side-job', 2, 70, 10), ('main', 3, 10, 5)], started=0)
        for moment, ceiling in ((0, 80), (2, 15), (5, 80), (7, 15)):
            with patch.object(gpu_pacing.time, 'monotonic', return_value=moment):
                sequence.current_job()
                self.assertEqual(sequence.ceiling, ceiling)

    def test_old_or_missing_samples_do_not_block_work(self):
        for sample in ((100, 0), (100, 11), (None, None), (100, float('nan')), (True, 10), ('100', 10)):
            with self.subTest(sample=sample):
                self.pacer.set_monitor(lambda: sample)
                self.pacer.wait(self.stopped)
        self.assertEqual(self.waits, [])

    def test_100_percent_disables_feedback(self):
        reader = Mock(return_value=(100, 10))
        self.pacer.set_monitor(reader)
        self.pacer.set_utilization(100)
        self.pacer.wait(self.stopped)
        reader.assert_not_called()
        self.assertEqual(self.waits, [])

    def test_stop_signal_interrupts_feedback_wait(self):
        self.pacer.set_monitor(lambda: (100, self.clock.now))
        self.stopped.wait.side_effect = lambda seconds: self.stopped.is_set.configure_mock(return_value=True)
        self.pacer.wait(self.stopped)
        self.stopped.wait.assert_called_once()

    def test_operation_mode_does_not_charge_cpu_loading_time_as_gpu_work(self):
        self.pacer.begin()
        with self.pacer.operations(self.stopped):
            self.clock.now += 10
            torch.zeros(2)
        with patch.object(gpu_pacing.torch.cuda, 'synchronize'):
            self.pacer.finish()
        self.pacer.wait(self.stopped)
        self.assertEqual(sum(self.waits), 0)

    def test_sensor_publishes_timestamped_values_and_clears_failed_samples(self):
        sensor = telemetry.GpuUtilization('cuda:0')
        observed = []

        def waited(_seconds):
            observed.append(sensor.read_sample())
            return len(observed) == 2

        with patch.object(telemetry, '_device_identifier', return_value='fixture'), \
                patch.object(telemetry, '_query_gpu_utilization', side_effect=[80, None]), \
                patch.object(sensor._stopped, 'wait', side_effect=waited):
            sensor._run()
        self.assertEqual(observed, [(80, 10.0), (None, None)])


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class OperationPacingGpuTests(unittest.TestCase):
    def test_forward_backward_and_optimizer_results_match_without_changing_batches(self):
        torch.manual_seed(42)
        baseline = OrientationCNN(channels=[4, 8], blocks_per_stage=[1, 1], hidden_size=8).cuda()
        paced = copy.deepcopy(baseline)
        images = torch.randn(2, 3, 32, 32, device='cuda')
        labels = torch.tensor([0, 1], device='cuda')
        baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.001)
        paced_optimizer = torch.optim.AdamW(paced.parameters(), lr=0.001)
        rng = torch.cuda.get_rng_state()
        baseline_loss = torch.nn.functional.cross_entropy(baseline(images), labels)
        baseline_loss.backward()
        baseline_optimizer.step()
        torch.cuda.set_rng_state(rng)
        pacer = gpu_pacing.GpuPacer('cuda:0', 70)
        pacer._quantum = 0.0001
        stopped = Mock()
        stopped.is_set.return_value = False
        stopped.wait.return_value = False
        pacer.begin()
        with pacer.operations(stopped):
            paced_loss = torch.nn.functional.cross_entropy(paced(images), labels)
            paced_loss.backward()
            paced_optimizer.step()
        pacer.finish()
        pacer.wait(stopped)
        self.assertTrue(torch.allclose(baseline_loss, paced_loss, atol=1e-6, rtol=1e-5))
        for name, expected in baseline.state_dict().items():
            self.assertTrue(torch.allclose(expected, paced.state_dict()[name], atol=1e-6, rtol=1e-5), name)
        self.assertGreater(stopped.wait.call_count, 1)
        self.assertTrue(any(call.args[0] > 0 for call in stopped.wait.call_args_list))

    def test_100_percent_operation_mode_adds_no_cuda_synchronization(self):
        pacer = gpu_pacing.GpuPacer('cuda:0', 100)
        stopped = Mock()
        value = torch.ones(2, device='cuda')
        with patch.object(gpu_pacing.torch.cuda, 'synchronize') as synchronize:
            pacer.begin()
            with pacer.operations(stopped):
                result = value + 1
            pacer.finish()
            pacer.wait(stopped)
        self.assertEqual(result.tolist(), [2, 2])
        synchronize.assert_not_called()
        stopped.wait.assert_not_called()


if __name__ == '__main__':
    unittest.main()
