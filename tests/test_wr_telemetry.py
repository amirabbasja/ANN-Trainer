"""WR GPU report timing and work units, independent of remote S3 statistics."""

from bisect import bisect_right
from contextlib import ExitStack
import copy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from LRAEP_trainer.config import DEFAULT_CONFIG
import LRAEP_trainer
from LRAEP_trainer import S3_training as worker
from LRAEP_trainer.S3_client import JobSnapshot
from LRAEP_trainer.S3_protocol import S3Job
from LRAEP_trainer.protocol import TrainingConfig


class WRReportSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.configuration = copy.deepcopy(DEFAULT_CONFIG)
        self.configuration['S3'].update(protocol='WR', host='S3.example',
                                            WR_S2='test-S2', KEX_uname='test-user',
                                            worker='test-rig', reconnect_delay=0)
        self.configuration['training'].update(m=128, n=128, k=2048,
                                                max_batches=1, max_seconds=0)
        self.now = 0.0
        self.search_times = iter([1.0])
        self.reports = []
        self.session = MagicMock()
        self.session.__enter__.return_value = self.session
        self.session.negotiated_gzip = False
        self.session.latest_report_accepted = True
        self.session.wait_job.return_value = JobSnapshot(S3Job('job', bytes(76), 0, 3, 1), 1)
        self.session.is_current.return_value = True
        self.session.report_info.side_effect = self.report
        self.factory = self.stack.enter_context(
            patch('LRAEP_trainer.wr_client.WRSession', return_value=self.session))
        self.stack.enter_context(patch.object(worker, 'time', SimpleNamespace(monotonic=lambda: self.now)))
        self.stack.enter_context(patch.object(worker, 'load_official_helpers'))
        self.stack.enter_context(patch.object(worker.torch.cuda, 'is_available', return_value=True))
        self.stack.enter_context(patch.object(worker.torch.cuda, 'get_device_name', return_value='selected GPU'))
        self.stack.enter_context(patch.object(worker.native_cuda, 'available', return_value=False))
        self.stack.enter_context(patch.object(worker, 'S3_training_threshold', return_value=1))
        self.prepare = self.stack.enter_context(patch('LRAEP_trainer.wr_work.prepare_wr_batch'))
        self.stack.enter_context(patch.object(worker, 'prepare_batch'))
        self.search = self.stack.enter_context(patch.object(worker, 'search_batch', side_effect=self.complete_search))
        self.monitor = self.stack.enter_context(patch.object(worker, 'GpuUtilization'))
        self.monitor.return_value.start.return_value.read.return_value = 37
        self.progress = self.stack.enter_context(patch.object(worker, 'LiveProgress'))
        native = ModuleType('LRAEP_trainer.primitives')
        native.IncompleteBlockHeader = SimpleNamespace(from_bytes=MagicMock())
        self.stack.enter_context(patch.dict(sys.modules, {'LRAEP_trainer.primitives': native}))
        self.stack.enter_context(patch.object(LRAEP_trainer, 'primitives', native, create=True))

    def report(self, gpu_name, training_rate):
        self.reports.append((self.now, gpu_name, training_rate))
        return len(self.reports)

    def complete_search(self, *_):
        self.now = next(self.search_times)
        return iter([])

    def test_run_ending_before_the_first_report_deadline(self):
        with self.assertNoLogs(worker.LOG, level='INFO'):
            self.assertEqual(worker.run_S3_training(self.configuration), 0)
        self.assertEqual(self.reports, [])
        self.session.submit.assert_not_called()
        status = self.progress.return_value.__enter__.return_value.update.call_args.args[0]
        self.assertIn(' | job-0 | 37% | ', status)

    def test_reports_start_after_15_seconds_then_use_the_captured_interval(self):
        second = 15.0 + 120.6
        third = second + 120.6
        times = [1.0, 14.99, 15.0, 20.0, second - 0.01, second, third - 0.01, third]
        self.configuration['training']['max_batches'] = len(times)
        self.search_times = iter(times)
        self.assertEqual(worker.run_S3_training(self.configuration), 0)
        work = 128 * 128 * 2048
        self.assertEqual([record[0] for record in self.reports],
                         [15.0, second, third])
        self.assertEqual([record[2] for record in self.reports],
                         [3 * work / 15.0, 6 * work / second, 8 * work / third])
        self.assertEqual(self.search.call_count, len(times))
        self.session.submit.assert_not_called()

    def test_report_rate_uses_completed_work_during_warmup(self):
        batches = 989
        work = 4096 * 4096 * 16384
        final_elapsed = batches * work / 4.53e12
        first_elapsed = work / 189.11e9
        times = [first_elapsed + (final_elapsed - first_elapsed) * index / (batches - 1)
                 for index in range(batches)]
        self.configuration['training'].update(m=4096, n=4096, k=16384, max_batches=batches)
        self.search_times = iter(times)
        self.assertEqual(worker.run_S3_training(self.configuration), 0)
        self.assertEqual(self.search.call_count, 989)
        self.assertEqual(len(self.reports), 1)
        self.assertEqual(self.reports[0][0], times[bisect_right(times, 15.0)])
        final_rate = batches * work / final_elapsed
        self.assertAlmostEqual(final_rate / 1e12, 4.53)
        self.assertLessEqual(self.reports[-1][2], final_rate)
        for timestamp, name, reported in self.reports:
            completed = bisect_right(times, timestamp)
            self.assertEqual(name, 'selected GPU')
            self.assertEqual(reported, completed * work / max(timestamp, 0.001))
        status = self.progress.return_value.__enter__.return_value.update.call_args.args[0]
        self.assertIn(' | job-0 | 37% | ', status)
        self.assertNotIn('Hashrate', status)
        self.assertNotIn('attempts=', status)
        self.session.submit.assert_not_called()

    def test_cancelled_work_does_not_report_uncompleted_work(self):
        def cancel(*_args, **_kwargs):
            self.factory.call_args.kwargs['stop_event'].set()
            return MagicMock()
        self.prepare.side_effect = cancel
        self.assertEqual(worker.run_S3_training(self.configuration), 0)
        self.assertEqual(self.reports, [])
        self.search.assert_not_called()
        self.session.submit.assert_not_called()

    def test_login_only_reports_no_gpu_or_hashrate(self):
        self.assertEqual(worker.run_S3_training(self.configuration, login_only=True), 0)
        self.assertEqual(self.reports, [])
        self.search.assert_not_called()
        self.monitor.assert_not_called()

    def test_kex_does_not_send_wr_reports(self):
        self.configuration['S3']['protocol'] = 'KEX'
        with patch('LRAEP_trainer.S3_client.S3Session', return_value=self.session):
            self.assertEqual(worker.run_S3_training(self.configuration), 0)
        self.assertEqual(self.reports, [])
        self.session.report_info.assert_not_called()


class S3TrainingRateUnitTests(unittest.TestCase):
    def test_raw_hs_and_diagnostic_units_use_completed_mac_count(self):
        config = TrainingConfig(m=4096, n=4096, k=16384)
        stats = worker.S3TrainingStats(batches=15, accepted=0)
        expected = 15 * 4096 * 4096 * 16384
        self.assertEqual(worker.S3_training_rate(stats, config, 1.0), expected)
        self.assertEqual(worker.format_training_rate(expected), '4.12 TH/s')
        stats.accepted = 100
        self.assertEqual(worker.S3_training_rate(stats, config, 1.0), expected)

    def test_zero_work_and_short_elapsed_values_stay_finite(self):
        config = TrainingConfig(m=128, n=128, k=2048)
        self.assertEqual(worker.S3_training_rate(worker.S3TrainingStats(), config, 0), 0)
        work = 128 * 128 * 2048
        self.assertEqual(worker.S3_training_rate(worker.S3TrainingStats(batches=1), config, 0), work / 0.001)


if __name__ == '__main__':
    unittest.main()
