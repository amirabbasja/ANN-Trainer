"""WR GPU report timing and work units, independent of remote pool statistics."""

from bisect import bisect_right
from contextlib import ExitStack
import copy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from LRAEP_trainer.config import DEFAULT_CONFIG
from LRAEP_trainer import pool_training as worker
from LRAEP_trainer.pool_client import JobSnapshot
from LRAEP_trainer.pool_protocol import PoolJob
from LRAEP_trainer.protocol import TrainingConfig


class WRReportSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.configuration = copy.deepcopy(DEFAULT_CONFIG)
        self.configuration['pool'].update(protocol='WR', host='pool.example',
                                            WR_wallet='test-wallet', KEX_uname='test-user',
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
        self.session.wait_job.return_value = JobSnapshot(PoolJob('job', bytes(76), 0, 3, 1), 1)
        self.session.is_current.return_value = True
        self.session.report_info.side_effect = self.report
        self.factory = self.stack.enter_context(
            patch('LRAEP_trainer.wr_client.WRSession', return_value=self.session))
        self.stack.enter_context(patch.object(worker, 'time', SimpleNamespace(monotonic=lambda: self.now)))
        self.stack.enter_context(patch.object(worker, 'load_official_helpers'))
        self.stack.enter_context(patch.object(worker.torch.cuda, 'is_available', return_value=True))
        self.stack.enter_context(patch.object(worker.torch.cuda, 'get_device_name', return_value='selected GPU'))
        self.stack.enter_context(patch.object(worker.native_cuda, 'available', return_value=False))
        self.stack.enter_context(patch.object(worker, 'pool_share_target', return_value=1))
        self.prepare = self.stack.enter_context(patch('LRAEP_trainer.wr_work.prepare_wr_batch'))
        self.stack.enter_context(patch.object(worker, 'prepare_batch'))
        self.search = self.stack.enter_context(patch.object(worker, 'search_batch', side_effect=self.complete_search))
        self.monitor = self.stack.enter_context(patch.object(worker, 'GpuUtilization'))
        self.monitor.return_value.start.return_value.read.return_value = 37
        self.progress = self.stack.enter_context(patch.object(worker, 'LiveProgress'))
        native = ModuleType('LRAEP_mining')
        native.IncompleteBlockHeader = SimpleNamespace(from_bytes=MagicMock())
        self.stack.enter_context(patch.dict(sys.modules, {'LRAEP_mining': native}))

    def report(self, gpu_name, hashrate):
        self.reports.append((self.now, gpu_name, hashrate))
        return len(self.reports)

    def complete_search(self, *_):
        self.now = next(self.search_times)
        return iter([])

    def test_registration_then_first_completed_batch_report_do_not_require_a_share(self):
        with self.assertLogs(worker.LOG, level='INFO') as logs:
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        work = 128 * 128 * 2048
        self.assertEqual(self.reports, [(0.0, 'selected GPU', 0.0),
                                        (1.0, 'selected GPU', float(work))])
        self.session.submit.assert_not_called()
        self.assertIn('submitted=0 accepted=0 rejected=0', ' '.join(logs.output))
        status = self.progress.return_value.__enter__.return_value.update.call_args.args[0]
        self.assertIn('accepted=0 rejected=0', status)

    def test_reports_refresh_during_startup_then_return_to_120_second_cadence(self):
        times = [1.0, 5.0, 11.0, 20.0, 21.0, 119.0, 120.0, 130.0, 238.0, 239.0, 359.0]
        self.configuration['training']['max_batches'] = len(times)
        self.search_times = iter(times)
        self.assertEqual(worker.WR_STARTUP_REPORT_INTERVAL, 10.0)
        self.assertEqual(worker.WR_STARTUP_REPORT_WINDOW, 120.0)
        self.assertEqual(worker.WR_REPORT_INTERVAL, 120.0)
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        work = 128 * 128 * 2048
        self.assertEqual([record[0] for record in self.reports],
                         [0.0, 1.0, 11.0, 21.0, 119.0, 239.0, 359.0])
        self.assertEqual([record[2] for record in self.reports],
                         [0.0, float(work), 3 * work / 11.0, 5 * work / 21.0,
                          6 * work / 119.0, 10 * work / 239.0, 11 * work / 359.0])
        self.assertEqual(self.search.call_count, len(times))
        self.session.submit.assert_not_called()

    def test_user_989_batch_startup_scenario_refreshes_189_gh_report_without_inflation(self):
        batches = 989
        work = 4096 * 4096 * 16384
        final_elapsed = batches * work / 4.53e12
        first_elapsed = work / 189.11e9
        times = [first_elapsed + (final_elapsed - first_elapsed) * index / (batches - 1)
                 for index in range(batches)]
        self.configuration['training'].update(m=4096, n=4096, k=16384, max_batches=batches)
        self.search_times = iter(times)
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(self.search.call_count, 989)
        self.assertEqual(len(self.reports), 7)
        self.assertAlmostEqual(self.reports[1][2] / 1e9, 189.11)
        self.assertGreater(self.reports[-1][0], 50.0)
        self.assertLess(self.now - self.reports[-1][0], 10.1)
        final_rate = batches * work / final_elapsed
        self.assertAlmostEqual(final_rate / 1e12, 4.53)
        self.assertGreater(self.reports[-1][2], 4.4e12)
        self.assertLessEqual(self.reports[-1][2], final_rate)
        self.assertLess(abs(final_rate - self.reports[-1][2]) / final_rate, 0.01)
        for timestamp, name, reported in self.reports:
            completed = bisect_right(times, timestamp)
            self.assertEqual(name, 'selected GPU')
            self.assertEqual(reported, completed * work / max(timestamp, 0.001))
        for previous, current in zip(self.reports[1:], self.reports[2:]):
            self.assertGreaterEqual(current[0] - previous[0], 10.0)
        status = self.progress.return_value.__enter__.return_value.update.call_args.args[0]
        self.assertIn('Hashrate(avg) 4.53 TH/s', status)
        self.assertIn('batches=989 attempts=259260416', status)
        self.session.submit.assert_not_called()

    def test_cancelled_work_only_registers_gpu_and_never_reports_uncompleted_work(self):
        def cancel(*_args, **_kwargs):
            self.factory.call_args.kwargs['stop_event'].set()
            return MagicMock()
        self.prepare.side_effect = cancel
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(self.reports, [(0.0, 'selected GPU', 0.0)])
        self.search.assert_not_called()
        self.session.submit.assert_not_called()

    def test_login_only_reports_no_gpu_or_hashrate(self):
        self.assertEqual(worker.run_pool_training(self.configuration, login_only=True), 0)
        self.assertEqual(self.reports, [])
        self.search.assert_not_called()
        self.monitor.assert_not_called()

    def test_kex_does_not_send_wr_reports(self):
        self.configuration['pool']['protocol'] = 'KEX'
        with patch('LRAEP_trainer.pool_client.PoolSession', return_value=self.session):
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(self.reports, [])
        self.session.report_info.assert_not_called()


class PoolHashrateUnitTests(unittest.TestCase):
    def test_raw_hs_and_console_ths_share_completed_mac_count(self):
        config = TrainingConfig(m=4096, n=4096, k=16384)
        stats = worker.PoolTrainingStats(batches=15, accepted=0)
        expected = 15 * 4096 * 4096 * 16384
        self.assertEqual(worker.pool_hashrate(stats, config, 1.0), expected)
        self.assertIn('Hashrate(avg) 4.12 TH/s', worker.pool_status(stats, config, 1.0, 50))
        stats.accepted = 100
        self.assertEqual(worker.pool_hashrate(stats, config, 1.0), expected)

    def test_zero_work_and_short_elapsed_values_stay_finite(self):
        config = TrainingConfig(m=128, n=128, k=2048)
        self.assertEqual(worker.pool_hashrate(worker.PoolTrainingStats(), config, 0), 0)
        work = 128 * 128 * 2048
        self.assertEqual(worker.pool_hashrate(worker.PoolTrainingStats(batches=1), config, 0), work / 0.001)


if __name__ == '__main__':
    unittest.main()
