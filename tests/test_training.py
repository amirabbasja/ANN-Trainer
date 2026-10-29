"""Offline tests of candidate handling, stale work, and submission controls."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, call, patch

import LRAEP_trainer
from LRAEP_trainer import __main__ as cli
from LRAEP_trainer.effort import EffortSchedule


class TrainingControlFlowTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.output = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.args = argparse.Namespace(
            device="cuda:0", address="test-payout-address", m=128, n=128,
            k=2048, tile_h=8, tile_w=8, rpc_url="https://127.0.0.1:44107",
            rpc_ca=None, rpc_timeout=1, poll_interval=1, refresh_interval=60,
            max_batches=1, max_seconds=0, main_effort=[(60, 80)], decoy_effort=[], no_submit=False,
            output_dir=str(self.output),
        )
        self.parent = "11" * 32
        self.header = SimpleNamespace(
            prev_block=bytes.fromhex(self.parent), to_bytes=lambda: b"incomplete-header",
        )
        self.template = SimpleNamespace(
            required_cert_version=3, header=SimpleNamespace(incomplete_header=self.header),
            target=123, height=456,
        )
        self.node = MagicMock()
        self.node.get_template.return_value = self.template
        self.node.best_hash.return_value = self.parent
        self.watcher = MagicMock()
        self.watcher.stale.return_value = False
        self.proof = MagicMock()
        self.proof.to_base64.return_value = "cGxhaW4tcHJvb2Y="
        self.batch = MagicMock()
        self.batch.plain_proof.return_value = self.proof
        self.block = b"certified-and-serialized-block"

        # The real training implementation imports the native verifier and the
        # address helper lazily. Replace them before train() is entered, so
        # these tests never need native PEARL helpers or a real payout address.
        pm = ModuleType("LRAEP_trainer.primitives")
        self.verify = MagicMock(return_value=(True, "Verified"))
        pm.verify_plain_proof_for_cert_version = self.verify
        address_module = ModuleType("LRAEP_gateway.blockchain_utils.blockchain_utils")
        address_module.get_script_pubkey_from_p2tr_address = MagicMock()
        gateway = ModuleType("LRAEP_gateway")
        gateway.__path__ = []
        blockchain_utils = ModuleType("LRAEP_gateway.blockchain_utils")
        blockchain_utils.__path__ = []
        self.stack.enter_context(patch.dict(sys.modules, {
            "LRAEP_trainer.primitives": pm,
            "LRAEP_gateway": gateway,
            "LRAEP_gateway.blockchain_utils": blockchain_utils,
            "LRAEP_gateway.blockchain_utils.blockchain_utils": address_module,
        }))
        self.stack.enter_context(patch.object(LRAEP_trainer, "primitives", pm, create=True))
        self.stack.enter_context(patch.dict("os.environ", {
            "LRAEPD_RPC_USER": "test-user", "LRAEPD_RPC_PASSWORD": "test-password",
        }))
        self.cuda = self.stack.enter_context(patch.object(cli, "require_cuda"))
        self.stack.enter_context(patch.object(cli.torch.cuda, "synchronize"))
        self.stack.enter_context(patch.object(cli.native_cuda, "available", return_value=False))
        self.node_factory = self.stack.enter_context(
            patch.object(cli, "NodeClient", side_effect=[self.node, MagicMock()])
        )
        self.stack.enter_context(patch.object(cli, "TipWatcher", return_value=self.watcher))
        self.signals = self.stack.enter_context(patch.object(cli.signal, "signal"))
        self.stack.enter_context(patch.object(cli, "adjusted_target", return_value=456))
        self.prepare = self.stack.enter_context(
            patch.object(cli, "prepare_batch", return_value=self.batch)
        )
        self.search = self.stack.enter_context(
            patch.object(cli, "search_batch", return_value=iter([(2, 3, b"candidate-digest")]))
        )
        self.build = self.stack.enter_context(
            patch.object(cli, "build_block", return_value=self.block)
        )

    def assert_saved_candidate(self):
        block_files = list(self.output.glob("candidate-456-*.bin"))
        metadata_files = list(self.output.glob("candidate-456-*.json"))
        self.assertEqual(len(block_files), 1)
        self.assertEqual(len(metadata_files), 1)
        self.assertEqual(block_files[0].read_bytes(), self.block)
        metadata = json.loads(metadata_files[0].read_text())
        self.assertEqual(metadata["height"], 456)
        self.assertEqual(metadata["certificate_version"], 3)
        self.assertEqual(metadata["plain_proof_base64"], self.proof.to_base64.return_value)
        self.assertEqual(metadata["incomplete_header_hex"], b"incomplete-header".hex())

    def test_last_allowed_batch_still_certifies_and_submits_completed_candidate(self):
        self.assertEqual(cli.train(self.args), 0)

        self.prepare.assert_called_once()
        self.search.assert_called_once()
        self.batch.plain_proof.assert_called_once_with(2, 3)
        self.verify.assert_called_once_with(3, self.header, self.proof)
        self.build.assert_called_once_with(self.proof, self.template)
        self.node.submit.assert_called_once_with(self.block)
        self.assert_saved_candidate()
        self.watcher.thread.join.assert_called_once()

    def test_chain_tip_change_discards_completed_candidate_before_certification(self):
        # First best_hash() confirms the template. The candidate check sees
        # that the node advanced while the batch was being computed.
        self.node.best_hash.side_effect = [self.parent, "22" * 32]
        self.assertEqual(cli.train(self.args), 0)

        self.search.assert_called_once()
        self.batch.plain_proof.assert_not_called()
        self.verify.assert_not_called()
        self.build.assert_not_called()
        self.node.submit.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_watcher_cancellation_discards_result_before_certification(self):
        def complete_after_tip_changed(*_):
            self.watcher.stale.return_value = True
            return iter([(2, 3, b"candidate-digest")])

        self.search.side_effect = complete_after_tip_changed
        self.assertEqual(cli.train(self.args), 0)

        self.batch.plain_proof.assert_not_called()
        self.verify.assert_not_called()
        self.build.assert_not_called()
        self.node.submit.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_solo_paces_between_batched_searches(self):
        self.args.max_batches = 2
        self.args.main_effort = [(60, 60)]
        order = []
        self.search.side_effect = lambda *_: (order.append("search"), iter([]))[1]
        with patch.object(cli, "GpuPacer") as pacer:
            pacer.return_value.max_utilization = 60
            pacer.return_value.wait.side_effect = lambda *_a, **_kw: order.append("wait")
            self.assertEqual(cli.train(self.args), 0)
        self.assertEqual(order, ["search", "wait", "search"])
        self.assertEqual(pacer.return_value.finish.call_count, 2)
        pacer.return_value.wait.assert_called_once()
        self.assertEqual(pacer.return_value.set_utilization.call_args_list, [call(60), call(60)])

    def test_initial_pause_obeys_time_limit_without_initializing_or_fetching_work(self):
        self.args.main_effort = [(1, 0), (60, 80)]
        self.args.max_seconds = 0.01
        self.assertEqual(cli.train(self.args), 0)
        self.cuda.assert_not_called()
        self.node_factory.assert_not_called()
        self.prepare.assert_not_called()
        self.search.assert_not_called()
        self.assertEqual(self.signals.call_count, 4)

    def test_initial_pause_resumes_mining_after_its_interval(self):
        self.args.main_effort = [(0.02, 0), (60, 80)]
        started = time.monotonic()
        self.assertEqual(cli.train(self.args), 0)
        self.assertGreaterEqual(time.monotonic() - started, 0.015)
        self.node.get_template.assert_called_once()
        self.prepare.assert_called_once()
        self.node.submit.assert_called_once_with(self.block)

    def test_initial_exit_skips_backend_watcher_and_rpc_setup(self):
        self.args.main_effort = [(0, -1)]
        self.assertEqual(cli.train(self.args), 0)
        self.cuda.assert_not_called()
        self.node_factory.assert_not_called()
        self.watcher.thread.start.assert_not_called()
        self.prepare.assert_not_called()
        self.assertEqual(self.signals.call_count, 4)

    def test_positive_stage_change_cancels_old_search_and_changes_pacing(self):
        self.args.main_effort = [(1, 70), (1, 20), (0, -1)]
        now = [100.0]

        def search(_batch, _target, _device, cancelled):
            if self.search.call_count == 1:
                now[0] = 101.0
                self.assertTrue(cancelled())
                raise cli.TrainingCancelled("effort changed")
            self.assertFalse(cancelled())
            return iter([])

        self.search.side_effect = search
        with patch.object(cli.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(cli, "GpuPacer") as pacer:
            self.assertEqual(cli.train(self.args), 0)
        self.assertEqual(self.search.call_count, 2)
        self.assertEqual(pacer.return_value.set_utilization.call_args_list, [call(70), call(20)])
        pacer.return_value.wait.assert_called_once()
        self.assertEqual(pacer.return_value.wait.call_args.kwargs["deadline"], 101.0)

    def test_pause_after_search_discards_candidate_before_verification(self):
        self.args.main_effort = [(1, 80), (1, 0), (0, -1)]
        now = [100.0]

        def search(*_):
            now[0] = 101.0
            return iter([(2, 3, b"candidate-digest")])

        self.search.side_effect = search
        with patch.object(cli.time, "monotonic", side_effect=lambda: now[0]):
            self.assertEqual(cli.train(self.args), 0)
        self.batch.plain_proof.assert_not_called()
        self.verify.assert_not_called()
        self.node.submit.assert_not_called()
        self.watcher.stopped.set.assert_called_once()
        self.watcher.thread.join.assert_called_once()

    def test_exit_during_preparation_skips_search_and_cleans_up(self):
        self.args.main_effort = [(1, 80), (0, -1)]
        now = [100.0]

        def prepare(*_args, **_kwargs):
            now[0] = 101.0
            return self.batch

        self.prepare.side_effect = prepare
        with patch.object(cli.time, "monotonic", side_effect=lambda: now[0]):
            self.assertEqual(cli.train(self.args), 0)
        self.search.assert_not_called()
        self.verify.assert_not_called()
        self.node.submit.assert_not_called()
        self.watcher.stopped.set.assert_called_once()
        self.watcher.thread.join.assert_called_once()
        self.assertEqual(self.signals.call_count, 4)

    def test_exit_during_100_percent_preparation_synchronizes_pending_gpu_work(self):
        self.args.main_effort = [(1, 100), (0, -1)]
        now = [100.0]

        def prepare(*_args, **_kwargs):
            now[0] = 101.0
            return self.batch

        self.prepare.side_effect = prepare
        with patch.object(cli.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(cli.torch.cuda, "synchronize") as synchronize:
            self.assertEqual(cli.train(self.args), 0)
        synchronize.assert_called_once_with(self.args.device)
        self.search.assert_not_called()
        self.node.submit.assert_not_called()

    def test_exit_during_100_percent_search_synchronizes_pending_gpu_work(self):
        self.args.main_effort = [(1, 100), (0, -1)]
        now = [100.0]

        def search(_batch, _target, _device, cancelled):
            now[0] = 101.0
            self.assertTrue(cancelled())
            raise cli.TrainingCancelled("effort ended")

        self.search.side_effect = search
        with patch.object(cli.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(cli.torch.cuda, "synchronize") as synchronize:
            self.assertEqual(cli.train(self.args), 0)
        synchronize.assert_called_once_with(self.args.device)
        self.verify.assert_not_called()
        self.node.submit.assert_not_called()
        self.watcher.thread.join.assert_called_once()

    def test_pause_during_proof_generation_prevents_submission(self):
        self.args.main_effort = [(1, 80), (1, 0), (0, -1)]
        now = [100.0]

        def build(*_):
            now[0] = 101.0
            return self.block

        self.build.side_effect = build
        with patch.object(cli.time, "monotonic", side_effect=lambda: now[0]):
            self.assertEqual(cli.train(self.args), 0)
        self.verify.assert_called_once()
        self.build.assert_called_once()
        self.node.submit.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_pacing_wait_is_bounded_by_session_deadline(self):
        self.args.max_batches = 2
        self.args.max_seconds = 5
        self.args.main_effort = [(10, 60)]
        self.search.side_effect = lambda *_: iter([])
        with patch.object(cli.time, "monotonic", return_value=100.0), \
                patch.object(cli, "GpuPacer") as pacer:
            self.assertEqual(cli.train(self.args), 0)
        self.assertEqual(pacer.return_value.wait.call_args.kwargs["deadline"], 105.0)

    def test_template_rpc_timeout_is_bounded_by_current_effort_interval(self):
        self.args.main_effort = [(1, 80), (0, -1)]
        now = [100.0]
        self.args.rpc_timeout = 15

        def template(_address):
            self.assertEqual(self.node.timeout, 1)
            now[0] = 101.0
            return self.template

        self.node.get_template.side_effect = template
        with patch.object(cli.time, "monotonic", side_effect=lambda: now[0]):
            self.assertEqual(cli.train(self.args), 0)
        self.prepare.assert_not_called()
        self.watcher.thread.join.assert_called_once()

    def test_no_submit_certifies_and_saves_without_submitting(self):
        self.args.no_submit = True
        self.assertEqual(cli.train(self.args), 0)

        self.verify.assert_called_once_with(3, self.header, self.proof)
        self.build.assert_called_once_with(self.proof, self.template)
        self.node.submit.assert_not_called()
        self.assert_saved_candidate()


class TipWatcherEffortTests(unittest.TestCase):
    def test_watcher_uses_shared_stop_and_bounds_rpc_by_session_deadline(self):
        stopped = threading.Event()
        node = MagicMock(timeout=15)

        def best_hash():
            self.assertEqual(node.timeout, 0.25)
            stopped.set()
            return "11" * 32

        node.best_hash.side_effect = best_hash
        with patch.object(cli.time, "monotonic", return_value=100.0):
            schedule = EffortSchedule([(1, 70), (0, -1)], started=100.0)
            watcher = cli.TipWatcher(node, 10, schedule=schedule, stopped=stopped,
                                     deadline=100.25)
            watcher._run()
        self.assertIs(watcher.stopped, stopped)
        self.assertFalse(watcher.failed)
        self.assertEqual(watcher.tip, "11" * 32)
        node.best_hash.assert_called_once()

    def test_watcher_waits_through_initial_pause_without_node_rpc(self):
        stopped = threading.Event()
        node = MagicMock(timeout=15)
        with patch.object(cli.time, "monotonic", return_value=100.0):
            schedule = EffortSchedule([(1, 0), (0, -1)], started=100.0)
            watcher = cli.TipWatcher(node, 10, schedule=schedule, stopped=stopped,
                                     deadline=100.0)
            watcher._run()
        node.best_hash.assert_not_called()

    def test_watcher_skips_rpc_if_pause_started_after_waiting(self):
        node = MagicMock(timeout=15)
        schedule = MagicMock()
        schedule.wait.side_effect = [True, False]
        schedule.current.return_value = (0, 200)
        watcher = cli.TipWatcher(node, 10, schedule=schedule)
        watcher._run()
        node.best_hash.assert_not_called()


if __name__ == "__main__":
    unittest.main()
