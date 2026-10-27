"""Offline tests of candidate handling, stale work, and submission controls."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from LRAEP_trainer import __main__ as cli


class TrainingControlFlowTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.output = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.args = argparse.Namespace(
            device="cuda:0", address="test-payout-address", m=128, n=128,
            k=2048, tile_h=8, tile_w=8, rpc_url="https://127.0.0.1:44107",
            rpc_ca=None, rpc_timeout=1, poll_interval=1, refresh_interval=60,
            max_batches=1, max_seconds=0, no_submit=False,
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
        pm = ModuleType("LRAEP_mining")
        self.verify = MagicMock(return_value=(True, "Verified"))
        pm.verify_plain_proof_for_cert_version = self.verify
        address_module = ModuleType("LRAEP_gateway.blockchain_utils.blockchain_utils")
        address_module.get_script_pubkey_from_p2tr_address = MagicMock()
        gateway = ModuleType("LRAEP_gateway")
        gateway.__path__ = []
        blockchain_utils = ModuleType("LRAEP_gateway.blockchain_utils")
        blockchain_utils.__path__ = []
        self.stack.enter_context(patch.dict(sys.modules, {
            "LRAEP_mining": pm,
            "LRAEP_gateway": gateway,
            "LRAEP_gateway.blockchain_utils": blockchain_utils,
            "LRAEP_gateway.blockchain_utils.blockchain_utils": address_module,
        }))
        self.stack.enter_context(patch.dict("os.environ", {
            "LRAEPD_RPC_USER": "test-user", "LRAEPD_RPC_PASSWORD": "test-password",
        }))
        self.stack.enter_context(patch.object(cli, "require_cuda"))
        self.stack.enter_context(patch.object(cli.native_cuda, "available", return_value=False))
        self.stack.enter_context(patch.object(cli, "NodeClient", side_effect=[self.node, MagicMock()]))
        self.stack.enter_context(patch.object(cli, "TipWatcher", return_value=self.watcher))
        self.stack.enter_context(patch.object(cli.signal, "signal"))
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

    def test_no_submit_certifies_and_saves_without_submitting(self):
        self.args.no_submit = True
        self.assertEqual(cli.train(self.args), 0)

        self.verify.assert_called_once_with(3, self.header, self.proof)
        self.build.assert_called_once_with(self.proof, self.template)
        self.node.submit.assert_not_called()
        self.assert_saved_candidate()


if __name__ == "__main__":
    unittest.main()
