"""Effort schedules must take precedence over starting mining resources."""

import copy
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from LRAEP_trainer.config import DEFAULT_CONFIG


class EntrypointEffortTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3] / 'LRAEP-WR-Trainer'
        spec = importlib.util.spec_from_file_location("wr_effort_entrypoint", root / "train.py")
        self.entrypoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.entrypoint)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        self.stack.enter_context(patch.dict(os.environ))
        self.temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["pool"].update(protocol="WR", host="pool.example", WR_wallet="test-wallet")
        self.config["vpn"].update(enabled=True, link="vless://dummy@example.invalid:443")
        self.stack.enter_context(patch("LRAEP_trainer.config.load_config", return_value=self.config))
        self.vpn = self.stack.enter_context(patch("LRAEP_trainer.vpn.ManagedVPN"))
        self.routed = self.stack.enter_context(patch("LRAEP_trainer.network.routed"))
        self.helpers = self.stack.enter_context(patch("LRAEP_trainer.protocol.load_official_helpers"))
        self.cuda = self.stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        self.worker = self.stack.enter_context(patch("LRAEP_trainer.pool_training.run_pool_training", return_value=0))

    def test_immediate_exit_skips_vpn_helpers_cuda_and_worker(self):
        self.config["training"]["job_sequence"] = [("decoy", 0, -1)]
        self.assertEqual(self.entrypoint.run(self.temporary), 0)
        self.vpn.assert_not_called()
        self.helpers.assert_not_called()
        self.cuda.assert_not_called()
        self.worker.assert_not_called()

    def test_pause_schedule_is_delegated_before_backend_initialization(self):
        self.config["training"]["job_sequence"] = [("main", 60, 0), ("main", 60, 80)]
        self.assertEqual(self.entrypoint.run(self.temporary), 0)
        self.worker.assert_called_once()
        self.assertEqual(self.worker.call_args.args[0]["training"]["job_sequence"],
                         [("main", 60, 0), ("main", 60, 80)])
        self.helpers.assert_not_called()
        self.cuda.assert_not_called()


if __name__ == "__main__":
    unittest.main()
