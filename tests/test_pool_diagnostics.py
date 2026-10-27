"""Unknown pool requests preserve replayable evidence without logging payloads."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from LRAEP_trainer.pool_diagnostics import LOG, UnknownRequestRecorder


class UnknownRequestRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "unknown"
        self.settings = {"host": "pool.example", "port": 9000,
                         "WR_wallet": "PRIVATE_WALLET", "KEX_uname": "PRIVATE_UNAME",
                         "password": "PRIVATE_PASSWORD"}

    def test_exact_incoming_message_is_saved_with_private_permissions_and_user_notice(self):
        recorder = UnknownRequestRecorder(self.path, self.settings)
        frame = b"unknown-wire-frame\n"
        message = {"method": "future_request", "params": ["REMOTE_PRIVATE_VALUE"]}
        with self.assertLogs(LOG, level="CRITICAL") as logged:
            artifact = recorder(frame, message, "Unknown PearlHash server method")
        saved = json.loads(artifact.read_text())
        self.assertEqual(bytes.fromhex(saved["raw_frame_hex"]), frame)
        self.assertEqual(saved["decoded_message"], message)
        self.assertTrue(saved["training_stopped"])
        self.assertEqual(os.stat(artifact).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o700)
        self.assertNotIn("PRIVATE_WALLET", artifact.read_text())
        self.assertNotIn("PRIVATE_UNAME", artifact.read_text())
        self.assertNotIn("PRIVATE_PASSWORD", artifact.read_text())
        output = " ".join(logged.output)
        self.assertIn("training stopped", output)
        self.assertIn(str(artifact), output)
        self.assertNotIn("REMOTE_PRIVATE_VALUE", output)

    def test_multiple_events_do_not_overwrite_first_evidence(self):
        recorder = UnknownRequestRecorder(self.path, self.settings)
        with self.assertLogs(LOG):
            first = recorder(b"first\n", None, "malformed frame")
            second = recorder(b"second\n", None, "malformed frame")
        self.assertNotEqual(first, second)
        self.assertEqual(len(list(self.path.glob("*.json"))), 2)

    def test_recording_failure_notifies_user_and_is_not_silently_ignored(self):
        recorder = UnknownRequestRecorder(self.path, self.settings)
        with patch("LRAEP_trainer.pool_diagnostics.os.open", side_effect=OSError), \
                self.assertLogs(LOG, level="CRITICAL"), \
                self.assertRaisesRegex(RuntimeError, "training stopped"):
            recorder(b"frame", None, "unknown")

    def test_nonfatal_extension_notice_does_not_claim_training_stopped(self):
        recorder = UnknownRequestRecorder(self.path, self.settings)
        with self.assertLogs(LOG, level="WARNING") as logged:
            artifact = recorder(b"frame", {"method": "extension"}, "unknown", fatal=False)
        self.assertFalse(json.loads(artifact.read_text())["training_stopped"])
        self.assertNotIn("training stopped", " ".join(logged.output))


if __name__ == "__main__":
    unittest.main()
