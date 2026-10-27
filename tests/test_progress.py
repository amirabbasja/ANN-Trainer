"""Terminal behavior without requiring a terminal or changing global logging."""

import io
import os
import unittest
from unittest.mock import patch

from LRAEP_trainer.progress import LiveProgress


class TerminalStream(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.width = patch("LRAEP_trainer.progress.shutil.get_terminal_size",
                           return_value=os.terminal_size((120, 24)))
        self.terminal_size = self.width.start()
        self.addCleanup(self.width.stop)

    def test_terminal_updates_reuse_a_line_and_clear_shorter_text(self):
        stream = TerminalStream()
        progress = LiveProgress(stream, interval=0)
        progress.update("a long status line")
        progress.update("short")
        self.assertEqual(stream.getvalue(), "\r\x1b[2Ka long status line\r\x1b[2Kshort")
        self.assertNotIn("\n", stream.getvalue())
        progress.close()
        self.assertTrue(stream.getvalue().endswith("short\n"))
        self.assertEqual(stream.getvalue().count("\n"), 1)

    def test_throttle_still_keeps_the_latest_status_for_close(self):
        stream = TerminalStream()
        with patch("LRAEP_trainer.progress.time.monotonic", side_effect=[10, 10.1, 11.1]):
            progress = LiveProgress(stream)
            progress.update("one")
            progress.update("two")
            self.assertNotIn("two", stream.getvalue())
            progress.update("three")
        self.assertIn("three", stream.getvalue())
        progress.close()

        stream = TerminalStream()
        with patch("LRAEP_trainer.progress.time.monotonic", side_effect=[20, 20.1]):
            with LiveProgress(stream) as progress:
                progress.update("old")
                progress.update("new")
        self.assertTrue(stream.getvalue().endswith("new\n"))

    def test_clear_allows_logs_then_immediate_status_redraw(self):
        stream = TerminalStream()
        with patch("LRAEP_trainer.progress.time.monotonic", side_effect=[10, 10.1]):
            with LiveProgress(stream) as progress:
                progress.update("old status")
                progress.clear()
                stream.write("a warning\n")
                progress.update("new status")
        output = stream.getvalue()
        self.assertIn("\r\x1b[2Ka warning\n\r\x1b[2Knew status", output)

    def test_terminal_width_avoids_wrapping_including_wide_characters(self):
        self.terminal_size.return_value = os.terminal_size((8, 24))
        stream = TerminalStream()
        with LiveProgress(stream) as progress:
            progress.update("GPU 漢字 extra")
        # 7 columns available: ASCII prefix is 4, and one CJK glyph uses 2.
        self.assertEqual(stream.getvalue(), "\r\x1b[2KGPU 漢\n")

    def test_control_characters_cannot_inject_lines_or_escape_sequences(self):
        stream = TerminalStream()
        with LiveProgress(stream) as progress:
            progress.update("a\n\r\tb\x00\x1b[31mred\x1b[0m\x1b]0;title\x07")
        self.assertEqual(stream.getvalue(), "\r\x1b[2Ka   bred\n")

    def test_redirected_output_only_writes_latest_status_on_close(self):
        stream = io.StringIO()
        progress = LiveProgress(stream)
        progress.update("old")
        progress.clear()
        progress.update("latest")
        self.assertEqual(stream.getvalue(), "")
        progress.close()
        progress.close()
        progress.update("after close")
        self.assertEqual(stream.getvalue(), "latest\n")

    def test_context_manager_closes_on_exception_without_swallowing_it(self):
        stream = TerminalStream()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with LiveProgress(stream) as progress:
                progress.update("running")
                raise RuntimeError("failure")
        self.assertTrue(stream.getvalue().endswith("running\n"))

    def test_no_updates_means_no_output(self):
        stream = TerminalStream()
        with LiveProgress(stream):
            pass
        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
