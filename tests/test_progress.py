"""Terminal behavior without requiring a terminal or changing global logging."""

import io
import os
import unittest
from unittest.mock import patch

from LRAEP_trainer.progress import LiveProgress, job_status


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

    def test_redirected_output_emits_throttled_progress_and_latest_on_close(self):
        stream = io.StringIO()
        with patch("LRAEP_trainer.progress.time.monotonic", side_effect=[10, 10.1]):
            progress = LiveProgress(stream)
            progress.update("1 | main | 80% | 00:01:00 | 123.45")
            progress.clear()
            progress.update("1 | main | 80% | 00:00:59 | 124")
        self.assertEqual(stream.getvalue(), "1 | main | 80% | 00:01:00 | 123.45\n")
        progress.close()
        progress.close()
        progress.update("after close")
        self.assertEqual(stream.getvalue(),
                         "1 | main | 80% | 00:01:00 | 123.45\n"
                         "1 | main | 80% | 00:00:59 | 124\n")

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


class JobStatusTests(unittest.TestCase):
    def test_main_rate_is_a_number_without_units(self):
        self.assertEqual(job_status(1, "main", 80, 60, 123.45),
                         "1 | main | 80% | 00:01:00 | 123.45")

    def test_side_speed_is_the_epoch_number(self):
        self.assertEqual(job_status(2, "side", 50, 30, 4),
                         "2 | side | 50% | 00:00:30 | 4")

    def test_zero_and_missing_values_remain_five_fields(self):
        self.assertEqual(job_status(3, "main", None, None, None),
                         "3 | main | N/A | --:--:-- | 0")
        self.assertEqual(job_status(3, "side-job", 0, -1, 0),
                         "3 | side | 0% | 00:00:00 | 0")

    def test_nonfinite_remaining_time_is_displayed_without_crashing(self):
        self.assertEqual(job_status(3, "main", 0, float("inf"), float("nan")),
                         "3 | main | 0% | --:--:-- | 0")

    def test_long_job_time_does_not_wrap_after_a_day(self):
        self.assertEqual(job_status(4, "main", 1, 90061, 1000.0),
                         "4 | main | 1% | 25:01:01 | 1000")


if __name__ == "__main__":
    unittest.main()
