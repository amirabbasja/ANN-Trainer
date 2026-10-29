"""Validated, repeating wall-clock schedules for mining effort."""

from contextlib import contextmanager
import math
import threading
import time


def normalize_effort(value, *, field="main_effort", allow_empty=False):
    """Copy pairs from JSON arrays or Python tuples without exposing values.

    Positive effort and pause intervals require positive, finite seconds. An
    exit entry accepts zero seconds too: reaching it exits immediately, so its
    duration is never consumed.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of (seconds, percentage) pairs")
    if not value and not allow_empty:
        raise ValueError(f"{field} must contain at least one interval")
    entries = []
    total = 0.0
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{field} intervals must contain seconds and percentage")
        seconds, percentage = pair
        if type(percentage) is not int or not -1 <= percentage <= 100:
            raise ValueError(f"{field} percentages must be integers between -1 and 100")
        try:
            finite = type(seconds) in (int, float) and math.isfinite(seconds)
        except OverflowError:
            finite = False
        if not finite or seconds < 0 or (seconds == 0 and percentage != -1):
            raise ValueError(f"{field} seconds must be positive and finite (zero is allowed for exit)")
        total += seconds
        if not math.isfinite(total):
            raise ValueError(f"{field} total duration must be finite")
        entries.append((seconds, percentage))
    return entries


class EffortSchedule:
    """Repeat intervals until stopped or the first exit entry is reached.

    All positions use monotonic wall time, so pauses, pacing and reconnections
    share the same timeline. An empty optional schedule is inactive.
    """

    def __init__(self, entries, *, started=None, allow_empty=False):
        self.entries = normalize_effort(entries, allow_empty=allow_empty)
        self.started = time.monotonic() if started is None else started
        self._total = sum(seconds for seconds, _ in self.entries)
        self._exit_offset = None
        elapsed = 0
        for seconds, percentage in self.entries:
            if percentage == -1:
                self._exit_offset = elapsed
                break
            elapsed += seconds

    @property
    def exit_deadline(self):
        """Absolute start of the first exit entry, or None for a repeating run."""
        return None if self._exit_offset is None else self.started + self._exit_offset

    def _current(self, now):
        elapsed = max(0, now - self.started)
        if self.exit_deadline is not None and now >= self.exit_deadline:
            return -1, math.inf
        if not self.entries:
            return 0, math.inf
        cycle, _ = divmod(elapsed, self._total)
        boundary = 0
        for seconds, percentage in self.entries:
            boundary += seconds
            until = self.started + cycle * self._total + boundary
            # Compare absolute timestamps: subtracting a large monotonic origin
            # can otherwise leave a completed fractional interval active.
            if now < until:
                return percentage, until
        # Floating-point rounding at a cycle boundary belongs to the next cycle.
        until = self.started + (cycle + 1) * self._total + self.entries[0][0]
        return self.entries[0][1], max(until, math.nextafter(now, math.inf))

    def current(self):
        """Return the active percentage and its absolute end time."""
        return self._current(time.monotonic())

    def ended(self):
        return self.current()[0] == -1

    def wait(self, stopped, *, deadline=None):
        """Wait through pauses; return whether mining may start right now."""
        while not stopped.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return False
            percentage, boundary = self._current(now)
            if percentage == -1:
                stopped.set()
                return False
            if percentage > 0:
                return True
            until = boundary if deadline is None else min(boundary, deadline)
            # Empty optional schedules can wait indefinitely for an external stop.
            stopped.wait(None if math.isinf(until) else min(threading.TIMEOUT_MAX, max(0, until - now)))
        return False

    @contextmanager
    def stopping(self, stopped):
        """Set the common stop event at exit, including during network waits."""
        timer = None
        if self._exit_offset is not None and not stopped.is_set():
            delay = self.exit_deadline - time.monotonic()
            if delay <= 0:
                stopped.set()
            else:
                callback = stopped.set
                if delay > threading.TIMEOUT_MAX:
                    # Event.wait cannot accept every finite float. Keep very
                    # distant exits cancellable without overflowing its timeout.
                    def wait_until_exit():
                        while not timer.finished.is_set():
                            remaining = self.exit_deadline - time.monotonic()
                            if remaining <= 0:
                                stopped.set()
                                return
                            timer.finished.wait(min(remaining, threading.TIMEOUT_MAX))
                    callback = wait_until_exit
                timer = threading.Timer(min(delay, threading.TIMEOUT_MAX), callback)
                timer.daemon = True
                timer.start()
        try:
            yield
        finally:
            if timer is not None:
                timer.cancel()
                if threading.current_thread() is not timer:
                    timer.join()
