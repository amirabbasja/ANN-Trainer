"""Best-effort GPU duty-cycle pacing between mining batches."""

import time

import torch


class GpuPacer:
    """Limit sustained miner load by resting after each completed GPU batch.

    Preparation and search may enqueue asynchronous CUDA work, so synchronize
    before timing the rest. This is not a hardware limit: individual kernels
    can still reach 100%, and other processes on the GPU are not controlled.
    """

    def __init__(self, device, max_utilization):
        self.device = device
        self._started = None
        self._deadline = None
        self.max_utilization = None
        self.set_utilization(max_utilization)

    def set_utilization(self, percentage):
        """Apply the current schedule interval and discard a previous rest."""
        if type(percentage) is not int or not 1 <= percentage <= 100:
            raise ValueError("GPU utilization must be an integer between 1 and 100")
        if percentage != self.max_utilization:
            self.max_utilization = percentage
            self._started = None
            self._deadline = None

    def begin(self):
        if self.max_utilization < 100:
            self._started = time.monotonic()

    def finish(self, *, cancelled=False):
        if self._started is None:
            if cancelled:
                # Unpaced work can still be queued when a stage is cancelled.
                # Drain it before a pause or shutdown begins.
                torch.cuda.synchronize(self.device)
            return
        torch.cuda.synchronize(self.device)
        now = time.monotonic()
        work_seconds = now - self._started
        self._deadline = now + work_seconds * (100 - self.max_utilization) / self.max_utilization
        self._started = None

    def wait(self, stopped, *, deadline=None):
        if self._deadline is None:
            return
        until = self._deadline if deadline is None else min(self._deadline, deadline)
        self._deadline = None
        stopped.wait(max(0, until - time.monotonic()))
