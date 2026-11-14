"""Best-effort GPU duty-cycle pacing between S1 batches."""

import time

import torch


class GpuPacer:
    """Limit sustained S5 load by resting after each completed GPU batch.

    Preparation and search may enqueue asynchronous CUDA work, so synchronize
    before timing the rest. This is not a hardware limit: individual kernels
    can still reach 100%, and other processes on the GPU are not controlled.
    """

    def __init__(self, device, max_utilization):
        self.devices = device if isinstance(device, tuple) else (device,)
        self._started = None
        self._deadline = None
        self.max_utilization = None
        self.shared_gpu = False
        self.set_utilization(max_utilization)

    def set_utilization(self, percentage, *, shared=False):
        """Apply the current schedule interval and discard a previous rest."""
        if shared:
            if type(percentage) not in (int, float) or not 0.5 <= percentage <= 50:
                raise ValueError("shared GPU utilization must be between 0.5 and 50")
        elif type(percentage) is not int or not 1 <= percentage <= 100:
            raise ValueError("GPU utilization must be an integer between 1 and 100")
        if percentage != self.max_utilization or shared != self.shared_gpu:
            self.max_utilization = percentage
            self.shared_gpu = shared
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
                self._synchronize()
            return
        self._synchronize()
        now = time.monotonic()
        work_seconds = now - self._started
        self._deadline = now + work_seconds * (100 - self.max_utilization) / self.max_utilization
        self._started = None

    def _synchronize(self):
        for device in self.devices:
            if self.shared_gpu:
                torch.cuda.current_stream(device).synchronize()
            else:
                torch.cuda.synchronize(device)

    def wait(self, stopped, *, deadline=None):
        if self._deadline is None:
            return
        until = self._deadline if deadline is None else min(self._deadline, deadline)
        self._deadline = None
        stopped.wait(max(0, until - time.monotonic()))
