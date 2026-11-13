from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np
import torch

from .gpu import TrainingCancelled
from .gpu_pacing import GpuPacer


@dataclass
class MiningDevice:
    device: torch.device
    pacer: GpuPacer
    native: bool
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    batches: int = 0


def batch_executor(resources, devices):
    if len(devices) > 1:
        return resources.enter_context(ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="mining-gpu"))
    return None


def run_batches(executor, workers, prepare, search, threshold, cancelled):
    def run(worker):
        worker.pacer.begin()
        completed = False
        try:
            batch = prepare(worker)
            if cancelled():
                return None
            matches = search(batch, threshold, worker.device, cancelled)
            completed = True
            return batch, matches
        except TrainingCancelled:
            return None
        finally:
            worker.pacer.finish(cancelled=not completed)

    if executor is None:
        return [run(workers[0])]
    futures = [executor.submit(run, worker) for worker in workers]
    wait(futures)
    return [future.result() for future in futures]
