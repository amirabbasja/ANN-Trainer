"""Scope artifact suppression to a training run, including its worker threads."""

from contextlib import contextmanager
import logging
import os
import sys
import threading


_lock = threading.Lock()
_quiet_runs = 0
_previous = None


def artifacts_enabled():
    return _quiet_runs == 0


@contextmanager
def training_output(generate_artifacts):
    global _quiet_runs, _previous
    if generate_artifacts:
        yield
        return

    with _lock:
        if _quiet_runs == 0:
            _previous = (logging.root.manager.disable, sys.dont_write_bytecode,
                         os.environ.get("CUDA_CACHE_DISABLE"))
            logging.disable(sys.maxsize)
            sys.dont_write_bytecode = True
            os.environ["CUDA_CACHE_DISABLE"] = "1"
        _quiet_runs += 1
    try:
        yield
    finally:
        with _lock:
            _quiet_runs -= 1
            if _quiet_runs == 0:
                log_level, sys.dont_write_bytecode, cuda_cache_disable = _previous
                logging.disable(log_level)
                if cuda_cache_disable is None:
                    os.environ.pop("CUDA_CACHE_DISABLE", None)
                else:
                    os.environ["CUDA_CACHE_DISABLE"] = cuda_cache_disable
                _previous = None
