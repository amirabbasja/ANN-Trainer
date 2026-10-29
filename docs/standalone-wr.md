# Standalone WR trainer

Open `LRAEP-WR-Trainer/train.ipynb` and run the setup and training cells. Training runs in a
managed Python worker, leaving the notebook available for other cells. Its
output appears below the training cell.

To stop all training and release the worker's GPU memory, run:

```python
trainer.stop_training()
```

This waits for training and its VPN/helper processes to exit. After 10 seconds
it forces termination if needed. To allow more time for graceful shutdown, use
`trainer.stop_training(timeout=30)`. It is safe to call the function repeatedly.
The worker owns its CUDA context, so exiting it releases its GPU allocations
and cached memory without restarting the notebook kernel. Other processes'
GPU allocations are unaffected.

`trainer.status()` returns `None` while the worker is running or before it has
started, and its exit code after it finishes. Zero means success. Rerun the
training cell to start again; it stops any previous worker first. In Python,
use `trainer.start(ROOT)` to start without blocking.

For a foreground terminal run, use `python train.py`; interrupt with Ctrl+C.


Configuration is maintained in `trainer/config.json` and exported from
`trainer/config_manager.ipynb` to the standalone directory. The shared
[`README`](../README.md) describes `training.job_sequence`, the encoded config,
and the decode text stored in the standalone `.env` file. The standalone copy
contains executable code and configuration; descriptions live in `trainer`.

The worker has its own CUDA context and process session. Stopping it signals
that session, including Xray and build or telemetry helpers, then waits for its
exit. If the notebook kernel exits unexpectedly, the worker's parent watcher
terminates the session. A monitor also cleans up remaining helpers as soon as
the worker exits. Output continues to drain even if the notebook output stream
has closed, so the worker cannot block on a full output pipe.

`trainer.start(root=None, login_only=False)` starts a background worker and
returns its `subprocess.Popen` object. Omit `root` to use the directory beside
`train.py`. Set `login_only=True` to check authentication without GPU work.
Starting a second worker raises an error while one is running. The notebook's
training cell calls `stop_training()` before loading and starting again.

`trainer.stop_training(timeout=10)` accepts a finite, nonnegative timeout. It
returns the worker's exit code, or `None` if this module has not started a
worker. `trainer.run(root=None, login_only=False)` runs in the current process
for foreground use. The process-control tests use a lightweight worker payload
to verify real process startup, shutdown, parent monitoring, and helper cleanup
without loading private configuration or contacting a pool. The tests reap
helpers after worker exit to avoid leaving zombie processes.
