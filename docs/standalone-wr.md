# Standalone WR trainer

Open `LRAEP-WR-Trainer/train.ipynb`, restart its kernel if changing from an older
worker-based version, and run all cells. The training cell defines and calls
ordinary Python functions directly in the notebook kernel. It stays busy and
prints progress, epoch summaries, connection messages, and errors through normal
notebook output. It uses no worker-code string, training subprocess, or display
panel, and it does not import or execute `train.py`.

## Repeating schedule

The application decrypts its own `config.json` with `open_text` from the adjacent
`.env` and follows that decoded `training.job_sequence`. It does not read the
shared authoring config at runtime or use a schedule embedded in the notebook.
Each new run reads the current config and key again.

The sequence is a continuous cycle: after the final interval, it goes back to
the first interval and repeats. The same CNN model, optimizer, and partial epoch
state survive across intervals and cycles. For example:

```json
"training": {
  "max_batches": 0,
  "max_seconds": 0,
  "job_sequence": [["main", 600, 100], ["side-job", 120, 100]]
},
"side_job": {
  "epochs": 0
}
```

This mines for ten minutes, trains the CNN for two minutes, and repeats until
manually stopped. Zero time, batch, and epoch limits mean unlimited operation.
Avoid `-1` exit entries for continuous training. A job percentage of `0` pauses
for that interval's duration, then the sequence advances normally. The legacy
explicit time, batch, epoch, and exit controls remain available for bounded runs.

A percentage of `100` disables intentional GPU pacing; actual utilization still
depends on workload and data loading. An optional fourth entry value sets the
percentage-point tolerance. Intervals switch between batches; an in-flight CUDA
operation or network request can finish after its scheduled boundary. Temporary
pool transport failures follow the existing reconnect behavior.

For percentages below `100`, CNN execution is paced between CUDA operations,
including forward, backward, and optimizer work. It rests after roughly 10 ms
of accumulated operation time instead of running an entire large batch before
resting. This preserves the configured batch size and model calculations.
Fresh hardware readings above the active entry's upper bound also hold back
new work for both jobs; for `70 ± 10`, that bound is `80`. The selected random
target still controls the operation-time budget. Hardware readings refresh
about five times per second. Feedback waits respect interval boundaries and manual stopping;
stale or unavailable telemetry uses time-based pacing instead.

The target regulates average busy time. Hardware utilization samples remain
raw and may briefly exceed the target because an already-running operation
cannot be paused. This is not a driver-enforced instantaneous utilization cap.

## Stopping and kernel memory

Use the notebook's **Interrupt** button to stop manually. The training loop
closes its CNN inputs, pool connection, telemetry, and VPN during cleanup.
Other cells wait until the training cell finishes. The old notebook
`trainer.stop_training()` worker controller is no longer used for direct runs.

CUDA state now belongs to the notebook kernel. Interrupting ends the active
training session, but the kernel may retain CUDA context and cached GPU memory.
Restart the kernel when you want to discard its state and completely reset its
GPU memory. Checkpoints saved to disk survive that restart. Rerun all cells to
resume training; preserving notebook variables is not required.

## Configuration and checkpoints

Maintain configuration in `trainer/config.json` and export it from
`trainer/config_manager.ipynb` to the standalone directory. The shared
[`README`](../README.md) describes editing, export, and the decode text stored
in the standalone `.env`. Documentation remains in `trainer`.

The encrypted standalone config includes `side_job` settings, the dataset
repository URL, and the CNN's independent VPN settings. Decryption reads the
lowercase `open_text` assignment from the adjacent `.env`, including when
launched from another directory. Plain JSON configs are rejected. The standalone
code has no dataset URL default; an active CNN interval requires
`side_job.repo_url` in the encrypted config. Validation and dataset download
errors omit repository values, and training logs omit repository and shard
paths. Config and key files are ignored by git and written with owner-only
permissions. Keep `.env` private because possession of it allows decryption.
The notebook setup installs missing general-purpose packages from requirements.

Completed CNN epochs save checkpoints under `LRAEP-WR-Trainer/data/side-job/`,
using the shared trainer's `weights.pth`, `backup.pth`, and `weights_backup.pth`
format. Copy existing shared checkpoints there before starting to continue them.
Stopping removes the current disposable dataset shard; the next run resumes
from the last completed epoch. The data directory is ignored by git.

## Optional terminal and worker API

`python train.py` remains available for terminal use; Ctrl+C stops it. Importing
that file also provides `trainer.start(root=None, login_only=False, output=None)`
for callers that explicitly want a separate background worker. It returns a
`subprocess.Popen` object. Omit `root` to use the directory beside `train.py`.
Set `login_only=True` to authenticate without CNN or GPU work. Starting a second
worker raises an error while one is running.

That separate API retains `trainer.stop_training(timeout=10)`, `trainer.status()`,
and `trainer.run(root=None, login_only=False)`. Worker process controls and GPU
memory release apply to that optional API; they are independent of the notebook's
direct execution. Its process-control tests use local fixture workers without
reading private config or contacting a live pool.
