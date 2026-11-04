# Standalone WR trainer

Open `LRAEP-WR-Trainer/train.ipynb`, restart its kernel if changing from an older
worker-based version, and run all cells. The training cell defines and calls
ordinary Python functions directly in the notebook kernel. It stays busy and
prints progress, epoch summaries, connection messages, and errors through normal
notebook output. It uses no worker-code string, training subprocess, or display
panel, and it does not import or execute `train.py`.

## Repeating schedule

The application reads its own `config.json`, decrypts only `S_3` with the
default key `qwe`, and follows the readable `training.job_sequence`. It does
not read `.env`, the shared trainer config, or a schedule embedded in the
notebook. Each new run reads the current config again.

The sequence is a continuous cycle: after the final interval, it goes back to
the first interval and repeats. The same CNN model, optimizer, and partial epoch
state survive across intervals and cycles. For example:

```json
"training": {
  "max_batches": 0,
  "max_seconds": 0,
  "job_sequence": [
    {"title": "job-0", "time": 600, "max_gpu_utilization": 100},
    {"title": "job-1", "time": 120, "max_gpu_utilization": 100}
  ]
},
"side_job": {
  "epochs": 0
}
```

This runs job-0 for ten minutes, trains job-1 for two minutes, and repeats until
manually stopped. Zero time, batch, and epoch limits mean unlimited operation.
Avoid `-1` exit entries for continuous training. A job percentage of `0` pauses
for that interval's duration, then the sequence advances normally. The legacy
explicit time, batch, epoch, and exit controls remain available for bounded runs.

A percentage of `100` disables intentional GPU pacing; actual utilization still
depends on workload and data loading. Each dictionary has exactly `title`,
`time` in seconds, and `max_gpu_utilization`; no tolerance is supported.
Intervals switch between batches; an in-flight CUDA
operation or network request can finish after its scheduled boundary. Temporary
S3 transport failures follow the existing reconnect behavior.

For percentages below `100`, CNN execution is paced between CUDA operations,
including forward, backward, and optimizer work. It rests after roughly 10 ms
of accumulated operation time instead of running an entire large batch before
resting. This preserves the configured batch size and model calculations.
Fresh hardware readings above the configured `max_gpu_utilization` hold back
new work for both jobs. The configured percentage also controls the operation-time
budget, with no random sampling or additional margin. Hardware readings refresh
about five times per second. Feedback waits respect interval boundaries and manual stopping;
stale or unavailable telemetry uses time-based pacing instead.

The target regulates average busy time. Hardware utilization samples remain
raw and may briefly exceed the target because an already-running operation
cannot be paused. This is not a driver-enforced instantaneous utilization cap.

## Stopping and kernel memory

Use the notebook's **Interrupt** button to stop manually. The training loop
closes its CNN inputs, S3 connection, telemetry, and VPN during cleanup.
Other cells wait until the training cell finishes. The old notebook
`trainer.stop_training()` worker controller is no longer used for direct runs.

CUDA state now belongs to the notebook kernel. Interrupting ends the active
training session, but the kernel may retain CUDA context and cached GPU memory.
Restart the kernel when you want to discard its state and completely reset its
GPU memory. Checkpoints saved to disk survive that restart. Rerun all cells to
resume training; preserving notebook variables is not required.

Both copies decode protocol strings, masks, and coefficient tables on demand;
decoded values are no longer stored as module globals. Native key and target
tensors are created for each operation without a persistent data cache. Each
completed or cancelled batch releases its matrices, candidate generator, and
proof references before pacing. Sessions close before side intervals or schedule
pauses, and cleanup collects unreachable objects and returns unused CUDA allocator
memory on the selected device. This shortens data lifetimes; it does not guarantee
that freed CPU or GPU memory has been overwritten. Runtime configuration remains
available for the active application. Restart an existing kernel or process to
load these changes and discard its previously loaded modules and state.

## Configuration and checkpoints

Use `trainer/config_manager.ipynb` to edit and save
`LRAEP-WR-Trainer/config.json` directly. Its update and replace helpers write
that destination. The shared [`README`](../README.md) describes those helpers
and the optional import from `trainer/config.json`.

Only the `S_3` value is encrypted. Its default decryption key is `qwe`; no
`.env`, key file, or startup prompt is required. All other settings, including
`side_job`, the dataset URL, Telegram, and VPN, remain readable JSON. The
loader returns decrypted connection settings as `S3` in memory. Use that
section name when passing dictionaries to the config manager helpers.
Existing plain configs remain readable for migration; the next save writes
`S_3` encrypted. Older whole-document exports can be read with an explicit
original key and saved in the new format. Config files stay ignored by git
and are saved atomically with owner-only permissions.
The standalone code has no dataset URL default; an active CNN interval
requires `side_job.repo_url` in the config.
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
reading private config or contacting a live S3.

Periodic Telegram training stats use `telegram.training_stats_tgBot_key`;
job lifecycle events use `telegram.tgBot_key`. Both retain the configured chat ID.
