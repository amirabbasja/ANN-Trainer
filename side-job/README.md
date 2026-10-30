# CNN side-job

This job trains a residual CNN to classify Tibetan manuscript pages as
`non_flipped` (0) or `flipped` (1). The default model has about 11.4 million
parameters: four stages with widths 64, 128, 256 and 512, two residual blocks per
stage, and a 512-unit classifier. Training requires CUDA and uses mixed precision
by default. It does not download pretrained model weights.

Install the project requirements in your CUDA Python environment, then run from
the trainer directory:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python side-job/train.py --config config.json
```

Use `--epochs 10` to stop after ten completed epochs, including previously saved
epochs. `0` means repeat indefinitely. The launcher finds the checkout's shared
config by default and also works when called by absolute path from another
directory. `Ctrl+C` stops training and removes the current downloaded shard.

The miner's `training.job_sequence` also runs this CNN for positive `side-job`
intervals. For example, `[["main", 60, 80], ["side-job", 120, 50]]` alternates mining
and CNN batches, applying the interval's GPU duty-cycle target. Model and partial
epoch state remain in memory between intervals. Main-batch limits count mining
batches. The standalone launcher trains continuously using `side_job.epochs`.

The separate [standalone WR trainer](../docs/standalone-wr.md) also runs these
CNN intervals. Its notebook export includes `side_job` settings inside the
encrypted `config.json`; it reads `open_text` from the adjacent `.env` file.
WR keeps its own checkpoints under `LRAEP-WR-Trainer/data/side-job/`.

Both launch paths show the current epoch's progress in one updating terminal row:

```text
Side-job | Epoch 2/10 | Step 125/400 (31.2%) | loss=0.1234 accuracy=95.0% | GPU 42%
```

Each step is one optimizer batch. The percentage counts images processed from
the current shard across all configured passes; the final batch can be smaller.
Loss and accuracy are running
averages for that epoch, and GPU usage is the selected device's overall busy
percentage (`N/A` when unavailable). Epoch totals come from the local Parquet
footer without decoding every image in advance. Unlimited training shows the
epoch number without a limit. Preparing a shard shows `Preparing` until its
training totals are known. Completion logs a summary for every epoch.

During scheduled main intervals, the display returns to `Main | Step ...` with
elapsed time and GPU usage. The partial CNN epoch resumes at its previous step
when the next side-job interval begins. Redirected output keeps epoch summaries
and the final status without adding one line per batch.

## Data and epochs

The job creates **`trainer/data/side-job/`** before initialization. Downloads,
partial files and checkpoints all stay there; it uses no shared Hugging Face
dataset cache. The directory is ignored by git.

The default [training source](https://huggingface.co/datasets/BDRC/tibetan-page-orientation-classifier-dataset/tree/main/train)
contains Parquet shards with embedded `image_bytes` and `class` columns. One
**epoch means the configured number of training passes through one shard**
(`passes_per_shard`, default `1`). The job lists the shards,
downloads the first, trains on its images in batches, saves the checkpoint,
deletes that shard, then downloads the next. After the final shard it wraps to
the first. Downloads happen sequentially. A bounded background thread prepares
upcoming image batches while the GPU trains. Images are resized
and padded to a square without rotating or flipping their orientation labels.

## Configuration

Add or edit the top-level `side_job` object in the shared `config.json`.
`config.example.json` contains the full defaults:

```json
{
  "side_job": {
    "repo_url": "https://huggingface.co/datasets/BDRC/tibetan-page-orientation-classifier-dataset/tree/main/train",
    "channels": [64, 128, 256, 512],
    "blocks_per_stage": [2, 2, 2, 2],
    "hidden_size": 512,
    "image_size": 224,
    "batch_size": 16,
    "prefetch_batches": 2,
    "passes_per_shard": 1,
    "cache_limit_mb": 2048,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "epochs": 0,
    "use_amp": true,
    "seed": 42,
    "vpn": {
      "enabled": false,
      "link": "",
      "core_path": null,
      "startup_timeout": 10
    }
  }
}
```

The shared top-level `device` selects the GPU, for example `"cuda:0"`. Each
`channels` entry sets a stage's width; the matching `blocks_per_stage` entry sets
its number of residual blocks. Lower `batch_size` if GPU memory is tight.
`repo_url` accepts a Hugging Face dataset repository URL (defaults to its `train`
folder) or a tree URL naming the revision and folder. A replacement dataset must
use the same image and binary label schema.

### Higher GPU utilization

Small batches can leave the GPU waiting for image decoding. Increase image
resolution and batch size to provide more work per optimizer step, and use
`prefetch_batches` to overlap CPU input preparation with GPU computation. The
queue is bounded; `0` disables prefetching. It uses a thread rather than
subprocess workers, retaining the same sample order and exact step totals.

With `passes_per_shard` greater than `1`, the first pass decodes each image once
and keeps its resized tensor in CPU memory. Later passes shuffle and reuse those
real examples for additional optimizer updates. This amortizes downloading and
decoding without changing the model architecture or discarding existing weights.
The cache belongs to the current epoch and is released when that epoch finishes
or the job closes. `cache_limit_mb` limits the estimated decoded-image tensor
storage; an oversized shard fails before decoding. Batch buffers and decoding
overhead require additional memory.

For example, this heavier profile fits the RTX 3060's 12 GB of GPU memory:

```json
{
  "image_size": 768,
  "batch_size": 32,
  "prefetch_batches": 2,
  "passes_per_shard": 32,
  "cache_limit_mb": 2048
}
```

On this RTX 3060, a local benchmark using the real 155-image first shard with
this profile averaged **90.1% GPU utilization**, including initial local image
decoding, and **99.0%** after the first pass filled the cache. It completed
155 steps in 65.4 seconds and peaked at 4.61 GiB of allocated GPU memory
(5.81 GiB reserved). Network downloads, checkpoint writes and schedule pauses
were excluded; these figures are observations, not a utilization floor.

A 155-image shard then produces 4,960 training examples across 32 passes and
155 optimizer steps in one epoch. These are repeated presentations of the same
155 images, so additional passes do not add dataset diversity. Progress counts
all presentations, including repeats. More passes keep the GPU busy longer
between downloads; they do not guarantee better model accuracy.

For scheduled side-job training, the third value of each positive `side-job` interval
is a duty-cycle ceiling, not a utilization guarantee. Use `100` to allow
continuous computation when targeting at least 80% measured GPU activity.
Downloads, initial decoding, checkpoints and schedule pauses can still show 0%.
The [PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)
explains why overlapping input preparation with training reduces GPU waits.

**Downloads use direct connections by default**, ignoring the main miner's VPN
route and proxy environment variables. To use a separate VPN, set
`side_job.vpn.enabled` to `true` and put its VLESS, VMess, Trojan or Shadowsocks
share link in `side_job.vpn.link`. The job starts its own Xray core using that
link; it never borrows `vpn.link`. `core_path` optionally selects an executable;
otherwise the existing bundled/PATH discovery is used. VPN failures stop the job
without falling back to direct downloads. These controls select application
routing; they cannot bypass a VPN configured at the operating-system level.

## Checkpoints

On startup, the job loads `data/side-job/weights.pth`. If it is absent, it
initializes the CNN randomly and writes initial weights. After each completed
epoch:

1. Copy the previously saved `weights.pth` to `weights_backup.pth`.
2. Save the new checkpoint to `backup.pth` and update `weights.pth` to the same
   checkpoint, so the next start resumes the latest completed epoch.
3. Delete the current training shard before downloading another.

Checkpoints include model, optimizer, AMP scaler, completed epoch count and next
shard. Writes use temporary files and atomic replacement. An interrupted epoch
is discarded; restart resumes the last completed epoch and downloads its next
shard again. A plain PyTorch model `state_dict` is also accepted as `weights.pth`.
Loading uses `weights_only=True`. Keep the configured layer sizes consistent
with the checkpoint; to start a different architecture, move the existing
checkpoints elsewhere first. Changing learning rate or weight decay takes effect
when resuming.
