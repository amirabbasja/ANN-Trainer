# CNN decoy job

This job trains a residual CNN to classify Tibetan manuscript pages as
`non_flipped` (0) or `flipped` (1). The default model has about 11.4 million
parameters: four stages with widths 64, 128, 256 and 512, two residual blocks per
stage, and a 512-unit classifier. Training requires CUDA and uses mixed precision
by default. It does not download pretrained model weights.

Install the project requirements in your CUDA Python environment, then run from
the trainer directory:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python decoy-job/train.py --config config.json
```

Use `--epochs 10` to stop after ten completed epochs, including previously saved
epochs. `0` means repeat indefinitely. The launcher finds the checkout's shared
config by default and also works when called by absolute path from another
directory. `Ctrl+C` stops training and removes the current downloaded shard.

The miner's `training.job_sequence` also runs this CNN for positive `decoy`
intervals. For example, `[["main", 60, 80], ["decoy", 120, 50]]` alternates mining
and CNN batches, applying the interval's GPU duty-cycle target. Model and partial
epoch state remain in memory between intervals. Main-batch limits count mining
batches. The standalone launcher trains continuously using `decoy_job.epochs`.

## Data and epochs

The job creates **`trainer/data/decoy-job/`** before initialization. Downloads,
partial files and checkpoints all stay there; it uses no shared Hugging Face
dataset cache. The directory is ignored by git.

The default [training source](https://huggingface.co/datasets/BDRC/tibetan-page-orientation-classifier-dataset/tree/main/train)
contains Parquet shards with embedded `image_bytes` and `class` columns. One
**epoch means one complete pass through one shard**. The job lists the shards,
downloads the first, trains on its images in small batches, saves the checkpoint,
deletes that shard, then downloads the next. After the final shard it wraps to
the first. Downloads happen sequentially without prefetching. Images are resized
and padded to a square without rotating or flipping their orientation labels.

## Configuration

Add or edit the top-level `decoy_job` object in the shared `config.json`.
`config.example.json` contains the full defaults:

```json
{
  "decoy_job": {
    "repo_url": "https://huggingface.co/datasets/BDRC/tibetan-page-orientation-classifier-dataset/tree/main/train",
    "channels": [64, 128, 256, 512],
    "blocks_per_stage": [2, 2, 2, 2],
    "hidden_size": 512,
    "image_size": 224,
    "batch_size": 16,
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

**Downloads use direct connections by default**, ignoring the main miner's VPN
route and proxy environment variables. To use a separate VPN, set
`decoy_job.vpn.enabled` to `true` and put its VLESS, VMess, Trojan or Shadowsocks
share link in `decoy_job.vpn.link`. The job starts its own Xray core using that
link; it never borrows `vpn.link`. `core_path` optionally selects an executable;
otherwise the existing bundled/PATH discovery is used. VPN failures stop the job
without falling back to direct downloads. These controls select application
routing; they cannot bypass a VPN configured at the operating-system level.

## Checkpoints

On startup, the job loads `data/decoy-job/weights.pth`. If it is absent, it
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
