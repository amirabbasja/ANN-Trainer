"""CUDA CNN training, with one disposable dataset shard per epoch."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import logging
from pathlib import Path
import os
import pickle
import shutil
import tempfile

import torch
from torch import nn

LOG = logging.getLogger("LRAEP_trainer.decoy")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "ann_trainer.py").is_file():
    PROJECT_ROOT = Path.cwd()


class _BatchNorm(nn.BatchNorm2d):
    """Use running statistics when a final batch has one spatial value."""

    def forward(self, values):
        if self.training and values.numel() // values.shape[1] == 1:
            return nn.functional.batch_norm(values, self.running_mean, self.running_var,
                                             self.weight, self.bias, training=False, eps=self.eps)
        return super().forward(values)


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
            _BatchNorm(outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            _BatchNorm(outputs),
        )
        self.skip = nn.Identity() if inputs == outputs and stride == 1 else nn.Sequential(
            nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False), _BatchNorm(outputs),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images):
        return self.activation(self.layers(images) + self.skip(images))


class OrientationCNN(nn.Module):
    """A configurable residual CNN; defaults to roughly 11.4 million parameters."""

    def __init__(self, channels=(64, 128, 256, 512), blocks_per_stage=(2, 2, 2, 2),
                 hidden_size=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, stride=2, padding=3, bias=False),
            _BatchNorm(channels[0]), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        layers = []
        inputs = channels[0]
        for stage, (outputs, count) in enumerate(zip(channels, blocks_per_stage, strict=True)):
            for block in range(count):
                layers.append(ResidualBlock(inputs, outputs, 2 if stage and block == 0 else 1))
                inputs = outputs
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], hidden_size),
            nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(hidden_size, 2),
        )

    def forward(self, images):
        return self.classifier(self.features(self.stem(images)))


def _atomic_copy(source, destination):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            with source.open("rb") as previous:
                shutil.copyfileobj(previous, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(directory, checkpoint, *, rotate=True):
    """Stage the full checkpoint before touching any previously saved weights."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "weights.pth"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if rotate and weights.exists():
            _atomic_copy(weights, directory / "weights_backup.pth")
        _atomic_copy(temporary, directory / "backup.pth")
        os.replace(temporary, weights)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DecoyJob:
    """Keep training state across scheduled intervals; create network clients lazily.

    All generated files stay under the trainer checkout's data/decoy-job. The
    explicit project_root argument supports embedding and isolated local tests.
    """

    def __init__(self, config, *, project_root=None):
        self.config = copy.deepcopy(config["decoy_job"])
        self.device = torch.device(config["device"])
        self.directory = Path(project_root or PROJECT_ROOT) / "data" / "decoy-job"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stack = ExitStack()
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.source = None
        self.shards = []
        self.iterator = None
        self.pending_batch = None
        self.shard_path = None
        self.epoch = 0
        self.next_shard = None
        self.examples = self.correct = 0
        self.total_loss = 0.0
        self.closed = False

    def _architecture(self):
        return {name: self.config[name] for name in ("channels", "blocks_per_stage", "hidden_size")}

    def _checkpoint(self):
        return {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(), "epoch": self.epoch,
            "next_shard": self.next_shard, "architecture": self._architecture(),
        }

    def _initialize(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("decoy job requires an available CUDA GPU")
        if self.device.index is not None and self.device.index >= torch.cuda.device_count():
            raise ValueError("decoy job device selects an unavailable CUDA GPU")
        from .decoy_data import DatasetSource

        torch.manual_seed(self.config["seed"])
        self.model = OrientationCNN(**self._architecture()).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config["use_amp"])
        weights = self.directory / "weights.pth"
        if weights.exists():
            try:
                state = torch.load(weights, map_location="cpu", weights_only=True)
                # A plain state_dict is also accepted for externally supplied weights.
                if isinstance(state, dict) and "model" in state:
                    if state.get("architecture", self._architecture()) != self._architecture():
                        raise ValueError("architecture mismatch")
                    self.model.load_state_dict(state["model"])
                    if "optimizer" in state:
                        self.optimizer.load_state_dict(state["optimizer"])
                    if "scaler" in state:
                        self.scaler.load_state_dict(state["scaler"])
                    self.epoch = state.get("epoch", 0)
                    self.next_shard = state.get("next_shard")
                    if type(self.epoch) is not int or self.epoch < 0:
                        raise ValueError("invalid saved epoch")
                    if self.next_shard is not None and not isinstance(self.next_shard, str):
                        raise ValueError("invalid saved shard")
                else:
                    self.model.load_state_dict(state)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
                raise RuntimeError("could not load decoy weights.pth; check its architecture and checkpoint format") from exc
            # Deliberate config changes to optimizer settings take effect on resume.
            for group in self.optimizer.param_groups:
                group["lr"] = self.config["learning_rate"]
                group["weight_decay"] = self.config["weight_decay"]
            LOG.info("Resumed decoy weights at epoch %d", self.epoch)
        else:
            save_checkpoint(self.directory, self._checkpoint(), rotate=False)
            LOG.info("Created new decoy weights")
        LOG.info("Decoy CNN: %s; %s parameters", torch.cuda.get_device_name(self.device),
                 f"{sum(parameter.numel() for parameter in self.model.parameters()):,}")
        if self.config["epochs"] and self.epoch >= self.config["epochs"]:
            return
        # No main-miner routing context or environment proxy is inherited.
        proxy = None
        if self.config["vpn"]["enabled"]:
            from .vpn import ManagedVPN
            proxy = self._stack.enter_context(ManagedVPN(self.config["vpn"]))
        self.source = DatasetSource(self.config["repo_url"], proxy=proxy)
        self.shards = self.source.list_shards()
        if not self.shards:
            raise RuntimeError("decoy dataset contains no training Parquet shards")

    def _start_epoch(self, cancelled=None):
        from .decoy_data import ShardDataset

        name = self.next_shard if self.next_shard in self.shards else self.shards[self.epoch % len(self.shards)]
        self.shard_path = self.directory / "train.parquet"
        # A process killed mid-epoch may have left its disposable shard behind.
        self.shard_path.unlink(missing_ok=True)
        for partial in self.directory.glob(".shard-*.part"):
            partial.unlink(missing_ok=True)
        self.source.download_shard(name, self.shard_path, cancelled=cancelled)
        dataset = ShardDataset(self.shard_path, image_size=self.config["image_size"],
                               seed=self.config["seed"] + self.epoch)
        self.iterator = iter(torch.utils.data.DataLoader(
            dataset, batch_size=self.config["batch_size"], num_workers=0, pin_memory=True,
        ))
        self.next_shard = self.shards[(self.shards.index(name) + 1) % len(self.shards)]
        self.examples = self.correct = 0
        self.total_loss = 0.0
        self.model.train()
        LOG.info("Decoy epoch %d: %s (one shard)", self.epoch + 1, name)

    def _finish_epoch(self):
        if not self.examples:
            raise RuntimeError("decoy training shard is empty")
        self.epoch += 1
        save_checkpoint(self.directory, self._checkpoint())
        LOG.info("Decoy epoch %d complete: loss=%.4f accuracy=%.1f%% examples=%d",
                 self.epoch, self.total_loss / self.examples,
                 100 * self.correct / self.examples, self.examples)
        self._clear_shard()

    def _clear_shard(self):
        # Closing the loader generator releases the Parquet file before unlinking.
        self.pending_batch = None
        if self.iterator is not None:
            iterator, self.iterator = self.iterator, None
            fetcher = getattr(iterator, "_dataset_fetcher", None)
            generator = getattr(fetcher, "dataset_iter", None)
            if generator is not None and hasattr(generator, "close"):
                generator.close()
        if self.shard_path is not None:
            self.shard_path.unlink(missing_ok=True)
            self.shard_path = None

    def step(self, *, cancelled=None):
        """Run one optimizer batch. False means the configured epoch limit is met."""
        if self.closed:
            raise RuntimeError("decoy job is closed")
        if cancelled is not None and cancelled():
            return True
        if self.model is None:
            self._initialize()
        if self.config["epochs"] and self.epoch >= self.config["epochs"]:
            return False
        if cancelled is not None and cancelled():
            return True
        if self.iterator is None:
            from .decoy_data import DatasetCancelled
            try:
                self._start_epoch(cancelled)
            except DatasetCancelled:
                self._clear_shard()
                return True
        if cancelled is not None and cancelled():
            return True
        try:
            if self.pending_batch is None:
                self.pending_batch = next(self.iterator)
        except StopIteration:
            self._finish_epoch()
            return not (self.config["epochs"] and self.epoch >= self.config["epochs"])
        if cancelled is not None and cancelled():
            return True
        images, labels = self.pending_batch
        self.pending_batch = None
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.config["use_amp"]):
            logits = self.model(images)
            loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("decoy training produced a nonfinite loss")
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        count = labels.numel()
        self.examples += count
        self.total_loss += loss.detach().item() * count
        self.correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        return True

    def close(self):
        """Discard the current shard; resume uses the last completed epoch."""
        try:
            self._clear_shard()
        finally:
            self._stack.close()
            self.optimizer = self.scaler = self.model = self.source = None
            self.closed = True


def main(argv=None):
    from .config import load_config
    from .paths import default_config_path

    parser = argparse.ArgumentParser(description="Train the decoy page-orientation CNN on CUDA")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--epochs", type=int, help="total completed epochs, including resumed epochs; 0 repeats")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    job = None
    try:
        config = load_config(args.config)
        if args.epochs is not None:
            if args.epochs < 0:
                parser.error("--epochs must be nonnegative")
            config["decoy_job"]["epochs"] = args.epochs
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        job = DecoyJob(config)
        while job.step():
            pass
        return 0
    except KeyboardInterrupt:
        LOG.info("Stopped; the last completed epoch is saved")
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        LOG.error("%s", exc)
        return 1
    finally:
        if job is not None:
            job.close()


if __name__ == "__main__":
    raise SystemExit(main())
