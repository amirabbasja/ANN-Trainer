"""CNN training and checkpoint lifecycle against small, local fixtures."""

import copy
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from LRAEP_trainer.config import DEFAULT_CONFIG
from LRAEP_trainer.decoy import DecoyJob, OrientationCNN, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_previous_and_latest_checkpoints_rotate_and_load_safely(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_checkpoint(directory, {"epoch": 0, "tensor": torch.tensor([0])}, rotate=False)
            self.assertFalse((directory / "weights_backup.pth").exists())
            for epoch in (1, 2):
                save_checkpoint(directory, {"epoch": epoch, "tensor": torch.tensor([epoch])})
                for name, expected in (("weights.pth", epoch), ("backup.pth", epoch),
                                       ("weights_backup.pth", epoch - 1)):
                    state = torch.load(directory / name, weights_only=True)
                    self.assertEqual(state["epoch"], expected)
                    self.assertEqual(state["tensor"].item(), expected)
            self.assertEqual(sorted(path.name for path in directory.iterdir()),
                             ["backup.pth", "weights.pth", "weights_backup.pth"])

    def test_serialization_failure_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_checkpoint(directory, {"epoch": 1})
            with patch("LRAEP_trainer.decoy.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_checkpoint(directory, {"epoch": 2})
            self.assertEqual(torch.load(directory / "weights.pth", weights_only=True)["epoch"], 1)
            self.assertEqual(sorted(path.name for path in directory.iterdir()), ["backup.pth", "weights.pth"])

    def test_data_directory_created_before_gpu_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = DecoyJob(copy.deepcopy(DEFAULT_CONFIG), project_root=temporary)
            self.assertTrue((Path(temporary) / "data" / "decoy-job").is_dir())
            with patch("LRAEP_trainer.decoy.torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "CUDA"):
                    job.step()
            job.close()


class CNNTests(unittest.TestCase):
    def test_configurable_network_has_two_orientation_logits(self):
        model = OrientationCNN(channels=[8, 16], blocks_per_stage=[1, 1], hidden_size=16)
        output = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 2))
        torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
        self.assertIsNotNone(model.stem[0].weight.grad)

    def test_default_network_is_substantial_and_moderately_sized(self):
        model = OrientationCNN()
        self.assertGreater(sum(value.numel() for value in model.parameters()), 10_000_000)
        self.assertLess(sum(value.numel() for value in model.parameters()), 15_000_000)

    def test_single_image_batch_at_smallest_spatial_size_can_train(self):
        model = OrientationCNN(channels=[1] * 6, blocks_per_stage=[1] * 6, hidden_size=1)
        output = model(torch.randn(1, 3, 32, 32))
        torch.nn.functional.cross_entropy(output, torch.tensor([0])).backward()
        self.assertEqual(tuple(output.shape), (1, 2))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GPUTrainingTests(unittest.TestCase):
    def test_local_shards_train_rotate_delete_and_resume_on_gpu(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
            from PIL import Image
        except ImportError:
            self.skipTest("requires Pillow and pyarrow")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.parquet"
            stream = io.BytesIO()
            Image.new("RGB", (60, 30), "gray").save(stream, format="PNG")
            parquet.write_table(arrow.table({"image_bytes": [stream.getvalue()] * 4,
                                             "class": ["flipped", "non_flipped"] * 2}), fixture)
            config = copy.deepcopy(DEFAULT_CONFIG)
            config["decoy_job"].update(channels=[8, 16], blocks_per_stage=[1, 1],
                                       hidden_size=16, image_size=64, batch_size=2, epochs=2)
            source_paths, old_epoch_checks = [], []

            def download(shard, destination, **_):
                self.assertFalse(destination.exists(), "old shard must be removed before next download")
                source_paths.append(shard)
                if len(source_paths) > 1:
                    old_epoch_checks.append(torch.load(destination.parent / "weights.pth", weights_only=True)["epoch"])
                shutil.copyfile(fixture, destination)
                return destination

            with patch("LRAEP_trainer.decoy_data.DatasetSource") as source:
                source.return_value.list_shards.return_value = ["train/a.parquet", "train/b.parquet"]
                source.return_value.download_shard.side_effect = download
                job = DecoyJob(config, project_root=root)
                try:
                    self.assertTrue(job.step())
                    initial = torch.load(job.directory / "weights.pth", weights_only=True)
                    checks = iter([False, False, False, True])
                    self.assertTrue(job.step(cancelled=lambda: next(checks)))
                    self.assertIsNotNone(job.pending_batch)
                    self.assertEqual(job.examples, 2)
                    for _ in range(20):
                        if not job.step():
                            break
                    else:
                        self.fail("job did not stop at configured epoch limit")
                    self.assertEqual(job.epoch, 2)
                    self.assertEqual(next(job.model.parameters()).device.type, "cuda")
                    latest = torch.load(job.directory / "weights.pth", map_location="cpu", weights_only=True)
                    self.assertFalse(torch.equal(initial["model"]["stem.0.weight"].cpu(), latest["model"]["stem.0.weight"]))
                    self.assertEqual(torch.load(job.directory / "weights_backup.pth", weights_only=True)["epoch"], 1)
                    self.assertEqual(torch.load(job.directory / "backup.pth", weights_only=True)["epoch"], 2)
                    self.assertFalse((job.directory / "train.parquet").exists())
                finally:
                    job.close()
                self.assertEqual(source_paths, ["train/a.parquet", "train/b.parquet"])
                self.assertEqual(old_epoch_checks, [1])
                resumed = DecoyJob(config, project_root=root)
                try:
                    self.assertFalse(resumed.step())
                    self.assertEqual(resumed.epoch, 2)
                    self.assertTrue(torch.equal(resumed.model.stem[0].weight.detach().cpu(), latest["model"]["stem.0.weight"]))
                finally:
                    resumed.close()

    def test_interrupted_epoch_is_discarded_and_independent_vpn_is_used(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["decoy_job"].update(channels=[8], blocks_per_stage=[1], hidden_size=8, image_size=32)
        config["vpn"].update(enabled=True, link="MAIN")
        config["decoy_job"]["vpn"].update(enabled=True, link="DECOY")
        with tempfile.TemporaryDirectory() as temporary, \
             patch("LRAEP_trainer.decoy_data.DatasetSource") as source, \
             patch("LRAEP_trainer.vpn.ManagedVPN") as vpn:
            source.return_value.list_shards.return_value = ["train/a.parquet"]
            job = DecoyJob(config, project_root=temporary)
            try:
                calls = iter([False, True])
                self.assertTrue(job.step(cancelled=lambda: next(calls)))
                vpn.assert_called_once_with(config["decoy_job"]["vpn"])
                source.assert_called_once_with(config["decoy_job"]["repo_url"], proxy=vpn.return_value.__enter__.return_value)
                source.return_value.download_shard.assert_not_called()
            finally:
                job.close()
            vpn.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
