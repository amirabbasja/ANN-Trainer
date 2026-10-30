"""CNN training and checkpoint lifecycle against small, local fixtures."""

import copy
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from LRAEP_trainer.config import DEFAULT_CONFIG
from LRAEP_trainer.side_job import SideJob, OrientationCNN, main, save_checkpoint


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
            with patch("LRAEP_trainer.side_job.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_checkpoint(directory, {"epoch": 2})
            self.assertEqual(torch.load(directory / "weights.pth", weights_only=True)["epoch"], 1)
            self.assertEqual(sorted(path.name for path in directory.iterdir()), ["backup.pth", "weights.pth"])

    def test_data_directory_created_before_gpu_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = SideJob(copy.deepcopy(DEFAULT_CONFIG), project_root=temporary)
            self.assertTrue((Path(temporary) / "data" / "side-job").is_dir())
            with patch("LRAEP_trainer.side_job.torch.cuda.is_available", return_value=False):
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


class ProgressTests(unittest.TestCase):
    def test_unstarted_job_has_no_invented_training_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = SideJob(copy.deepcopy(DEFAULT_CONFIG), project_root=temporary)
            try:
                status = job.status()
                self.assertIn("Epoch 1", status)
                self.assertIn("Preparing", status)
                self.assertIn("GPU N/A", status)
                self.assertNotIn("loss=", status)
                self.assertNotIn("accuracy=", status)
            finally:
                job.close()

    def test_standalone_reports_optimizer_steps_and_closes_display_and_job(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["device"] = "cuda"
        with patch("LRAEP_trainer.config.load_config", return_value=config), \
             patch("LRAEP_trainer.side_job.torch.cuda.current_device", return_value=2), \
             patch("LRAEP_trainer.side_job.SideJob") as factory, \
             patch("LRAEP_trainer.progress.LiveProgress") as display, \
             patch("LRAEP_trainer.telemetry.GpuUtilization") as monitor:
            job = factory.return_value
            job.step.side_effect = [True, True, False]
            job.status.side_effect = ["Preparing", "Step 1/2", "Step 2/2 (100.0%)", "Epoch 1 Complete"]
            progress = display.return_value.__enter__.return_value
            monitor.return_value.__enter__.return_value.read.return_value = 40
            self.assertEqual(main([]), 0)
            self.assertEqual([call.args[0] for call in progress.update.call_args_list],
                             ["Preparing", "Step 1/2", "Step 2/2 (100.0%)", "Epoch 1 Complete"])
            self.assertIs(factory.call_args.kwargs["before_log"], progress.clear)
            monitor.assert_called_once_with("cuda:2")
            display.return_value.__exit__.assert_called_once()
            monitor.return_value.__exit__.assert_called_once()
            job.close.assert_called_once()

    def test_standalone_cleans_up_after_interrupt(self):
        with patch("LRAEP_trainer.config.load_config", return_value=copy.deepcopy(DEFAULT_CONFIG)), \
             patch("LRAEP_trainer.side_job.SideJob") as factory, \
             patch("LRAEP_trainer.progress.LiveProgress") as display, \
             patch("LRAEP_trainer.telemetry.GpuUtilization") as monitor:
            factory.return_value.step.side_effect = KeyboardInterrupt
            self.assertEqual(main([]), 130)
            factory.return_value.close.assert_called_once()
            display.return_value.__exit__.assert_called_once()
            monitor.return_value.__exit__.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GPUTrainingTests(unittest.TestCase):
    def test_repeated_cached_passes_train_with_exact_steps_and_release_input_thread(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("requires pyarrow")
        from PIL import Image
        from LRAEP_trainer.side_job_data import PrefetchedBatches

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = io.BytesIO()
            Image.new("RGB", (32, 32), "gray").save(payload, format="PNG")
            fixture = root / "fixture.parquet"
            parquet.write_table(arrow.table({"image_bytes": [payload.getvalue()] * 4,
                                              "class": [0, 1, 0, 1]}), fixture)
            config = copy.deepcopy(DEFAULT_CONFIG)
            config["side_job"].update(channels=[8], blocks_per_stage=[1], hidden_size=8,
                                        image_size=32, batch_size=3, passes_per_shard=3,
                                        prefetch_batches=2, epochs=1)
            with patch("LRAEP_trainer.side_job_data.DatasetSource") as source:
                source.return_value.list_shards.return_value = ["train/a.parquet"]
                source.return_value.download_shard.side_effect = (
                    lambda _name, destination, **_kw: shutil.copyfile(fixture, destination)
                )
                job = SideJob(config, project_root=root)
                try:
                    self.assertTrue(job.step())
                    iterator = job.iterator
                    self.assertIsInstance(iterator, PrefetchedBatches)
                    for _ in range(5):
                        if not job.step():
                            break
                    else:
                        self.fail("repeated epoch did not finish")
                    self.assertEqual(job.examples, 12)
                    self.assertEqual(job.epoch_batches, 4)
                    self.assertIn("Epoch 1/1 | Step 4/4 (100.0%)", job.status())
                    self.assertFalse(iterator._thread.is_alive())
                    self.assertFalse((job.directory / "train.parquet").exists())
                    source.return_value.download_shard.assert_called_once()
                finally:
                    job.close()

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
            parquet.write_table(arrow.table({"image_bytes": [stream.getvalue()] * 5,
                                             "class": ["flipped", "non_flipped"] * 2 + ["flipped"]}), fixture)
            config = copy.deepcopy(DEFAULT_CONFIG)
            config["side_job"].update(channels=[8, 16], blocks_per_stage=[1, 1],
                                       hidden_size=16, image_size=64, batch_size=2, epochs=2)
            source_paths, old_epoch_checks = [], []

            def download(shard, destination, **_):
                self.assertFalse(destination.exists(), "old shard must be removed before next download")
                source_paths.append(shard)
                if len(source_paths) > 1:
                    old_epoch_checks.append(torch.load(destination.parent / "weights.pth", weights_only=True)["epoch"])
                shutil.copyfile(fixture, destination)
                return destination

            with patch("LRAEP_trainer.side_job_data.DatasetSource") as source:
                source.return_value.list_shards.return_value = ["train/a.parquet", "train/b.parquet"]
                source.return_value.download_shard.side_effect = download
                clear_progress = Mock()
                job = SideJob(config, project_root=root, before_log=clear_progress)
                try:
                    self.assertTrue(job.step())
                    self.assertEqual(job.total_examples, 5)
                    self.assertEqual(job.total_batches, 3)
                    self.assertIn("Epoch 1/2 | Step 1/3 (40.0%)", job.status(37))
                    self.assertIn("loss=", job.status())
                    self.assertIn("accuracy=", job.status())
                    self.assertIn("GPU 37%", job.status(37))
                    initial = torch.load(job.directory / "weights.pth", weights_only=True)
                    checks = iter([False, False, False, True])
                    self.assertTrue(job.step(cancelled=lambda: next(checks)))
                    self.assertIsNotNone(job.pending_batch)
                    self.assertEqual(job.examples, 2)
                    self.assertIn("Epoch 1/2 | Step 1/3 (40.0%)", job.status())
                    for _ in range(20):
                        if not job.step():
                            break
                    else:
                        self.fail("job did not stop at configured epoch limit")
                    self.assertEqual(job.epoch, 2)
                    self.assertIn("Epoch 2/2 | Step 3/3 (100.0%)", job.status())
                    self.assertIn("Complete", job.status())
                    self.assertGreaterEqual(clear_progress.call_count, 1)
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
                resumed = SideJob(config, project_root=root)
                try:
                    self.assertFalse(resumed.step())
                    self.assertEqual(resumed.epoch, 2)
                    self.assertIn("Epoch 2/2 | Complete", resumed.status())
                    self.assertNotIn("loss=", resumed.status())
                    self.assertTrue(torch.equal(resumed.model.stem[0].weight.detach().cpu(), latest["model"]["stem.0.weight"]))
                finally:
                    resumed.close()

    def test_interrupted_epoch_is_discarded_and_independent_vpn_is_used(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["side_job"].update(channels=[8], blocks_per_stage=[1], hidden_size=8, image_size=32)
        config["vpn"].update(enabled=True, link="MAIN")
        config["side_job"]["vpn"].update(enabled=True, link="SIDE_JOB")
        with tempfile.TemporaryDirectory() as temporary, \
             patch("LRAEP_trainer.side_job_data.DatasetSource") as source, \
             patch("LRAEP_trainer.vpn.ManagedVPN") as vpn:
            source.return_value.list_shards.return_value = ["train/a.parquet"]
            job = SideJob(config, project_root=temporary)
            try:
                calls = iter([False, True])
                self.assertTrue(job.step(cancelled=lambda: next(calls)))
                vpn.assert_called_once_with(config["side_job"]["vpn"])
                source.assert_called_once_with(config["side_job"]["repo_url"], proxy=vpn.return_value.__enter__.return_value)
                source.return_value.download_shard.assert_not_called()
            finally:
                job.close()
            vpn.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
