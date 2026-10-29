"""Decoy downloads have independent routing and keep only local embedded data."""

import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch

import torch
from PIL import Image

from LRAEP_trainer import network
from LRAEP_trainer.decoy_data import DatasetCancelled, DatasetError, DatasetSource, ShardDataset


REPOSITORY = "https://huggingface.co/datasets/BDRC/tibetan-page-orientation-classifier-dataset"


def response(body, **headers):
    stream = io.BytesIO(body)
    stream.headers = headers
    return stream


def listing(entries, **headers):
    return response(json.dumps(entries).encode(), **headers)


def image_bytes(size=(30, 10), color="black"):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


class DatasetSourceTests(unittest.TestCase):
    def test_repository_and_tree_urls_select_train_folder(self):
        for suffix in ("", "/", "/tree/main", "/tree/main/train"):
            with self.subTest(suffix=suffix):
                source = DatasetSource(REPOSITORY + suffix)
                self.assertEqual(source.repo, "BDRC/tibetan-page-orientation-classifier-dataset")
                self.assertEqual(source.revision, "main")
                self.assertEqual(source.folder, "train")

    def test_hostname_case_and_explicit_https_port_are_accepted(self):
        for host in ("HuggingFace.co", "huggingface.co:443", "HUGGINGFACE.CO:443"):
            with self.subTest(host=host):
                source = DatasetSource(f"https://{host}/datasets/BDRC/dataset/tree/main/train")
                self.assertEqual(source.repo, "BDRC/dataset")
                self.assertEqual(source.folder, "train")

    def test_invalid_repository_urls_are_rejected_without_echoing_values(self):
        for value in (None, "SECRET", "http://huggingface.co/datasets/org/repo",
                      "https://SECRET@huggingface.co/datasets/org/repo",
                      "https://huggingface.co.evil/datasets/org/repo",
                      "https://huggingface.co:444/datasets/org/repo",
                      "https://SECRET:password@huggingface.co:443/datasets/org/repo",
                      REPOSITORY + "?token=SECRET", REPOSITORY + "#SECRET",
                      REPOSITORY + "/resolve/main/train", REPOSITORY + "/tree/main/../SECRET",
                      REPOSITORY + "/tree/main/train/%2e%2e/SECRET"):
            with self.subTest(value=value), self.assertRaises(DatasetError) as caught:
                DatasetSource(value)
            self.assertNotIn("SECRET", str(caught.exception))

    def test_paginated_listing_sorts_and_deduplicates_parquet_files(self):
        source = DatasetSource(REPOSITORY + "/tree/main/train")
        next_url = source.listing_url + "&cursor=next"
        source.opener = Mock()
        source.opener.open.side_effect = [
            listing([{"type": "file", "path": "train/b.parquet", "size": 10},
                     {"type": "directory", "path": "train/sub"},
                     {"type": "file", "path": "train/readme.txt"}],
                    Link=f'<{next_url}>; rel="next"'),
            listing([{"type": "file", "path": "train/a.parquet"},
                     {"type": "file", "path": "train/b.parquet", "size": 10}]),
        ]
        self.assertEqual(source.list_shards(), ["train/a.parquet", "train/b.parquet"])
        self.assertEqual(source.opener.open.call_args_list[1].args[0].full_url, next_url)

    def test_pagination_allows_equivalent_https_origin(self):
        source = DatasetSource(REPOSITORY)
        source.opener = Mock()
        next_url = source.listing_url.replace("huggingface.co", "HUGGINGFACE.CO:443") + "&cursor=next"
        source.opener.open.side_effect = [
            listing([], Link=f'<{next_url}>; rel="next"'),
            listing([{"type": "file", "path": "train/a.parquet"}]),
        ]
        self.assertEqual(source.list_shards(), ["train/a.parquet"])

    def test_listing_rejects_pagination_outside_repository(self):
        source = DatasetSource(REPOSITORY)
        source.opener = Mock()
        source.opener.open.return_value = listing([], Link='<https://example.org/SECRET>; rel="next"')
        with self.assertRaisesRegex(DatasetError, "pagination") as caught:
            source.list_shards()
        self.assertNotIn("SECRET", str(caught.exception))
        source.opener.open.assert_called_once()

    def test_listing_rejects_invalid_paths_and_empty_files(self):
        for entry in ([], [{"type": "file", "path": "train/../escape.parquet"}],
                      [{"type": "file", "path": "validation/other.parquet"}]):
            source = DatasetSource(REPOSITORY)
            source.opener = Mock()
            source.opener.open.return_value = listing(entry)
            with self.subTest(entry=entry), self.assertRaises(DatasetError):
                source.list_shards()

    def test_direct_download_route_ignores_main_vpn_and_environment(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"direct")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            proxy = network.SocksProxy("127.0.0.1", 1)
            proxy_env = {"http_proxy": "http://must-not-connect.invalid:9999",
                         "https_proxy": "http://must-not-connect.invalid:9999",
                         "ALL_PROXY": "socks5://must-not-connect.invalid:9999", "NO_PROXY": ""}
            with network.routed(proxy), patch.dict(os.environ, proxy_env), \
                    patch.object(network, "create_connection", side_effect=AssertionError("main VPN must be bypassed")):
                source = DatasetSource(REPOSITORY)
                with source.opener.open(f"http://127.0.0.1:{server.server_port}/", timeout=3) as result:
                    self.assertEqual(result.read(), b"direct")
                self.assertIs(network.current_proxy(), proxy)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)

    def test_explicit_decoy_proxy_passed_even_under_main_vpn(self):
        main_proxy = network.SocksProxy("127.0.0.1", 1080)
        decoy_proxy = network.SocksProxy("127.0.0.1", 1081)
        with network.routed(main_proxy), patch.object(network, "http_opener") as opener:
            DatasetSource(REPOSITORY, proxy=decoy_proxy)
            opener.assert_called_once_with(decoy_proxy)
            self.assertIs(network.current_proxy(), main_proxy)

    def test_atomic_download_is_in_requested_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DatasetSource(REPOSITORY)
            source.opener = Mock()
            source.opener.open.return_value = response(b"shard bytes", **{"Content-Length": "11"})
            path = Path(directory) / "data" / "shard.parquet"
            self.assertEqual(source.download_shard("train/train-00000.parquet", path), path)
            self.assertEqual(path.read_bytes(), b"shard bytes")
            self.assertEqual(list(path.parent.iterdir()), [path])
            self.assertIn("/resolve/main/train/train-00000.parquet", source.opener.open.call_args.args[0].full_url)

    def test_partial_download_does_not_replace_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.parquet"
            path.write_bytes(b"old")
            source = DatasetSource(REPOSITORY)
            source.opener = Mock()
            source.opener.open.return_value = response(b"partial", **{"Content-Length": "100"})
            with self.assertRaisesRegex(DatasetError, "incomplete"):
                source.download_shard("train/a.parquet", path)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_download_size_checked_against_listing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DatasetSource(REPOSITORY)
            source._sizes["train/a.parquet"] = 10
            source.opener = Mock()
            source.opener.open.return_value = response(b"bad")
            path = Path(directory) / "shard.parquet"
            with self.assertRaises(DatasetError):
                source.download_shard("train/a.parquet", path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cancelled_download_cleans_partial_and_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.parquet"
            path.write_bytes(b"old")
            source = DatasetSource(REPOSITORY)
            source.opener = Mock()
            source.opener.open.return_value = response(b"new data")
            cancelled = Mock(side_effect=[False, False, True])
            with self.assertRaises(DatasetCancelled):
                source.download_shard("train/a.parquet", path, cancelled=cancelled)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_pre_cancelled_download_does_not_open_network(self):
        source = DatasetSource(REPOSITORY)
        source.opener = Mock()
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(DatasetCancelled):
            source.download_shard("train/a.parquet", Path(directory) / "shard.parquet", cancelled=lambda: True)
        source.opener.open.assert_not_called()

    def test_download_connection_failure_is_sanitized(self):
        source = DatasetSource(REPOSITORY)
        source.opener = Mock()
        source.opener.open.side_effect = OSError("SECRET-SIGNED-CDN-URL")
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(DatasetError) as caught:
            source.download_shard("train/a.parquet", Path(directory) / "shard.parquet")
        self.assertNotIn("SECRET", str(caught.exception))


class ShardDatasetTests(unittest.TestCase):
    def setUp(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is required for local Parquet fixtures")
        self.arrow, self.parquet = arrow, parquet
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "shard.parquet"

    def write(self, rows):
        self.parquet.write_table(self.arrow.Table.from_pylist(rows), self.path)

    def test_binary_images_and_string_labels_decode_without_orientation_change(self):
        self.write([{"image_bytes": image_bytes(), "class": "non_flipped"},
                    {"image_bytes": image_bytes(color="white"), "class": "flipped"}])
        rows = sorted(ShardDataset(self.path, image_size=24), key=lambda row: row[1])
        self.assertEqual([row[1] for row in rows], [0, 1])
        self.assertEqual(tuple(rows[0][0].shape), (3, 24, 24))
        self.assertEqual(rows[0][0].dtype, torch.float32)
        self.assertEqual(rows[0][0][:, 12, 12].tolist(), [-1.0, -1.0, -1.0])
        self.assertEqual(rows[0][0][:, 0, 0].tolist(), [1.0, 1.0, 1.0])
        self.assertTrue(torch.all(rows[1][0] == 1))

    def test_huggingface_embedded_images_and_integer_labels(self):
        self.write([{"image_bytes": {"bytes": image_bytes(), "path": "https://never-fetch.invalid/a"}, "class": 1}])
        with patch("urllib.request.urlopen", side_effect=AssertionError("images must be local")):
            rows = list(ShardDataset(self.path, image_size=24))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], 1)

    def test_path_only_images_are_rejected(self):
        self.write([{"image_bytes": {"bytes": None, "path": "SECRET-PATH"}, "class": 0}])
        with self.assertRaisesRegex(DatasetError, "embedded") as caught:
            list(ShardDataset(self.path, image_size=24))
        self.assertNotIn("SECRET", str(caught.exception))

    def test_unknown_label_is_rejected(self):
        self.write([{"image_bytes": image_bytes(), "class": "SECRET-LABEL"}])
        with self.assertRaisesRegex(DatasetError, "class") as caught:
            list(ShardDataset(self.path, image_size=24))
        self.assertNotIn("SECRET", str(caught.exception))

    def test_missing_columns_fail_with_actionable_message(self):
        self.write([{"image": image_bytes(), "label": 0}])
        with self.assertRaisesRegex(DatasetError, "image_bytes and class"):
            list(ShardDataset(self.path, image_size=24))

    def test_repeated_iteration_is_reproducible_and_covers_all_records(self):
        self.write([{"image_bytes": image_bytes(), "class": index % 2} for index in range(77)])
        dataset = ShardDataset(self.path, image_size=8, seed=123)
        first = [row[1] for row in dataset]
        self.assertEqual(len(first), 77)
        self.assertEqual(first, [row[1] for row in dataset])


if __name__ == "__main__":
    unittest.main()
