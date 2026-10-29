"""Download one Hugging Face training shard and decode its embedded images.

The HTTP opener has an explicit route: ``None`` means direct, independent of
both the miner's active route and proxy environment variables. No dataset
library or shared Hugging Face cache is used.
"""

from __future__ import annotations

import http.client
import io
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from urllib.parse import quote, unquote, urljoin, urlsplit
import urllib.request

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import network


class DatasetError(RuntimeError):
    """A training shard could not be listed, downloaded, or decoded."""


class DatasetCancelled(DatasetError):
    """The caller stopped a shard download before it was committed."""


def _parts(value):
    parts = value.split("/")
    if (not parts or any(not part or part in {".", ".."} or "\\" in part
                         or any(ord(char) < 33 or ord(char) == 127 for char in part)
                         for part in parts)):
        raise DatasetError("dataset repository URL or shard path is invalid")
    return parts


def _huggingface_origin(parsed):
    try:
        return (
            parsed.scheme == "https" and parsed.hostname == "huggingface.co"
            and parsed.port in {None, 443}
            and parsed.username is None and parsed.password is None
        )
    except ValueError:
        return False


def _repository(repo_url):
    try:
        parsed = urlsplit(repo_url)
        if (not _huggingface_origin(parsed) or parsed.query or parsed.fragment):
            raise ValueError
        parts = _parts(unquote(parsed.path.strip("/")))
        if len(parts) < 3 or parts[0] != "datasets":
            raise ValueError
        repo = "/".join(parts[1:3])
        if len(parts) == 3:
            return repo, "main", "train"
        if len(parts) < 5 or parts[3] != "tree":
            raise ValueError
        return repo, parts[4], "/".join(parts[5:]) or "train"
    except (TypeError, ValueError, UnicodeError):
        raise DatasetError("decoy_job.repo_url must be a Hugging Face dataset repository or tree URL") from None


class DatasetSource:
    """List Parquet shards and download selected files without an SDK cache."""

    def __init__(self, repo_url, *, proxy=None, timeout=60):
        self.repo, self.revision, self.folder = _repository(repo_url)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise DatasetError("dataset download timeout must be positive and finite")
        self.timeout = timeout
        self.opener = network.http_opener(proxy)
        base = f"https://huggingface.co/api/datasets/{quote(self.repo, safe='/')}/tree/{quote(self.revision, safe='')}"
        self._listing_path = urlsplit(base).path + "/"
        self.listing_url = f"{base}/{quote(self.folder, safe='/')}?recursive=true&limit=1000"
        self._sizes = {}

    def _shard_path(self, path):
        if not isinstance(path, str):
            raise DatasetError("dataset listing contains an invalid shard path")
        _parts(path)
        if not path.startswith(self.folder + "/") or not path.endswith(".parquet"):
            raise DatasetError("dataset shard must be a Parquet file in the selected training folder")
        return path

    def list_shards(self):
        """Read every metadata page and return stable, sorted repository paths."""
        url = self.listing_url
        visited, shards = set(), set()
        while url:
            if url in visited:
                raise DatasetError("dataset listing contains a repeated pagination link")
            visited.add(url)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "LRAEP-decoy-job/0.1"})
                with self.opener.open(request, timeout=self.timeout) as response:
                    payload = response.read(8 * 1024 * 1024 + 1)
                    links = response.headers.get("Link", "")
                if len(payload) > 8 * 1024 * 1024:
                    raise DatasetError("dataset listing page exceeds the metadata limit")
                entries = json.loads(payload)
            except DatasetError:
                raise
            except (OSError, ValueError, UnicodeError, http.client.HTTPException):
                raise DatasetError("could not read the dataset shard listing") from None
            if not isinstance(entries, list):
                raise DatasetError("dataset listing must contain a list of files")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise DatasetError("dataset listing contains an invalid file entry")
                path = entry.get("path")
                if entry.get("type") != "file" or not isinstance(path, str) or not path.endswith(".parquet"):
                    continue
                path = self._shard_path(path)
                shards.add(path)
                size = entry.get("size")
                if type(size) is int and size >= 0:
                    self._sizes[path] = size
            match = re.search(r'<([^>]+)>\s*;\s*rel=[\"\']?next[\"\']?', links)
            url = urljoin(url, match.group(1)) if match else None
            if url:
                try:
                    parsed = urlsplit(url)
                except ValueError:
                    raise DatasetError("dataset pagination link is invalid") from None
                if (not _huggingface_origin(parsed)
                        or not parsed.path.startswith(self._listing_path) or parsed.fragment):
                    raise DatasetError("dataset pagination link leaves the selected repository")
        if not shards:
            raise DatasetError("the selected training folder contains no Parquet shards")
        return sorted(shards)

    def download_shard(self, shard, destination, *, cancelled=None):
        """Stream one file into an atomic destination; remove failed partials."""
        if cancelled is not None and cancelled():
            raise DatasetCancelled("dataset shard download was cancelled")
        shard = self._shard_path(shard)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://huggingface.co/datasets/{quote(self.repo, safe='/')}"
               f"/resolve/{quote(self.revision, safe='')}/{quote(shard, safe='/')}")
        temporary = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LRAEP-decoy-job/0.1"})
            with self.opener.open(request, timeout=self.timeout) as response:
                content_length = response.headers.get("Content-Length")
                expected = int(content_length) if content_length is not None else self._sizes.get(shard)
                size = 0
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".shard-", suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    read = getattr(response, "read1", response.read)
                    while True:
                        if cancelled is not None and cancelled():
                            raise DatasetCancelled("dataset shard download was cancelled")
                        chunk = read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        size += len(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                if expected is not None and size != expected:
                    raise DatasetError("dataset shard download is incomplete")
                if shard in self._sizes and size != self._sizes[shard]:
                    raise DatasetError("dataset shard size does not match its repository listing")
            if cancelled is not None and cancelled():
                raise DatasetCancelled("dataset shard download was cancelled")
            os.replace(temporary, destination)
            return destination
        except DatasetError:
            raise
        except (OSError, ValueError, UnicodeError, http.client.HTTPException):
            raise DatasetError("could not download the selected dataset shard") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _label(value):
    if type(value) is int and value in (0, 1):
        return value
    if isinstance(value, str) and value in {"non_flipped", "flipped"}:
        return 0 if value == "non_flipped" else 1
    raise DatasetError("dataset class must be non_flipped/flipped or integer 0/1")


def _image_tensor(value, image_size):
    from PIL import Image, ImageOps

    if isinstance(value, dict):
        # A Hugging Face Image feature can include a path alongside its bytes.
        # Always decode the embedded bytes; never load a path or external URL.
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
        raise DatasetError("dataset image_bytes must contain an embedded image")
    try:
        with Image.open(io.BytesIO(value)) as image:
            image = image.convert("RGB")
            image = ImageOps.pad(image, (image_size, image_size), method=Image.Resampling.BILINEAR, color="white")
            array = np.array(image, dtype=np.float32, copy=True)
        # Center and scale pixels without changing orientation labels.
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(127.5).sub_(1)
    except (OSError, ValueError, Image.DecompressionBombError):
        raise DatasetError("could not decode an embedded dataset image") from None


class ShardDataset(IterableDataset):
    """Decode bounded Parquet batches from exactly one local shard per epoch."""

    def __init__(self, path, *, image_size, seed=42):
        super().__init__()
        self.path = Path(path)
        if type(image_size) is not int or image_size < 1:
            raise DatasetError("dataset image_size must be a positive integer")
        self.image_size = image_size
        self.seed = seed

    def __iter__(self):
        try:
            import pyarrow.parquet as parquet
        except ImportError:
            raise DatasetError("the decoy job requires pyarrow; install the project requirements") from None
        worker = get_worker_info()
        rng = random.Random(self.seed)
        try:
            with parquet.ParquetFile(self.path) as shard:
                if not {"image_bytes", "class"}.issubset(shard.schema_arrow.names):
                    raise DatasetError("dataset shard must contain image_bytes and class columns")
                for index, batch in enumerate(shard.iter_batches(batch_size=32, columns=["image_bytes", "class"])):
                    if worker is not None and index % worker.num_workers != worker.id:
                        continue
                    rows = batch.to_pylist()
                    rng.shuffle(rows)
                    for row in rows:
                        yield _image_tensor(row["image_bytes"], self.image_size), _label(row["class"])
        except DatasetError:
            raise
        except (OSError, ValueError, TypeError):
            raise DatasetError("could not read the selected Parquet training shard") from None
