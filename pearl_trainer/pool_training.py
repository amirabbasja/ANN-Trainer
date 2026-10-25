"""Our GPU training loop connected directly to a supported PEARL pool."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import logging
import signal
import threading
import time

import numpy as np
import torch

from .gpu import TrainingCancelled
from .progress import LiveProgress
from .protocol import (TrainingConfig, load_official_helpers, pool_share_target,
                       prepare_batch, search_batch, verified_pool_share)
from .telemetry import GpuUtilization
from . import native_cuda

LOG = logging.getLogger("pearl_trainer")


@dataclass
class PoolTrainingStats:
    batches: int = 0
    attempts: int = 0
    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    stale: int = 0


def pool_status(stats: PoolTrainingStats, cfg: TrainingConfig, elapsed: float,
                gpu_utilization: int | None) -> str:
    """Report local average PEARL work, counting one int8 MAC per hash unit.

    At rank 128 each hash tile contributes tile_h * tile_w * k units to
    the target scaling. For our fully tiled matrices this is m * n * k
    per completed batch, rather than the number of final BLAKE3 tickets.
    """
    rate = stats.batches * cfg.m * cfg.n * cfg.k / max(elapsed, 0.001)
    units = ("H/s", "kH/s", "MH/s", "GH/s", "TH/s", "PH/s", "EH/s")
    unit = 0
    while rate >= 1000 and unit < len(units) - 1:
        rate /= 1000
        unit += 1
    utilization = "N/A" if gpu_utilization is None else f"{gpu_utilization}%"
    return (f"Hashrate(avg) {rate:.2f} {units[unit]} | GPU {utilization} | "
            f"accepted={stats.accepted} rejected={stats.rejected} | "
            f"submitted={stats.submitted} batches={stats.batches} "
            f"attempts={stats.attempts} stale={stats.stale}")


def select_pool_protocol(settings: dict) -> str:
    """Select only an explicitly supported dialect; never guess a custom one."""
    selected = settings.get("protocol", "auto")
    host = settings["host"].lower().rstrip(".")
    if selected == "auto":
        if host == "prl.kryptex.network" or (
            host.startswith("prl-") and host.endswith(".kryptex.network")
        ):
            selected = "kryptex"
        elif host == "pool.pearlhash.xyz" or host.endswith(".pearlhash.net"):
            selected = "pearlhash"
        else:
            raise ValueError("unknown pool dialect; set pool.protocol for a documented compatible endpoint")
    if selected == "pearlhash":
        raise RuntimeError(
            "Pearlhash uses a separate, undocumented protocol. Its job/share adapter "
            "is not implemented; pool-check can test transport only. Kryptex is supported."
        )
    if selected != "kryptex":
        raise ValueError("pool.protocol must select auto, kryptex, or pearlhash")
    return selected


def validate_pool_identity(settings: dict) -> None:
    """Check configuration before loading helpers or opening a connection."""
    wallet = settings.get("wallet", "")
    worker = settings.get("worker", "")
    if not isinstance(wallet, str) or not wallet.strip():
        raise ValueError("fill pool.wallet in your configuration with your payout address or Kryptex account")
    if any(character.isspace() for character in wallet + worker):
        raise ValueError("pool.wallet and pool.worker must not contain whitespace")
    if wallet.startswith("solo:"):
        raise ValueError("remove the solo: prefix to use the requested pool training mode")
    if wallet.startswith("prl1"):
        from pearl_gateway.blockchain_utils.blockchain_utils import get_script_pubkey_from_p2tr_address

        get_script_pubkey_from_p2tr_address(wallet)


@contextmanager
def _signal_stop(stopped, time_limit=0):
    previous = {}
    timer = None
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stopped.set())
    if time_limit > 0:
        timer = threading.Timer(time_limit, stopped.set)
        timer.daemon = True
        timer.start()
    try:
        yield
    finally:
        stopped.set()
        if timer is not None:
            timer.cancel()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_pool_training(configuration: dict, *, login_only: bool = False) -> int:
    """Authenticate, process live jobs, and submit verified native PEARL proofs.

    ``login_only`` stops after the first authenticated job and does no GPU work.
    Continuous sessions reconnect after transport failures, invalidating every
    job and pending submission from the disconnected session.
    """
    from .pool_client import (PoolAuthenticationError, PoolCompatibilityError,
                              PoolSession, StalePoolJob)
    from .pool_transport import PoolConnectionError

    pool = configuration["pool"]
    select_pool_protocol(pool)
    # Empty configuration has a useful error before any helper initialization.
    if not pool.get("wallet", "").strip():
        raise ValueError("fill pool.wallet in your configuration with your payout address or Kryptex account")
    load_official_helpers(configuration["official_source"])
    validate_pool_identity(pool)
    import pearl_mining as pm

    settings = configuration["training"]
    cfg = TrainingConfig(settings["m"], settings["n"], settings["k"],
                         settings["tile_h"], settings["tile_w"])
    if not login_only:
        device = torch.device(configuration["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("an available CUDA GPU is required for pool training")
        # Capture the caller's current device before the telemetry thread
        # starts; CUDA's implicit current-device selection is thread-local.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        LOG.info("GPU: %s", torch.cuda.get_device_name(device))
    use_native = (not login_only and (cfg.tile_h, cfg.tile_w) == (8, 8)
                  and native_cuda.available(configuration["device"]))
    if not login_only:
        LOG.info("Mining backend: %s", "native CUDA DP4A" if use_native else "PyTorch reference")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    stopped = threading.Event()
    stats = PoolTrainingStats()
    rng = np.random.default_rng()
    started = time.monotonic()
    limit = settings["max_seconds"]
    max_batches = settings["max_batches"]

    def expired():
        return stopped.is_set() or (limit > 0 and time.monotonic() - started >= limit)

    def finished():
        return expired() or (max_batches > 0 and stats.batches >= max_batches)

    with _signal_stop(stopped, limit), ExitStack() as resources:
        progress = resources.enter_context(LiveProgress())
        monitor = None
        if not login_only:
            monitor = GpuUtilization(str(device)).start()
            resources.callback(monitor.close)

        def update_progress(force=False):
            if monitor is not None:
                progress.update(pool_status(stats, cfg, time.monotonic() - started,
                                            monitor.read()), force=force)

        def log(level, message, *args):
            progress.clear()
            LOG.log(level, message, *args)

        update_progress(force=True)
        while not finished():
            try:
                session_settings = dict(pool)
                if limit > 0:
                    remaining = max(0.001, limit - (time.monotonic() - started))
                    session_settings["connect_timeout"] = min(pool["connect_timeout"], remaining)
                with PoolSession(session_settings, stop_event=stopped) as session:
                    log(logging.INFO, "Pool authenticated at %s:%d; proof compression=%s",
                        pool["host"], pool["port"],
                        "gzip" if session.negotiated_gzip else "none")
                    previous_generation = None
                    job_deadline = time.monotonic() + pool["read_timeout"]
                    while not finished():
                        update_progress()
                        snapshot = session.wait_job(timeout=0.25)
                        if snapshot is None:
                            if time.monotonic() >= job_deadline:
                                raise PoolConnectionError("pool sent no usable training job before the job timeout")
                            continue
                        job = snapshot.job
                        if snapshot.generation != previous_generation:
                            log(logging.INFO, "Pool job=%s height=%d certificate=V%d",
                                job.job_id, job.height, job.cert_version)
                            previous_generation = snapshot.generation
                            job_deadline = time.monotonic() + pool["read_timeout"]
                        header = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
                        target = pool_share_target(job.target, cfg, "base")
                        if login_only:
                            log(logging.INFO, "Authenticated job verified: header=%d bytes, target=%d bits; no GPU work or shares submitted",
                                len(job.header_bytes), target.bit_length())
                            return 0

                        batch = prepare_batch(header, job.cert_version, cfg, rng,
                                              device=configuration["device"], gpu_signals=True,
                                              native=use_native)
                        if expired() or not session.is_current(snapshot):
                            continue
                        cancelled = lambda: expired() or not session.is_current(snapshot)
                        try:
                            matches = search_batch(batch, target, configuration["device"], cancelled)
                        except TrainingCancelled:
                            continue
                        stats.batches += 1
                        stats.attempts += (cfg.m // cfg.tile_h) * (cfg.n // cfg.tile_w)
                        # Limits stop the next batch; completed work remains
                        # eligible for submission, as in the solo training loop.
                        for row, col, digest in matches:
                            if expired() or not session.is_current(snapshot):
                                stats.stale += 1
                                break
                            proof, _ = verified_pool_share(
                                batch, header, job.cert_version, row, col, digest, target,
                            )
                            if expired() or not session.is_current(snapshot):
                                stats.stale += 1
                                break
                            try:
                                result = session.submit(snapshot, proof.to_base64())
                            except StalePoolJob:
                                stats.stale += 1
                                break
                            stats.submitted += 1
                            if result.accepted:
                                stats.accepted += 1
                                log(logging.INFO, "Pool accepted training share job=%s", job.job_id)
                            else:
                                stats.rejected += 1
                                log(logging.WARNING, "Pool rejected training share job=%s: %s", job.job_id, result.error)
                        update_progress()
            except (PoolAuthenticationError, PoolCompatibilityError):
                raise
            except PoolConnectionError as exc:
                if stopped.is_set():
                    break
                if login_only:
                    raise
                if finished():
                    break
                log(logging.WARNING, "%s; reconnecting in %.1f seconds", exc, pool["reconnect_delay"])
                delay = pool["reconnect_delay"]
                if limit > 0:
                    delay = min(delay, max(0, limit - (time.monotonic() - started)))
                stopped.wait(delay)
        update_progress(force=True)
    LOG.info("Pool training stopped: batches=%d submitted=%d accepted=%d rejected=%d stale=%d",
             stats.batches, stats.submitted, stats.accepted, stats.rejected, stats.stale)
    if login_only:
        raise RuntimeError("no authenticated pool job was received before stopping")
    return 0
