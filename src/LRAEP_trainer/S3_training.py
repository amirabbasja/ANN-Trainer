"""Our GPU training loop connected directly to a supported S6 S3."""

from .fixed_constants import FixedAlias, constant_text
from .wire_constants import WR_FIRST_REPORT_DELAY, WR_REPORT_INTERVAL, has_identity_mode_prefix

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import gc
import logging
from pathlib import Path
import signal
import threading
import time

import numpy as np
import torch

from .training_artifacts import training_output

from .cuda_devices import resolve_cuda_devices
from .gpu import TrainingCancelled as TrainingCancelled
from .gpu_pacing import GpuPacer
from .mining_devices import MiningDevice, batch_executor, run_batches
from .effort import JobSequence, MINING_JOBS, ScheduledSideJob
from .progress import LiveProgress, format_elapsed, in_notebook, main_status
from .protocol import (TrainingConfig, load_official_helpers, S3_training_threshold,
                       prepare_batch, search_batch, verified_S3_submission)
from .telemetry import GpuUtilization, busiest_gpu
from .telegram import TelegramUpdates
from .job_control import JobEvents, VPNJobGuard
from .progress import job_status
from . import native_cuda
from .word_codec import word_matches

LOG = logging.getLogger("LRAEP_trainer")


@dataclass
class S3TrainingStats:
    batches: int = 0
    attempts: int = 0
    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    stale: int = 0


def S3_training_rate(stats: S3TrainingStats, cfg: TrainingConfig, elapsed: float) -> float:
    """Return completed S6 matrix work per second in raw S3 H/s units.

    At rank 128 each hash tile contributes tile_h * tile_w * k units to
    the target scaling. For our fully tiled matrices this is m * n * k
    per completed batch, rather than the number of final BLAKE3 tickets.
    """
    return stats.batches * cfg.m * cfg.n * cfg.k / max(elapsed, 0.001)


def format_training_rate(rate: float) -> str:
    """Use identical display units for local work and the rate sent to WR."""
    units = ("H/s", "kH/s", "MH/s", "GH/s", "TH/s", "PH/s", "EH/s")
    unit = 0
    while rate >= 1000 and unit < len(units) - 1:
        rate /= 1000
        unit += 1
    return f"{rate:.2f} {units[unit]}"


def S3_status(stats: S3TrainingStats, cfg: TrainingConfig, elapsed: float,
                gpu_utilization: int | None, *, max_batches: int = 0) -> str:
    """Show completed compute steps using the CNN job's progress layout."""
    return main_status(stats.batches, elapsed, gpu_utilization, max_steps=max_batches)


def select_S3_protocol(settings: dict) -> str:
    """Select only an explicitly supported dialect; never guess a custom one."""
    selected = settings.get("protocol", "KEX")
    if selected == "auto":
        raise ValueError("S3.protocol must explicitly select KEX or WR; automatic detection is unavailable")
    if selected == "S7" or word_matches("legacy_protocol", selected):
        selected = "WR"
    if selected not in ("KEX", "WR"):
        raise ValueError("S3.protocol must select KEX or WR")
    return selected


def validate_S3_endpoint(settings: dict) -> None:
    """Require the user's endpoint before helper setup or any network activity."""
    host = settings.get("host", "")
    if not isinstance(host, str) or not host.strip():
        selected = settings.get("protocol")
        profile = "WR" if selected in ("WR", "S7") or word_matches("legacy_protocol", selected) else "KEX"
        suffix = constant_text(FixedAlias('training.validation.coordinator_hostname_suffix'))
        raise ValueError(f"fill S3.host for a flat configuration, or S3.{profile}{suffix}")


def validate_S3_identity(settings: dict, selected_protocol: str | None = None, *,
                           validate_address: bool = True) -> None:
    """Check configuration before loading helpers or opening a connection."""
    selected = select_S3_protocol(settings if selected_protocol is None else {"protocol": selected_protocol})
    identity_field = "WR_S2" if selected == "WR" else "KEX_uname"
    identity = settings.get(identity_field, "")
    worker = settings.get(constant_text(FixedAlias('training.field.training_worker')), "")
    if not isinstance(identity, str) or not identity.strip():
        description = "WR payout address" if selected == "WR" else "KEX username or payout address"
        raise ValueError(f"fill S3.{identity_field} in your configuration with your {description}")
    if not isinstance(worker, str):
        raise ValueError(constant_text(FixedAlias('training.validation.worker_identity_type')))
    if any(character.isspace() for character in identity + worker):
        suffix = constant_text(FixedAlias('training.validation.worker_identity_whitespace_suffix'))
        raise ValueError(f"S3.{identity_field}{suffix}")
    if has_identity_mode_prefix(identity):
        raise ValueError("remove the unsupported identity mode prefix to use S3 training")
    if validate_address and identity.startswith(constant_text('identity_prefix')):
        from LRAEP_gateway.blockchain_utils.blockchain_utils import get_script_pubkey_from_p2tr_address

        try:
            get_script_pubkey_from_p2tr_address(identity)
        except ValueError:
            suffix = constant_text(FixedAlias('training.validation.account_identity_required_suffix'))
            raise ValueError(f"S3.{identity_field}{suffix}") from None


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


def run_S3_training(configuration: dict, *, login_only: bool = False) -> int:
    """Run a repeating job sequence, closing every resource on stop or exit."""
    settings = configuration["training"]
    generate_artifacts = login_only or settings.get("generate_artifacts", True)
    stopped = threading.Event()
    started = time.monotonic()
    effort = None if login_only else JobSequence(settings["job_sequence"], started=started)
    deadline = started + settings["max_seconds"] if settings["max_seconds"] else None
    with training_output(generate_artifacts), _signal_stop(stopped, settings["max_seconds"]), ExitStack() as resources:
        if effort is not None:
            resources.enter_context(effort.stopping(stopped))
            if not effort.wait(stopped, deadline=deadline):
                return 0
        return _run_S3_training(configuration, login_only=login_only, stopped=stopped,
                                  effort=effort, started=started)


def _release_work(devices):
    gc.collect()
    if torch.cuda.is_initialized():
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()


def _run_S3_training(configuration, *, login_only, stopped, effort, started):
    """Authenticate, process live jobs, and submit verified S3-specific work.

    ``login_only`` stops after the first authenticated job and does no GPU work.
    Continuous sessions reconnect after transport failures, invalidating every
    job and pending submission from the disconnected session.
    """
    from .config import ConfigError, resolve_S3_settings
    from .S3_client import (S3AuthenticationError, S3CompatibilityError,
                              S3Session, StaleS3Job)
    from .S3_transport import S3ConnectionError
    from .wire_constants import WR_CAPTURED_SHAPE

    settings = configuration["training"]
    generate_artifacts = login_only or settings.get("generate_artifacts", True)
    S3 = resolve_S3_settings(configuration["S3"])
    selected_protocol = select_S3_protocol(S3)
    validate_S3_endpoint(S3)
    validate_S3_identity(S3, selected_protocol, validate_address=False)
    session_class = S3Session
    session_options = {}
    shape = tuple(settings[field] for field in ("m", "n", "k", "tile_h", "tile_w"))
    captured_shape = selected_protocol == "WR" and shape == WR_CAPTURED_SHAPE
    config_class = TrainingConfig
    search = search_batch
    if selected_protocol == "WR":
        from .S3_diagnostics import UnknownRequestRecorder
        from .wr_client import WRSession
        from .wr_protocol import SHAPE_REJECTED
        from .wr_work import (
            CapturedBatchPreparation, CapturedTrainingConfig, compute_captured_batch,
            prepare_wr_batch, verified_wr_submission,
        )

        if captured_shape:
            config_class = CapturedTrainingConfig
            search = compute_captured_batch
        session_class = WRSession
        if generate_artifacts:
            diagnostic_dir = S3.get("unknown_request_dir") or (
                Path(settings["output_dir"]) / "unknown-S3-requests"
            )
            session_options["unknown_handler"] = UnknownRequestRecorder(diagnostic_dir, S3)
    load_official_helpers(configuration["official_source"])
    validate_S3_identity(S3, selected_protocol)
    from LRAEP_trainer import primitives as pm

    if selected_protocol != "WR" and shape == WR_CAPTURED_SHAPE:
        raise ConfigError("The captured S7 training shape requires the WR protocol")
    cfg = config_class(*shape)
    devices = ()
    if not login_only:
        devices = resolve_cuda_devices(configuration["device"], unavailable="an available CUDA GPU is required for S3 training")
    gpu_names = [torch.cuda.get_device_name(device) for device in devices]
    workers = [MiningDevice(device, GpuPacer(device, 100),
                            (captured_shape or (cfg.tile_h, cfg.tile_w) == (8, 8)) and native_cuda.available(device))
               for device in devices]
    for worker, name in zip(workers, gpu_names):
        if captured_shape and not worker.native:
            raise ConfigError("The configured S7 shape requires native CUDA kernels")
        LOG.debug("GPU %s (%s): %s", worker.device, name, "native CUDA DP4A" if worker.native else "PyTorch reference")
    preparations = ({worker.device: CapturedBatchPreparation(cfg, worker.device)
                     for worker in workers if worker.native}
                    if selected_protocol == "WR" else {})
    torch.set_num_threads(min(4, torch.get_num_threads()))
    stats = S3TrainingStats()
    limit = settings["max_seconds"]
    max_batches = settings["max_batches"]
    side_pacer = None
    if workers:
        side_pacer = (GpuPacer(devices, 100) if len(devices) > 1 or effort.has_mixed_job
                      else workers[0].pacer)
    deadline = started + limit if limit else None

    def expired():
        return (stopped.is_set() or (limit > 0 and time.monotonic() - started >= limit)
                or (effort is not None and effort.ended()))

    def finished():
        return expired() or (max_batches > 0 and stats.batches >= max_batches)

    with ExitStack() as resources:
        executor = batch_executor(resources, devices)
        show_progress = generate_artifacts or in_notebook()
        progress = resources.enter_context(LiveProgress(
            enabled=show_progress, trainer_name=configuration.get("trainer_name", "trainer"),
        ))
        telegram_enabled = configuration.get("telegram", {}).get("enabled", False)
        monitors = []
        for device in devices if show_progress or telegram_enabled else ():
            monitor = GpuUtilization(str(device)).start()
            resources.callback(monitor.close)
            monitors.append(monitor)

        active_job = "job-0"

        def show_side_job(job):
            update_progress()

        side_job = (ScheduledSideJob(configuration, before_log=progress.clear, on_progress=show_side_job)
                 if not login_only else None)

        def telegram_snapshot():
            now = time.monotonic()
            number, kind, percentage, boundary = effort.position(now)
            remaining = (0 if percentage == -1 else
                         max(0, (boundary if deadline is None else min(boundary, deadline)) - now))
            cnn_epoch = 0
            if kind in ("job-1", "job-2"):
                job = side_job.job if side_job is not None else None
                cnn_epoch = getattr(job, "_progress_epoch", 0)
            if kind == "job-1":
                speed = cnn_epoch
            else:
                speed = (0 if percentage <= 0 or (vpn_guard is not None and vpn_guard.blocked())
                         else S3_training_rate(stats, cfg, now - started))
            status = job_status(number, kind,
                              busiest_gpu(monitors), remaining, speed, cnn_epoch=cnn_epoch)
            if vpn_guard is not None and vpn_guard.enabled:
                status += f" | VPN: {vpn_guard.status()}"
            return status

        updates = None
        events = None
        vpn_guard = None
        if not login_only:
            updates = resources.enter_context(TelegramUpdates(configuration, telegram_snapshot))
            events = JobEvents(effort, updates)
            resources.callback(events.finish)
            vpn_guard = resources.enter_context(VPNJobGuard(effort, configuration, updates))

        def update_progress(force=False):
            nonlocal active_job
            if events is not None:
                events.update()
            if show_progress and monitors:
                kind = effort.current_job()[0]
                force = force or active_job != kind
                active_job = kind
                progress.update(telegram_snapshot(), force=force)

        def log(level, message, *args):
            LOG.debug(message, *args)

        def report_gpu(session, sample_time):
            elapsed = sample_time - started
            rate = S3_training_rate(stats, cfg, elapsed)
            if len(workers) == 1:
                request_id = session.report_info(gpu_names[0], rate)
            else:
                scale = cfg.m * cfg.n * cfg.k / max(elapsed, 0.001)
                request_id = session.report_devices([(name, worker.batches * scale)
                                                     for worker, name in zip(workers, gpu_names)])
            log(logging.DEBUG, constant_text(FixedAlias('training.log.accelerator_metrics_sent')),
                format_training_rate(rate), request_id)

        update_progress(force=True)
        if not login_only:
            progress.start_refresh(telegram_snapshot)
        if side_job is not None:
            resources.callback(side_job.close)
            if effort.has_mixed_job:
                side_job.start_concurrent(effort, stopped, devices, side_pacer, deadline=deadline)
        while not finished():
            if effort is not None and not effort.wait(stopped, deadline=deadline):
                break
            if side_job is not None and side_job.run_if_active(effort, stopped, side_pacer, deadline=deadline):
                continue
            if vpn_guard is not None and not vpn_guard.resume():
                vpn_guard.wait(stopped, deadline=deadline, on_progress=update_progress)
                continue
            session = None
            retry_delay = None
            wait_for_vpn = False
            session_boundary = None
            if effort is not None:
                kind, percentage, session_boundary = effort.current_job()
                if kind not in MINING_JOBS or percentage <= 0:
                    continue
            try:
                session_settings = dict(S3)
                connect_deadline = deadline
                if effort is not None and effort.exit_deadline is not None:
                    connect_deadline = (effort.exit_deadline if connect_deadline is None else
                                        min(connect_deadline, effort.exit_deadline))
                if effort is not None:
                    connect_deadline = (session_boundary if connect_deadline is None else
                                        min(session_boundary, connect_deadline))
                if connect_deadline is not None:
                    remaining = max(0.001, connect_deadline - time.monotonic())
                    session_settings["connect_timeout"] = min(S3["connect_timeout"], remaining)
                if vpn_guard is not None and vpn_guard.enabled:
                    session_settings["connect_timeout"] = min(session_settings.get("connect_timeout", 15), 5.0)
                session = session_class(session_settings, stop_event=stopped, **session_options)
                if vpn_guard is not None:
                    vpn_guard.attach(session)
                with session:
                    log(logging.INFO, constant_text(FixedAlias('training.log.coordinator_session_authenticated')),
                        S3["host"], S3["port"],
                        "gzip" if session.negotiated_gzip else "none")
                    next_report = time.monotonic() + WR_FIRST_REPORT_DELAY
                    previous_generation = None
                    previous_effort = None
                    job_deadline = time.monotonic() + S3["read_timeout"]
                    while not finished():
                        if effort is not None and effort.current_job()[2] != session_boundary:
                            break
                        if vpn_guard is not None and vpn_guard.check():
                            break
                        if effort is not None:
                            kind, utilization, boundary = effort.current_job()
                            if boundary != session_boundary or kind not in MINING_JOBS or utilization <= 0:
                                break
                        update_progress()
                        wait_after_batch = False
                        try:
                            snapshot = session.wait_job(timeout=0.25)
                            if snapshot is None:
                                if time.monotonic() >= job_deadline:
                                    raise S3ConnectionError("S3 sent no usable training job before the job timeout")
                                continue
                            job = snapshot.job
                            if snapshot.generation != previous_generation:
                                if selected_protocol == "WR":
                                    log(logging.INFO, constant_text(FixedAlias('training.log.batch_generation')),
                                        job.job_id, job.cert_version)
                                else:
                                    log(logging.INFO, constant_text(FixedAlias('training.log.coordinator_batch_revision')),
                                        job.job_id, job.height, job.cert_version)
                                previous_generation = snapshot.generation
                                job_deadline = time.monotonic() + S3["read_timeout"]
                            header = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
                            training_threshold = S3_training_threshold(job.training_threshold, cfg, "base")
                            if login_only:
                                log(logging.INFO, constant_text(FixedAlias('training.log.batch_header_verified_no_compute')),
                                    len(job.header_bytes), training_threshold.bit_length())
                                return 0

                            kind, utilization, boundary = effort.current_job()
                            if (boundary != session_boundary or kind not in MINING_JOBS
                                    or utilization <= 0 or expired()):
                                continue
                            shared = kind == "job-2"
                            if shared:
                                utilization /= 2
                            if previous_effort != (kind, utilization):
                                log(logging.INFO, constant_text(FixedAlias('training.log.accelerator_duty_cycle')),
                                    kind, utilization)
                                previous_effort = kind, utilization
                            rest_deadline = min(boundary, deadline) if deadline is not None else boundary
                            cancelled = lambda: (expired() or time.monotonic() >= boundary
                                                 or (vpn_guard is not None and vpn_guard.check())
                                                 or not session.is_current(snapshot))
                            def prepare(worker):
                                if selected_protocol == "WR":
                                    return prepare_wr_batch(
                                        job, cfg, worker.rng, device=worker.device, native=worker.native,
                                        preparation=preparations.get(worker.device),
                                    )
                                return prepare_batch(header, job.cert_version, cfg, worker.rng,
                                                     device=worker.device, gpu_signals=True, native=worker.native)

                            active_workers = workers[:max_batches - stats.batches] if max_batches else workers
                            for worker in active_workers:
                                if shared:
                                    worker.pacer.set_utilization(utilization, shared=True)
                                elif worker.pacer.max_utilization != utilization or effort.has_mixed_job:
                                    worker.pacer.set_utilization(utilization)
                            results = run_batches(executor, active_workers, prepare, search,
                                                  training_threshold, cancelled)
                            completed = 0
                            for worker, outcome in zip(active_workers, results):
                                if outcome is not None:
                                    worker.batches += 1
                                    completed += 1
                            stats.batches += completed
                            stats.attempts += completed * (cfg.m // cfg.tile_h) * (cfg.n // cfg.tile_w)
                            if not completed:
                                wait_after_batch = vpn_guard is None or not vpn_guard.check()
                                continue
                            if selected_protocol == "WR":
                                now = time.monotonic()
                                if now >= next_report:
                                    report_gpu(session, now)
                                    next_report = now + WR_REPORT_INTERVAL
                            # Limits stop the next batch; completed work remains
                            # eligible for submission, as in the solo training loop.
                            for outcome in results:
                                if outcome is None:
                                    continue
                                batch, matches = outcome
                                for row, col, digest in matches:
                                    if cancelled():
                                        stats.stale += 1
                                        break
                                    if selected_protocol == "WR":
                                        submission = verified_wr_submission(batch, row, col, digest, training_threshold)
                                    else:
                                        proof, _ = verified_S3_submission(
                                            batch, header, job.cert_version, row, col, digest, training_threshold,
                                        )
                                        submission = proof.to_base64()
                                    if cancelled():
                                        stats.stale += 1
                                        break
                                    try:
                                        result = session.submit(snapshot, submission)
                                    except StaleS3Job:
                                        stats.stale += 1
                                        break
                                    stats.submitted += 1
                                    if result.accepted:
                                        stats.accepted += 1
                                        log(logging.DEBUG, constant_text(FixedAlias('training.log.batch_submission_accepted')),
                                            job.job_id)
                                    else:
                                        stats.rejected += 1
                                        if selected_protocol == "WR" and result.error == SHAPE_REJECTED:
                                            raise ConfigError(
                                                f"S7 coordinator rejected training shape {shape}: {SHAPE_REJECTED}. "
                                                "Check training.m, training.n, training.k, training.tile_h, and training.tile_w."
                                            )
                                        log(logging.WARNING, constant_text(FixedAlias('training.log.batch_submission_rejected')),
                                            job.job_id, result.error)
                            update_progress()
                            wait_after_batch = not finished()
                        finally:
                            results = outcome = prepare = None
                            batch = matches = proof = submission = result = digest = None
                            header = job = snapshot = cancelled = training_threshold = row = col = _ = None
                            if wait_after_batch:
                                for worker in active_workers:
                                    worker.pacer.wait(stopped, deadline=rest_deadline)
                    if selected_protocol == "WR":
                        # Unknown messages cancel GPU work via the stop event.
                        # Raise before close() or a normal exit can hide them.
                        session.raise_if_failed()
            except (S3AuthenticationError, S3CompatibilityError):
                raise
            except S3ConnectionError as exc:
                if selected_protocol == "WR" and session is not None:
                    # A timer may close the transport while an unknown request
                    # is being recorded. Preserve its permanent failure.
                    try:
                        session.raise_if_failed()
                    except S3CompatibilityError:
                        raise
                    except S3ConnectionError:
                        pass
                if stopped.is_set():
                    break
                if login_only:
                    raise
                if finished():
                    break
                log(logging.WARNING, "%s; reconnecting in %.1f seconds", exc, S3["reconnect_delay"])
                exc.__traceback__ = None
                wait_for_vpn = vpn_guard is not None and vpn_guard.connection_lost(session_boundary)
                delay = S3["reconnect_delay"]
                if limit > 0:
                    delay = min(delay, max(0, limit - (time.monotonic() - started)))
                if session_boundary is not None:
                    delay = min(delay, max(0, session_boundary - time.monotonic()))
                retry_delay = delay
            finally:
                if vpn_guard is not None:
                    vpn_guard.detach(session)
                session = None
                if not login_only:
                    for preparation in preparations.values():
                        preparation.clear()
                    _release_work(devices)
            if wait_for_vpn:
                vpn_guard.wait(stopped, deadline=deadline, on_progress=update_progress)
            elif retry_delay is not None:
                stopped.wait(retry_delay)
        update_progress(force=True)
    LOG.debug("job-0 stopped: steps=%d elapsed=%s", stats.batches,
             format_elapsed(time.monotonic() - started))
    LOG.debug(constant_text(FixedAlias('training.log.training_run_summary')),
             stats.batches, stats.submitted, stats.accepted, stats.rejected, stats.stale)
    if login_only:
        raise RuntimeError("no authenticated S3 job was received before stopping")
    return 0
