"""Our GPU training loop connected directly to a supported S6 S3."""

from .fixed_constants import constant_text

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

from .gpu import TrainingCancelled
from .gpu_pacing import GpuPacer
from .effort import JobSequence, ScheduledSideJob
from .progress import LiveProgress, format_elapsed, main_status
from .protocol import (TrainingConfig, load_official_helpers, S3_share_target,
                       prepare_batch, search_batch, verified_S3_share)
from .telemetry import GpuUtilization
from .telegram import TelegramUpdates
from .job_control import JobEvents, VPNJobGuard
from .progress import job_status
from . import native_cuda
from .word_codec import word_matches

LOG = logging.getLogger("LRAEP_trainer")
WR_REPORT_INTERVAL = 120.0
WR_STARTUP_REPORT_INTERVAL = 10.0
WR_STARTUP_REPORT_WINDOW = 120.0


@dataclass
class S3TrainingStats:
    batches: int = 0
    attempts: int = 0
    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    stale: int = 0


def S3_hashrate(stats: S3TrainingStats, cfg: TrainingConfig, elapsed: float) -> float:
    """Return completed S6 matrix work per second in raw S3 H/s units.

    At rank 128 each hash tile contributes tile_h * tile_w * k units to
    the target scaling. For our fully tiled matrices this is m * n * k
    per completed batch, rather than the number of final BLAKE3 tickets.
    """
    return stats.batches * cfg.m * cfg.n * cfg.k / max(elapsed, 0.001)


def format_hashrate(rate: float) -> str:
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
    if selected == "S7" or word_matches("legacyMethod", selected):
        selected = "WR"
    if selected not in ("KEX", "WR"):
        raise ValueError("S3.protocol must select KEX or WR")
    return selected


def validate_S3_endpoint(settings: dict) -> None:
    """Require the user's endpoint before helper setup or any network activity."""
    host = settings.get("host", "")
    if not isinstance(host, str) or not host.strip():
        selected = settings.get("protocol")
        profile = "WR" if selected in ("WR", "S7") or word_matches("legacyMethod", selected) else "KEX"
        raise ValueError(
            f"fill S3.host for a flat configuration, or S3.{profile}{constant_text(b'iO<N?(}&K<htto(ht$u;)xyJv#K_6P&cM>u#?#Ko$%}{3&eP9_-pJC`hxC+(&&br%$;QgUhsf23?)Qhr!^73W)YJ')}"
        )


def validate_S3_identity(settings: dict, selected_protocol: str | None = None, *,
                           validate_address: bool = True) -> None:
    """Check configuration before loading helpers or opening a connection."""
    selected = select_S3_protocol(settings if selected_protocol is None else {"protocol": selected_protocol})
    identity_field = "WR_S2" if selected == "WR" else "KEX_uname"
    identity = settings.get(identity_field, "")
    worker = settings.get(constant_text(b'(8$%y!qo'), "")
    if not isinstance(identity, str) or not identity.strip():
        description = "WR payout address" if selected == "WR" else "KEX username or payout address"
        raise ValueError(f"fill S3.{identity_field} in your configuration with your {description}")
    if not isinstance(worker, str):
        raise ValueError(constant_text(b'^puIv$koik)rZQ`)YFH>!iUC()YH|@$-n'))
    if any(character.isspace() for character in identity + worker):
        raise ValueError(f"S3.{identity_field}{constant_text(b'hsMdnhxC+*(8$%y!qtb$($v$3$;i`(#K_6h#?Hxy(9h1(!qnHs#KH')}")
    if identity.startswith(constant_text(b')X2-ood')):
        raise ValueError("remove the solo: prefix to use the requested S3 training mode")
    if validate_address and identity.startswith(constant_text('account_prefix')):
        from LRAEP_gateway.blockchain_utils.blockchain_utils import get_script_pubkey_from_p2tr_address

        try:
            get_script_pubkey_from_p2tr_address(identity)
        except ValueError:
            raise ValueError(f"S3.{identity_field}{constant_text(b'hsx5_(}%>!$<xNp$%n>=(Z<Wp!-w>dhx5kQ)yT-xhsMLh)xy-&')}") from None


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
    stopped = threading.Event()
    started = time.monotonic()
    effort = None if login_only else JobSequence(settings["job_sequence"], started=started)
    deadline = started + settings["max_seconds"] if settings["max_seconds"] else None
    with _signal_stop(stopped, settings["max_seconds"]), ExitStack() as resources:
        if effort is not None:
            resources.enter_context(effort.stopping(stopped))
            if not effort.wait(stopped, deadline=deadline):
                return 0
        return _run_S3_training(configuration, login_only=login_only, stopped=stopped,
                                  effort=effort, started=started)


def _release_work(device):
    gc.collect()
    if torch.cuda.is_initialized():
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def _run_S3_training(configuration, *, login_only, stopped, effort, started):
    """Authenticate, process live jobs, and submit verified S3-specific work.

    ``login_only`` stops after the first authenticated job and does no GPU work.
    Continuous sessions reconnect after transport failures, invalidating every
    job and pending submission from the disconnected session.
    """
    from .config import resolve_S3_settings
    from .S3_client import (S3AuthenticationError, S3CompatibilityError,
                              S3Session, StaleS3Job)
    from .S3_transport import S3ConnectionError

    S3 = resolve_S3_settings(configuration["S3"])
    selected_protocol = select_S3_protocol(S3)
    validate_S3_endpoint(S3)
    validate_S3_identity(S3, selected_protocol, validate_address=False)
    session_class = S3Session
    session_options = {}
    if selected_protocol == "WR":
        from .S3_diagnostics import UnknownRequestRecorder
        from .wr_client import WRSession
        from .wr_work import prepare_wr_batch, verified_wr_share

        session_class = WRSession
        diagnostic_dir = S3.get("unknown_request_dir") or (
            Path(configuration["training"]["output_dir"]) / "unknown-S3-requests"
        )
        session_options["unknown_handler"] = UnknownRequestRecorder(diagnostic_dir, S3)
    load_official_helpers(configuration["official_source"])
    validate_S3_identity(S3, selected_protocol)
    from LRAEP_trainer import primitives as pm

    settings = configuration["training"]
    cfg = TrainingConfig(settings["m"], settings["n"], settings["k"],
                         settings[constant_text(b')6UDn_|E')], settings[constant_text(b')6UDn_|O')])
    gpu_name = None
    if not login_only:
        device = torch.device(configuration["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("an available CUDA GPU is required for S3 training")
        # Capture the caller's current device before the telemetry thread
        # starts; CUDA's implicit current-device selection is thread-local.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        gpu_name = torch.cuda.get_device_name(device)
        LOG.debug("GPU: %s", gpu_name)
    use_native = (not login_only and (cfg.tile_h, cfg.tile_w) == (8, 8)
                  and native_cuda.available(configuration["device"]))
    if not login_only:
        LOG.debug("Compute backend: %s", "native CUDA DP4A" if use_native else "PyTorch reference")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    stats = S3TrainingStats()
    rng = np.random.default_rng()
    limit = settings["max_seconds"]
    max_batches = settings["max_batches"]
    pacer = GpuPacer(device, 100) if not login_only else None
    deadline = started + limit if limit else None

    def expired():
        return (stopped.is_set() or (limit > 0 and time.monotonic() - started >= limit)
                or (effort is not None and effort.ended()))

    def finished():
        return expired() or (max_batches > 0 and stats.batches >= max_batches)

    with ExitStack() as resources:
        progress = resources.enter_context(LiveProgress())
        monitor = None
        if not login_only:
            monitor = GpuUtilization(str(device)).start()
            resources.callback(monitor.close)

        active_job = "job-0"

        def show_side_job(job):
            update_progress()

        side_job = (ScheduledSideJob(configuration, before_log=progress.clear, on_progress=show_side_job)
                 if not login_only else None)
        if side_job is not None:
            resources.callback(side_job.close)

        def telegram_snapshot():
            now = time.monotonic()
            number, kind, percentage, boundary = effort.position(now)
            remaining = (0 if percentage == -1 else
                         max(0, (boundary if deadline is None else min(boundary, deadline)) - now))
            if kind == "job-1":
                job = side_job.job if side_job is not None else None
                speed = getattr(job, "_progress_epoch", 0) if job is not None else 0
            else:
                speed = (0 if percentage <= 0 or (vpn_guard is not None and vpn_guard.blocked())
                         else S3_hashrate(stats, cfg, now - started))
            return job_status(number, kind,
                              monitor.read(), remaining, speed)

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
            if monitor is not None:
                if events is not None:
                    events.update()
                kind = effort.current_job()[0]
                force = force or active_job != kind
                active_job = kind
                progress.update(telegram_snapshot(), force=force)

        def log(level, message, *args):
            LOG.debug(message, *args)

        def report_gpu(session, sample_time):
            rate = S3_hashrate(stats, cfg, sample_time - started)
            request_id = session.report_info(gpu_name, rate)
            log(logging.DEBUG, constant_text(b'^pJ<(_wt9;!q>>v(}&c;$<v*Og4BnP)xy@&!qn56g2Rp'),
                format_hashrate(rate), request_id)

        update_progress(force=True)
        while not finished():
            if effort is not None and not effort.wait(stopped, deadline=deadline):
                break
            if side_job is not None and side_job.run_if_active(effort, stopped, pacer, deadline=deadline):
                continue
            if vpn_guard is not None and not vpn_guard.resume():
                vpn_guard.wait(stopped, deadline=deadline, on_progress=update_progress)
                continue
            session = None
            retry_delay = None
            wait_for_vpn = False
            session_boundary = effort.current_job()[2] if effort is not None else None
            try:
                session_settings = dict(S3)
                connect_deadline = deadline
                if effort is not None and effort.exit_deadline is not None:
                    connect_deadline = (effort.exit_deadline if connect_deadline is None else
                                        min(connect_deadline, effort.exit_deadline))
                if effort is not None:
                    boundary = effort.current_job()[2]
                    connect_deadline = boundary if connect_deadline is None else min(boundary, connect_deadline)
                if connect_deadline is not None:
                    remaining = max(0.001, connect_deadline - time.monotonic())
                    session_settings["connect_timeout"] = min(S3["connect_timeout"], remaining)
                if vpn_guard is not None and vpn_guard.enabled:
                    session_settings["connect_timeout"] = min(session_settings.get("connect_timeout", 15), 5.0)
                session = session_class(session_settings, stop_event=stopped, **session_options)
                if vpn_guard is not None:
                    vpn_guard.attach(session)
                with session:
                    log(logging.INFO, constant_text(b'^puCj($mkv$<xln#?!*XhsM)~g4CUY!<>iL)yT-fhs4Or*VV$*)XvDsnu62'),
                        S3["host"], S3["port"],
                        "gzip" if session.negotiated_gzip else "none")
                    report_connection_started = time.monotonic()
                    report_started = report_connection_started
                    reported_work = False
                    if selected_protocol == "WR" and not login_only:
                        report_gpu(session, report_started)
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
                            if boundary != session_boundary or kind != "job-0" or utilization <= 0:
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
                                    log(logging.INFO, constant_text(b'^pJ<m$i<q1)Q7;r$->pf)6U4rnu5a'), job.job_id, job.cert_version)
                                else:
                                    log(logging.INFO, constant_text(b'^puCq$i<q1)Q8W)&cM&pnu5cJ#KP6n&cV*a#?!)@@q)t'),
                                        job.job_id, job.height, job.cert_version)
                                previous_generation = snapshot.generation
                                job_deadline = time.monotonic() + S3["read_timeout"]
                            header = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
                            target = S3_share_target(job.target, cfg, "base")
                            if login_only:
                                log(logging.INFO, constant_text(b'=F-#8!pYOl#KzOY!-vht#fQ<t)y~1r!o!`1&%(yT!qu9B!-vJ*)56q?httN@z{1m-g2RW!&ePPKhsnr?;P>)}(8$%yhu6Z@!N}Fh!ov'),
                                    len(job.header_bytes), target.bit_length())
                                return 0

                            kind, utilization, boundary = effort.current_job()
                            if kind != "job-0" or utilization <= 0 or expired():
                                continue
                            if pacer.max_utilization != utilization:
                                pacer.set_utilization(utilization)
                            if previous_effort != (kind, utilization):
                                log(logging.INFO, constant_text(b'g4Bn?!NJJY(}&>q@`uCH)82~2-o(qohttN@z{1m=hl0a`f&'), kind, utilization)
                                previous_effort = kind, utilization
                            rest_deadline = min(boundary, deadline) if deadline is not None else boundary
                            cancelled = lambda: (expired() or time.monotonic() >= boundary
                                                 or (vpn_guard is not None and vpn_guard.check())
                                                 or not session.is_current(snapshot))
                            pacer.begin()
                            if selected_protocol == "WR":
                                batch = prepare_wr_batch(job, cfg, rng,
                                                              device=configuration["device"], native=use_native)
                            else:
                                batch = prepare_batch(header, job.cert_version, cfg, rng,
                                                      device=configuration["device"], gpu_signals=True,
                                                      native=use_native)
                            if cancelled():
                                pacer.finish(cancelled=True)
                                wait_after_batch = vpn_guard is None or not vpn_guard.check()
                                continue
                            try:
                                matches = search_batch(batch, target, configuration["device"], cancelled)
                            except TrainingCancelled:
                                pacer.finish(cancelled=True)
                                wait_after_batch = vpn_guard is None or not vpn_guard.check()
                                continue
                            pacer.finish()
                            stats.batches += 1
                            stats.attempts += (cfg.m // cfg.tile_h) * (cfg.n // cfg.tile_w)
                            if selected_protocol == "WR":
                                now = time.monotonic()
                                interval = (WR_STARTUP_REPORT_INTERVAL
                                            if now - report_connection_started < WR_STARTUP_REPORT_WINDOW
                                            else WR_REPORT_INTERVAL)
                                if not reported_work or now - report_started >= interval:
                                    report_gpu(session, now)
                                    report_started = now
                                    reported_work = True
                            # Limits stop the next batch; completed work remains
                            # eligible for submission, as in the solo training loop.
                            for row, col, digest in matches:
                                if cancelled():
                                    stats.stale += 1
                                    break
                                if selected_protocol == "WR":
                                    submission = verified_wr_share(batch, row, col, digest, target)
                                else:
                                    proof, _ = verified_S3_share(
                                        batch, header, job.cert_version, row, col, digest, target,
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
                                    log(logging.DEBUG, constant_text(b'^puCj#Kgka)561t)78e#$<E2Zht$u;)xw9($i<q1)B'), job.job_id)
                                else:
                                    stats.rejected += 1
                                    log(logging.WARNING, constant_text(b'^puCy!p*|O)561t)78e#$<E2Zht$u;)xw9($i<q1)SZWd)B'), job.job_id, result.error)
                            update_progress()
                            wait_after_batch = not finished()
                        finally:
                            batch = matches = proof = submission = result = digest = None
                            header = job = snapshot = cancelled = target = row = col = _ = None
                            if wait_after_batch:
                                pacer.wait(stopped, deadline=rest_deadline)
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
                retry_delay = delay
            finally:
                if vpn_guard is not None:
                    vpn_guard.detach(session)
                session = None
                if not login_only:
                    _release_work(device)
            if wait_for_vpn:
                vpn_guard.wait(stopped, deadline=deadline, on_progress=update_progress)
            elif retry_delay is not None:
                stopped.wait(retry_delay)
        update_progress(force=True)
    LOG.debug("job-0 stopped: steps=%d elapsed=%s", stats.batches,
             format_elapsed(time.monotonic() - started))
    LOG.debug(constant_text(b'^puCw)yB@r&dI=s)YHh<*TTb{hsDOz#LvRinu5cJ)Y8Su&ePMv!<vG_hsMOj!q?Nn!<vG_ht<N(!o<_U!<vG_ht$)?%fgz1!v'),
             stats.batches, stats.submitted, stats.accepted, stats.rejected, stats.stale)
    if login_only:
        raise RuntimeError("no authenticated S3 job was received before stopping")
    return 0
