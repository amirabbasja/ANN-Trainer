"""Concurrent KEX session for the custom S6 trainer.

Only network framing and job lifecycle live here. GPU work and official
S6 proof generation are performed by the caller after authentication.
"""

from __future__ import annotations

from .fixed_constants import constant_text, neutral_protocol_text

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any

from . import S3_methods as methods
from .S3_protocol import KEXDialect, S3Job, S3ProtocolError
from .S3_transport import MAX_IO, S3ConnectionError, S3Transport


class S3AuthenticationError(S3ConnectionError):
    """The S3 explicitly refused the configured login."""


class S3CompatibilityError(S3ConnectionError):
    """The S3 protocol cannot be used safely without a client update."""


class StaleS3Job(RuntimeError):
    """A submission belongs to work that the session has replaced."""


@dataclass(frozen=True)
class JobSnapshot:
    job: S3Job
    generation: int


@dataclass(frozen=True)
class ShareResult:
    accepted: bool
    error: str | None = None


@dataclass
class _Pending:
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: S3ConnectionError | None = None


def _seconds(value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("S3 wait timeout must be a finite number")
    try:
        seconds = float(value)
    except OverflowError:
        seconds = math.inf
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and not allow_zero):
        raise ValueError("S3 wait timeout must be a positive finite number")
    return seconds






class S3Session:
    """Receive job replacements while the caller's GPU is busy.

    Each instance represents one connection. Reconnection uses a new instance
    with the same configured host; S3 redirect targets are never followed.
    """

    AUTH_METHOD = methods.COMPUTE_SESSION

    MAX_FRAME_BYTES = MAX_IO
    MAX_PENDING_SUBMISSIONS = 64

    def __init__(self, settings: dict, stop_event: threading.Event | None = None) -> None:
        self.settings = dict(settings)
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        compression = self.settings.get("compression", "none")
        if compression not in ("none", "gzip"):
            raise S3ProtocolError("S3 compression must be none or gzip")
        self.dialect = self._create_dialect()
        self.transport = S3Transport(
            host=self.settings["host"], port=self.settings["port"],
            tls=self.settings.get("tls", False), ca_file=self.settings.get("ca_file"),
            connect_timeout=self.settings.get("connect_timeout", 15),
            read_timeout=self.settings.get("read_timeout", 60),
        )
        self._condition = threading.Condition()
        self._send_lock = threading.Lock()
        self._closed_event = threading.Event()
        self._opened = False
        self._authorized = False
        self._failure: S3ConnectionError | None = None
        self._job: S3Job | None = None
        self._generation = 0
        self._next_id = 1
        self._auth_id: int | None = None
        self._pending: dict[int, _Pending] = {}
        self._negotiated_gzip = False
        self._reader: threading.Thread | None = None
        self._stop_watcher: threading.Thread | None = None

    def _create_dialect(self) -> KEXDialect:
        """Select this session's identity without reading another S3's field."""
        return KEXDialect(
            KEX_uname=self.settings.get("KEX_uname", ""), worker=self.settings.get(constant_text(b'(8$%y!qo'), ""),
            password=self.settings.get("password", "x"),
            agent=self.settings.get("agent", "LRAEP-trainer/0.1"),
            request_gzip=self.settings.get("compression", "none") == "gzip",
        )

    @property
    def negotiated_gzip(self) -> bool:
        with self._condition:
            return self._negotiated_gzip

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self.stop_event.is_set() or self._closed_event.is_set():
            raise S3ConnectionError("S3 session is stopped")

    @neutral_protocol_text()
    def open(self) -> S3Session:
        with self._condition:
            self._raise_failure()
            if self._authorized:
                return self
            if self._opened:
                raise S3ConnectionError("S3 session is already opening")
            self._opened = True
        deadline = time.monotonic() + self.transport.connect_timeout
        try:
            self.transport.open()
            requests = self.dialect.auth_requests(start_id=self._next_id)
            request = None
            try:
                self._auth_id = next(r["id"] for r in requests if r["method"] == self.AUTH_METHOD)
                self._next_id = max(r["id"] for r in requests) + 1
                self._reader = threading.Thread(target=self._read_loop, name="LRAEP-S3-reader", daemon=True)
                self._stop_watcher = threading.Thread(target=self._watch_stop, name="LRAEP-S3-stop", daemon=True)
                self._reader.start()
                self._stop_watcher.start()
                # Registration can be silent. Open the session immediately.
                with self._send_lock:
                    for request in requests:
                        self._send(request)
            finally:
                requests.clear()
                del requests, request
            with self._condition:
                while not self._authorized:
                    self._raise_failure()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise S3ConnectionError("S3 authentication timed out")
                    self._condition.wait(min(remaining, 0.1))
                self._raise_failure()
            return self
        except (S3ConnectionError, S3ProtocolError):
            self.close()
            raise

    def __enter__(self) -> S3Session:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _fail(self, error: S3ConnectionError) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = error
            self._authorized = False
            self._job = None
            self._generation += 1
            self._closed_event.set()
            for pending in self._pending.values():
                if not pending.event.is_set():
                    pending.error = self._failure
                    pending.event.set()
            self._condition.notify_all()
        self.transport.close()

    def close(self) -> None:
        self._fail(S3ConnectionError("S3 session closed"))
        current = threading.current_thread()
        for thread in (self._reader, self._stop_watcher):
            if thread is not None and thread is not current:
                thread.join(timeout=1)

    def _watch_stop(self) -> None:
        while not self._closed_event.wait(0.05):
            if self.stop_event.is_set():
                self._fail(S3ConnectionError("S3 session stopped"))
                return

    @neutral_protocol_text()
    def _send(self, message: dict[str, Any]) -> None:
        try:
            message = methods.stratum_request(message)
            try:
                self.transport.send_message(message, profile="s3", max_frame=self.MAX_FRAME_BYTES)
            except (TypeError, ValueError, RecursionError):
                raise S3ProtocolError("S3 request cannot be serialized within the frame limit") from None
        finally:
            del message

    @neutral_protocol_text()
    def _read_loop(self) -> None:
        try:
            while not self._closed_event.is_set():
                try:
                    response = self.transport.receive_message(profile="s3", max_frame=self.MAX_FRAME_BYTES)
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    raise S3ProtocolError("S3 sent a malformed or oversized message") from None
                try:
                    if response is not None:
                        self._handle(response)
                finally:
                    del response
        except S3ConnectionError as error:
            self._fail(error)
        except S3ProtocolError as error:
            self._fail(S3CompatibilityError(str(error)))
        except Exception:
            self._fail(S3ConnectionError("S3 reader failed"))

    def _handle(self, response: dict[str, Any]) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
        method = response.get("method")
        notification = methods.stratum_notification_method(method)
        if notification == methods.WORK_UPDATE:
            job = self.dialect.parse_job(
                response.get("params"), self.settings.get(constant_text(b'#KP6n&cV*a#?!+1(Zbc#&dA9'))
            )
            with self._condition:
                if self._closed_event.is_set():
                    return
                if job != self._job:
                    self._job = job
                    self._generation += 1
                    self._condition.notify_all()
            return
        if notification == methods.COMPUTE_RECONNECT:
            self._fail(S3ConnectionError("S3 requested reconnection to the configured endpoint"))
            return
        if method is not None:
            # The target inside each notify is authoritative. Ignore separate
            # difficulty notifications and unknown extension notifications.
            return
        request_id = response.get("id")
        if type(request_id) is not int:
            return
        with self._condition:
            if request_id == self._auth_id and not self._authorized:
                if not self.dialect.accept(response):
                    self._fail(S3AuthenticationError("S3 rejected authentication"))
                    return
                self._negotiated_gzip = self.dialect.negotiate_gzip(response)
                self._authorized = True
                self._condition.notify_all()
            elif request_id in self._pending:
                pending = self._pending[request_id]
                if not pending.event.is_set():
                    pending.response = response
                    pending.event.set()
            # Unmatched subscribe acknowledgements/errors and late replies
            # cannot authorize a session or credit a submitted share.

    def wait_job(self, timeout: float = 1) -> JobSnapshot | None:
        deadline = time.monotonic() + _seconds(timeout, allow_zero=True)
        with self._condition:
            if not self._opened:
                raise S3ConnectionError("S3 session is not open")
            while True:
                self._raise_failure()
                if self._authorized and self._job is not None:
                    return JobSnapshot(self._job, self._generation)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.1))

    def is_current(self, snapshot: JobSnapshot) -> bool:
        with self._condition:
            return (
                isinstance(snapshot, JobSnapshot) and self._authorized
                and self._failure is None and not self.stop_event.is_set()
                and not self._closed_event.is_set()
                and snapshot.generation == self._generation and snapshot.job == self._job
            )

    @neutral_protocol_text()
    def submit(self, snapshot: JobSnapshot, proof_b64: str, timeout: float | None = None) -> ShareResult:
        wait_seconds = _seconds(self.transport.read_timeout if timeout is None else timeout)
        with self._send_lock:
            with self._condition:
                self._raise_failure()
                if not self.is_current(snapshot):
                    raise StaleS3Job("S3 replaced this training job before submission")
                if len(self._pending) >= self.MAX_PENDING_SUBMISSIONS:
                    raise S3ConnectionError("Too many pending S3 submissions")
                request_id = self._next_id
                self._next_id += 1
                negotiated = self._negotiated_gzip
            # Compression may be expensive; keep it outside the state lock so
            # new notify messages can immediately invalidate the old job.
            request = self.dialect.submit_request(snapshot.job, proof_b64, request_id, negotiated)
            try:
                pending = _Pending()
                with self._condition:
                    self._raise_failure()
                    if not self.is_current(snapshot):
                        raise StaleS3Job("S3 replaced this training job before submission")
                    self._pending[request_id] = pending
                try:
                    self._send(request)
                except (S3ConnectionError, S3ProtocolError):
                    with self._condition:
                        self._pending.pop(request_id, None)
                    self._fail(S3ConnectionError(constant_text(b'^puCx&&Jilht$H!!-v7f&db8X')))
                    raise
            finally:
                del request
        if not pending.event.wait(wait_seconds):
            with self._condition:
                self._pending.pop(request_id, None)
            error = S3ConnectionError(constant_text(b'^puCx&&JilhsMOr$;i;l!o$GA%EHOhhttl=!o!Ei($f'))
            self._fail(error)
            raise error
        with self._condition:
            self._pending.pop(request_id, None)
        if pending.error is not None:
            raise pending.error
        if pending.response is None:
            raise S3ConnectionError(constant_text(b'^puCh%gEHi!-vJf!N}FZhsMOr$;i;l!o$GM$-sxx&%%e)&&Jil'))
        if self.dialect.accept(pending.response):
            return ShareResult(True)
        # Do not echo untrusted S3 text, credentials, or proof data to logs.
        return ShareResult(False, constant_text(b'^puCy!p*|O)561t)X&D%!T'))
