"""Concurrent, bounded S7 session using the independently recovered codec."""

from __future__ import annotations

from .fixed_constants import constant_ascii_matches, constant_text, neutral_protocol_text

from dataclasses import replace
import logging
import threading
import time
from typing import Callable

from .S3_client import (JobSnapshot, S3AuthenticationError, S3CompatibilityError,
                          S3Session, _seconds)
from .S3_protocol import S3ProtocolError
from .S3_transport import S3ConnectionError
from .wr_protocol import UnknownWRRequest, WRDialect


LOG = logging.getLogger("LRAEP_trainer")


class _AuthMethod:
    def __get__(self, instance, owner=None):
        return constant_text(b'%gDga$p')


class WRSession(S3Session):
    """A S7 connection with fatal, recorded unknown-message handling.

    ``unknown_handler`` receives the encoded wire frame, neutral parsed object
    when available, and a fixed diagnostic reason. It runs once before closing the
    connection. Compatibility failures never trigger automatic reconnection.
    """

    AUTH_METHOD = _AuthMethod()
    MAX_PENDING_REPORTS = 4

    def __init__(self, settings: dict, stop_event: threading.Event | None = None,
                 unknown_handler: Callable[..., None] | None = None) -> None:
        if settings.get("compression", "none") != "none":
            raise S3ProtocolError(constant_text(b'^pJ<d$jaB%!qn8x$jOJw($v$3#lnZl$jQP'))
        super().__init__(settings, stop_event)
        self._worker_nonce: bytes | None = None
        self._unknown_handler = unknown_handler
        self._recording_unknown = False
        self._compatibility_failure: S3CompatibilityError | None = None
        self._report_pending: dict[int, float] = {}
        self._latest_report_id: int | None = None
        self._latest_report_accepted: bool | None = None

    def _create_dialect(self) -> WRDialect:
        """Initialize only WR's selected identity during shared session setup."""
        return WRDialect(
            WR_S2=self.settings.get("WR_S2", ""),
            worker=self.settings.get(constant_text(b'(8$%y!qo'), ""),
            agent=self.settings.get("agent", "LRAEP-trainer/0.1"),
        )

    def _raise_failure(self) -> None:
        if self._compatibility_failure is not None:
            raise self._compatibility_failure
        super()._raise_failure()

    def _fail(self, error: S3ConnectionError) -> None:
        with self._condition:
            self._report_pending.clear()
            if self._compatibility_failure is not None:
                error = self._compatibility_failure
                self._failure = error
        super()._fail(error)

    def _expire_reports_locked(self) -> None:
        now = time.monotonic()
        expired = [request_id for request_id, deadline in self._report_pending.items() if now >= deadline]
        for request_id in expired:
            del self._report_pending[request_id]
            LOG.warning(constant_text(b'^pJ<(_wt9;!q>>v(}%{y%*n{m%fiFJ!pg$Q(}&Z}%EH5k$kNj'))

    @property
    def latest_report_id(self) -> int | None:
        with self._condition:
            return self._latest_report_id

    @property
    def latest_report_accepted(self) -> bool | None:
        """Latest report acknowledgement; None means pending or unconfirmed."""
        with self._condition:
            self._expire_reports_locked()
            return self._latest_report_accepted

    @neutral_protocol_text()
    def report_info(self, gpu_name: str, hashrate: float) -> int:
        """Send actual device/rate telemetry without waiting for its reply.

        Reports share the request-ID allocator and serialized socket writes
        with submissions, but have separate bounded acknowledgement state.
        A report acknowledgement never authorizes or credits a share.
        """
        request = self.dialect.report_request(gpu_name, hashrate, 0)
        with self._send_lock:
            with self._condition:
                self._raise_failure()
                if not self._authorized:
                    raise S3ConnectionError(constant_text(b'^pJ<t!qn8x$jOJw($v$3#?sTz!pYOl#KzOYhsDCd$koD!)xy`v)zi+&z=z=X@`ujJ!N}Fh#?#Ko$p'))
                self._expire_reports_locked()
                if len(self._report_pending) >= self.MAX_PENDING_REPORTS:
                    del self._report_pending[next(iter(self._report_pending))]
                    LOG.warning(constant_text(b'^pJ<c&eX)l)x*NWhsMc=(#gie%*n{m%fiFJ!o!E)_wt9;!q>>v(*'))
                request_id = self._next_id
                # Validate the same uint64 representation used by all WR
                # requests before reserving an ID in this session.
                request = self.dialect.report_request(gpu_name, hashrate, request_id)
                self._next_id += 1
                self._report_pending[request_id] = time.monotonic() + self.transport.read_timeout
                self._latest_report_id = request_id
                self._latest_report_accepted = None
            try:
                self._send(request)
            except (S3ConnectionError, S3ProtocolError):
                self._fail(S3ConnectionError(constant_text(b'^pJ<(_wt9;!q>>v(}&c;$-{@i#?H&a!v')))
                raise
        return request_id

    def _watch_stop(self) -> None:
        while not self._closed_event.wait(0.05):
            with self._condition:
                self._expire_reports_locked()
            if self.stop_event.is_set():
                self._fail(S3ConnectionError("S3 session stopped"))
                return

    @neutral_protocol_text()
    def _send(self, message: dict) -> None:
        try:
            try:
                self.transport.send_message(message, profile="wr", max_frame=self.MAX_FRAME_BYTES)
            except (TypeError, ValueError, RecursionError):
                raise S3ProtocolError("WR request cannot be serialized within the frame limit") from None
        finally:
            del message

    def _unknown(self, frame: bytes, response: dict | None, reason: str) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
            # Cancel active GPU work before filesystem persistence. Keep the
            # reader alive until the evidence has been saved, then publish the
            # permanent compatibility failure and wake every waiter together.
            self._authorized = False
            self._job = None
            self._generation += 1
            self._recording_unknown = True
            self._compatibility_failure = S3CompatibilityError(
                constant_text(b'^2yA}$k54$^pJ<u!q(El)YF`Y)78e#$<E2Zht$)^*Vn?s'),
            )
        diagnostic_failed = False
        if self._unknown_handler is not None:
            try:
                self._unknown_handler(frame, response, reason, fatal=True)
            except Exception:
                diagnostic_failed = True
        message = constant_text(b'^2yA}$k54$^pJ<u!q(El)YF`Y)78e#$<E2Zht$)^*Vn?s')
        if diagnostic_failed:
            message += "; diagnostic recording failed"
        error = S3CompatibilityError(message)
        with self._condition:
            # A timer, signal or explicit close may have arrived while fsync
            # ran. The recorded compatibility failure remains the reason this
            # session cannot resume, even when ordinary shutdown raced it.
            self._failure = error
            self._compatibility_failure = error
        self.stop_event.set()
        self._fail(error)
        with self._condition:
            self._recording_unknown = False
            self._condition.notify_all()

    @neutral_protocol_text()
    def _read_loop(self) -> None:
        try:
            while not self._closed_event.is_set():
                try:
                    response = self.transport.receive_message(profile="wr", max_frame=self.MAX_FRAME_BYTES)
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    frame = self.transport.message_frame()
                    try:
                        self._unknown(frame, None, "Malformed or oversized WR frame")
                    finally:
                        self.transport.discard_message_frame()
                        del frame
                    return
                except S3ConnectionError as error:
                    frame = self.transport.message_frame()
                    try:
                        if frame and not self._closed_event.is_set():
                            self._unknown(frame, None, "WR connection closed with an incomplete frame")
                        else:
                            self._fail(error)
                    finally:
                        self.transport.discard_message_frame()
                        del frame
                    return
                try:
                    if response is not None:
                        self._handle(response)
                except UnknownWRRequest as error:
                    frame = self.transport.message_frame()
                    try:
                        if self.settings.get("stop_on_unknown_request", True):
                            self._unknown(frame, response, str(error))
                            return
                        if self._unknown_handler is not None:
                            try:
                                self._unknown_handler(frame, response, str(error), fatal=False)
                            except Exception:
                                self._unknown(frame, response, "Could not record the unknown WR message")
                                return
                    finally:
                        del frame
                except S3ProtocolError:
                    frame = self.transport.message_frame()
                    try:
                        self._unknown(frame, response, "Malformed or unsupported WR message")
                        return
                    finally:
                        del frame
                finally:
                    self.transport.discard_message_frame()
                    del response
        except Exception:
            self._fail(S3ConnectionError("WR reader failed"))

    def _handle(self, response: dict) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
        kind = self.dialect.classify(response)
        if constant_ascii_matches('extracounter', kind):
            worker_nonce = self.dialect.parse_extranonce(response["result"])
            with self._condition:
                if self._closed_event.is_set():
                    return
                if worker_nonce != self._worker_nonce:
                    self._worker_nonce = worker_nonce
                    if self._job is not None:
                        self._job = replace(self._job, worker_nonce=worker_nonce)
                    self._generation += 1
                    self._condition.notify_all()
            return
        if kind == constant_text(b'$;i{r!QK'):
            with self._condition:
                if self._closed_event.is_set():
                    return
                job = self.dialect.parse_job(
                    response["result"], self.settings.get(constant_text(b'#KP6n&cV*a#?!+1(Zbc#&dA9')), self._worker_nonce,
                )
                if job != self._job:
                    self._job = job
                    self._generation += 1
                    self._condition.notify_all()
            return
        request_id = response["id"]
        with self._condition:
            self._expire_reports_locked()
            if request_id == self._auth_id and not self._authorized:
                if not self.dialect.accept(response):
                    self._fail(S3AuthenticationError(constant_text(b'^pJ<u!p*|O)561t#?sTz!pYOl#KzOk$jJ')))
                    return
                self._authorized = True
                self._condition.notify_all()
            elif request_id in self._report_pending:
                del self._report_pending[request_id]
                accepted = self.dialect.accept(response)
                if request_id == self._latest_report_id:
                    self._latest_report_accepted = accepted
                if accepted:
                    LOG.debug(constant_text(b'^pJ<f#LUUa(96QZz{10a;P>)}&dI^Z)yl@x&dAA!)xy`v)zb'))
                else:
                    LOG.warning(constant_text(b'^pJ<u!p*|O)561t;P>)}&dI^Z)yl@x&dAA!)xy`v)zb'))
                self._condition.notify_all()
            elif request_id in self._pending:
                pending = self._pending[request_id]
                if not pending.event.is_set():
                    pending.response = response
                    pending.event.set()

    def wait_job(self, timeout: float = 1) -> JobSnapshot | None:
        deadline = time.monotonic() + _seconds(timeout, allow_zero=True)
        with self._condition:
            if not self._opened:
                raise S3ConnectionError("S3 session is not open")
            while True:
                self._raise_failure()
                if self._authorized and self._job is not None and self._job.worker_nonce is not None:
                    return JobSnapshot(self._job, self._generation)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.1))

    def raise_if_failed(self) -> None:
        """Surface a fatal reader failure even after its stop event cancelled GPU work."""
        deadline = time.monotonic() + 1
        with self._condition:
            while self._recording_unknown:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(min(remaining, 0.1))
            if self._compatibility_failure is not None:
                raise self._compatibility_failure
            if self._failure is not None:
                raise self._failure
