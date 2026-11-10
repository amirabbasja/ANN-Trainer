"""KEX's S6 message framing, independent of transport and GPU code.

Protocol references only:
* KEX's published gzip note, gist eaaf4332dbc5ca99d0a78f24a733fffe,
  revision 45cad9645bd9cd57655851fa823e1299eb4d1a0a.
* Captured TLS messages in spark-pearl-miner, revision
  65365633e3bff61b0f29d72109622dd22ff5a379,
  tests/fixtures/capture-kex-8048-authorize.jsonl.
* That revision's docs/protocol/kex.md for submission fields.

All proof construction and matrix arithmetic remain in S6's official
helpers and this project's own GPU implementation.
"""

from __future__ import annotations

from . import S3_methods as methods
from .fixed_constants import FixedAlias, constant_text

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
import gzip
import re
from typing import Any


DEFAULT_AGENT = "LRAEP-trainer/0.1"
MAX_DECODED_PROOF_BYTES = 4 * 1024 * 1024
_HEX = re.compile(r"[0-9a-fA-F]+\Z")


def s2():
    return constant_text('account')


class S3ProtocolError(RuntimeError):
    """A S3 message does not satisfy the supported protocol."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise S3ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise S3ProtocolError(f"{name} must be a string of at most {maximum} characters")
    if (not value and not allow_empty) or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise S3ProtocolError(f"{name} must contain no control characters and must not be empty")
    return value


def _certificate(value: Any) -> int:
    version = _integer(value, constant_text(FixedAlias('training.label.attestation_version')), 1)
    if version not in (1, 2, 3):
        raise S3ProtocolError(f"{constant_text(FixedAlias('training.validation.attestation_version_prefix'))}{version}")
    return version


def _proof_bytes(encoded: str) -> bytes:
    if not isinstance(encoded, str) or not encoded:
        raise S3ProtocolError(constant_text(FixedAlias('training.validation.dense_attestation_canonical_format')))
    if len(encoded) > 4 * ((MAX_DECODED_PROOF_BYTES + 2) // 3):
        raise S3ProtocolError(constant_text(FixedAlias('training.validation.dense_attestation_size_limit')))
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise S3ProtocolError(constant_text(FixedAlias('training.validation.dense_attestation_canonical_format'))) from None
    if not raw or base64.b64encode(raw).decode("ascii") != encoded:
        raise S3ProtocolError(constant_text(FixedAlias('training.validation.dense_attestation_canonical_format')))
    if len(raw) > MAX_DECODED_PROOF_BYTES:
        raise S3ProtocolError(constant_text(FixedAlias('training.validation.dense_attestation_size_limit')))
    return raw


@dataclass(frozen=True)
class S3Job:
    """Immutable training job with its unadjusted, big-endian training threshold."""

    job_id: str
    header_bytes: bytes
    height: int
    cert_version: int
    training_threshold: int
    encoded_training_threshold: str | None = None

    def __post_init__(self) -> None:
        _text(self.job_id, constant_text(FixedAlias('training.field.batch_identifier')), 128)
        if not isinstance(self.header_bytes, bytes) or len(self.header_bytes) != 76:
            raise S3ProtocolError("header must contain exactly 76 bytes")
        _integer(self.height, "height")
        _certificate(self.cert_version)
        _integer(self.training_threshold, constant_text(FixedAlias('training.field.acceptance_bound')), 1)
        if self.training_threshold >= 1 << 256:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.acceptance_bound_bit_length')))


class KEXDialect:
    """Build neutral application requests without opening a connection.

    S3Session applies the explicit Stratum wire mapping in S3_methods.

    The default session sends uncompressed native S6 PlainProof bytes.
    Gzip requires both an explicit request and the S3's matching auth reply.
    Neither session type changes the native proof's binary structure.
    """

    def __init__(
        self,
        KEX_uname: str,
        worker: str,
        password: str = "x",
        agent: str = DEFAULT_AGENT,
        request_gzip: bool = False,
    ) -> None:
        self.KEX_uname = _text(KEX_uname, "KEX_uname", 256)
        self.worker = _text(worker, constant_text(FixedAlias('training.field.training_worker')), 128, allow_empty=True)
        self.password = _text(password, "password", 1024, allow_empty=True)
        self.agent = _text(agent, "agent", 128)
        if any(c.isspace() for c in self.KEX_uname + self.worker):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.coordinator_worker_identity_whitespace')))
        if type(request_gzip) is not bool:
            raise S3ProtocolError("request_gzip must be a boolean")
        self.request_gzip = request_gzip

    @property
    def login(self) -> str:
        # The endpoint expects a dot-separated account and device identity.
        return f"{self.KEX_uname}.{self.worker}" if self.worker else self.KEX_uname

    def auth_requests(self, start_id: int = 1) -> list[dict[str, Any]]:
        """Register the client and open a session without waiting for registration."""
        _integer(start_id, "request ID", 1)
        if self.request_gzip:
            return [{
                "jsonrpc": "2.0", "id": start_id, "method": methods.COMPUTE_SESSION,
                "params": {
                    s2(): self.login, "agent": self.agent,
                    "password": self.password, "type": "v2",
                },
            }]
        return [
            {
                "jsonrpc": "2.0", "id": start_id, "method": methods.COMPUTE_REGISTER,
                "params": [self.agent],
            },
            {
                "jsonrpc": "2.0", "id": start_id + 1, "method": methods.COMPUTE_SESSION,
                "params": [self.login, self.password],
            },
        ]

    @staticmethod
    def accept(response: Mapping[str, Any]) -> bool:
        """Only a literal successful result with no error confirms acceptance."""
        if not isinstance(response, Mapping) or "method" in response:
            return False
        return response.get("result") is True and response.get("error") is None

    def negotiate_gzip(self, auth_response: Mapping[str, Any]) -> bool:
        """Apply the primary protocol note's explicit, acknowledged opt-in."""
        return (
            self.request_gzip
            and self.accept(auth_response)
            and auth_response.get("type") == "v2"
        )

    def parse_job(
        self,
        params: Mapping[str, Any],
        configured_certificate_version: int | None = None,
    ) -> S3Job:
        if not isinstance(params, Mapping):
            raise S3ProtocolError("work.update params must be an object")
        if configured_certificate_version is not None:
            _certificate(configured_certificate_version)
        identifier_field = constant_text(FixedAlias('training.field.batch_identifier'))
        job_id = _text(params.get(identifier_field), identifier_field, 128)
        encoded_header = params.get("header")
        if (
            not isinstance(encoded_header, str)
            or len(encoded_header) != 152
            or _HEX.fullmatch(encoded_header) is None
        ):
            raise S3ProtocolError("header must be exactly 152 hexadecimal characters")
        encoded_training_threshold = params.get(constant_text(FixedAlias('training.field.acceptance_bound')))
        if not isinstance(encoded_training_threshold, str):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.acceptance_bound_hex_format')))
        digits = encoded_training_threshold[2:] if encoded_training_threshold.startswith("0x") else encoded_training_threshold
        if not 1 <= len(digits) <= 64 or _HEX.fullmatch(digits) is None:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.acceptance_bound_positive_hex_range')))
        # The received training threshold is unadjusted. The official helper
        # applies rank and batch scaling independently of the job ID.
        training_threshold = int(digits, 16)
        version = params.get(constant_text(FixedAlias('training.field.attestation_revision')), configured_certificate_version)
        if version is None:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_attestation_revision_required')))
        return S3Job(
            job_id=job_id,
            header_bytes=bytes.fromhex(encoded_header),
            height=_integer(params.get("height"), "height"),
            cert_version=_certificate(version),
            training_threshold=training_threshold,
            encoded_training_threshold=encoded_training_threshold,
        )

    def submission_request(
        self,
        job: S3Job,
        plain_proof_b64: str,
        request_id: int,
        negotiated_gzip: bool = False,
    ) -> dict[str, Any]:
        """Frame official proof bytes without rewriting the native serialization."""
        if not isinstance(job, S3Job):
            raise S3ProtocolError("submission requires a S3Job")
        _integer(request_id, "request ID", 1)
        if type(negotiated_gzip) is not bool or (negotiated_gzip and not self.request_gzip):
            raise S3ProtocolError("gzip submissions require an explicitly requested gzip session")
        raw = _proof_bytes(plain_proof_b64)
        encoded = plain_proof_b64
        if negotiated_gzip:
            encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
        return {
            "jsonrpc": "2.0", "id": request_id, "method": methods.WORK_COMPLETE,
            "params": {
                constant_text(FixedAlias('training.field.training_worker')): self.login,
                constant_text(FixedAlias('training.field.batch_identifier')): job.job_id,
                constant_text(FixedAlias('training.field.dense_attestation')): encoded,
            },
        }
