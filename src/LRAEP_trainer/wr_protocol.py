"""S7 protocol facts recovered independently in wr-checker.

This adapter supports the observed WildRig 0.52.3 exchange and its version-3
work. The opaque notify tail is preserved, never interpreted as instructions.
No external S5 executable or native S5 implementation is required.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any

from .fixed_constants import FixedAlias, constant_ascii_matches, constant_buffer, constant_bytes, constant_text
from .S6_requests import build_submission_request
from .S3_protocol import DEFAULT_AGENT, S3Job, S3ProtocolError, _text
from .wire_constants import WR_CLIENT_AGENT


_HEX = re.compile(r"[0-9a-fA-F]+\Z")


_HEADER_MASK = FixedAlias('training.format.batch_header_mask')


def header_mask():
    return constant_bytes(_HEADER_MASK)


_WORK_PREFIX_MASK = FixedAlias('training.format.batch_update_prefix_mask')


def work_prefix_mask():
    return constant_bytes(_WORK_PREFIX_MASK)


SHAPE_REJECTED = "shape not allowed"


class UnknownWRRequest(S3ProtocolError):
    """An otherwise framed server command has no implemented meaning."""


def _hex_bytes(value: Any, length: int, label: str) -> bytes:
    if (not isinstance(value, str) or len(value) != length * 2
            or _HEX.fullmatch(value) is None):
        prefix = constant_text(FixedAlias('training.label.legacy_batch_protocol_prefix'))
        raise S3ProtocolError(f"{prefix}{label} must contain exactly {length} hexadecimal bytes")
    return bytes.fromhex(value)


def decode_update_payload(value: str) -> bytes:
    """Recover the observed 76-byte block header from its 200-byte work blob."""
    payload = _hex_bytes(value, 200, constant_text(FixedAlias('training.label.batch_update_payload')))
    with constant_buffer(_WORK_PREFIX_MASK) as prefix, constant_buffer(_HEADER_MASK) as mask:
        header = bytes(byte ^ prefix[index] ^ mask[index % 32]
                       for index, byte in enumerate(payload[:76]))
    # A valid training payload must also carry a valid compact training threshold.
    nbits = int.from_bytes(header[72:76], "little")
    exponent, mantissa = nbits >> 24, nbits & 0x007FFFFF
    training_threshold = (mantissa >> (8 * (3 - exponent)) if exponent <= 3
                          else mantissa << (8 * (exponent - 3)))
    if nbits & 0x00800000 or not 0 < training_threshold < 1 << 256:
        raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_header_compact_bound_invalid')))
    return header


@dataclass(frozen=True)
class WRJob(S3Job):
    """Version-3 training work with its server-issued device seed.

    The wire has no height. ``height=0`` is a compatibility placeholder for
    the common training interface, never a height inferred from generation.
    """

    update_payload: bytes = b""
    device_seed: bytes | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cert_version != 3 or self.height != 0:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_generation_unsupported')))
        if not isinstance(self.update_payload, bytes) or len(self.update_payload) != 200:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_update_payload_length')))
        if self.device_seed is not None and (
            not isinstance(self.device_seed, bytes) or len(self.device_seed) != 8
        ):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_device_seed_length')))


@dataclass(frozen=True)
class WRSubmission:
    """Compact reconstruction witness, rather than a KEX base64 proof."""

    local_seed: bytes
    device_seed: bytes
    m: int
    n: int
    k: int
    rank: int
    tile_h: int
    tile_w: int
    a_rows: tuple[int, ...]
    b_cols: tuple[int, ...]
    sb: int = 1

    def request(self, request_id: int) -> dict[str, Any]:
        try:
            return build_submission_request(
                request_id, local_seed=self.local_seed, device_seed=self.device_seed,
                m=self.m, n=self.n, k=self.k, rank=self.rank, tile_h=self.tile_h,
                tile_w=self.tile_w, sb=self.sb, a_rows=self.a_rows, b_cols=self.b_cols,
            )
        except (ValueError, TypeError) as error:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_submission_fields_invalid'))) from error


class WRDialect:
    """Build client requests and strictly classify the observed server schema."""

    def __init__(self, WR_S2: str, worker: str, agent: str = WR_CLIENT_AGENT) -> None:
        self.WR_S2 = _text(WR_S2, "WR_S2", 256)
        self.worker = _text(worker, constant_text(FixedAlias('training.field.training_worker')), 128, allow_empty=True)
        # Existing WR configs can contain the previously saved default agent.
        self.agent = WR_CLIENT_AGENT if agent == DEFAULT_AGENT else _text(agent, "agent", 128)
        if any(character.isspace() for character in WR_S2 + worker):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_account_identity_whitespace')))

    def auth_requests(self, start_id: int = 0) -> list[dict[str, Any]]:
        if type(start_id) is not int or not 0 <= start_id < 1 << 64:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_request_identifier_range')))
        return [{
            "id": start_id,
            "method": constant_text(FixedAlias('training.message.open_session')),
            "params": [self.WR_S2, self.worker, self.agent],
        }]

    @staticmethod
    def report_request(gpu_name: str, training_rate: float, request_id: int) -> dict[str, Any]:
        """Report one GPU's measured work rate in raw hash units per second."""
        return WRDialect.devices_report_request(((gpu_name, training_rate),), request_id)

    @staticmethod
    def devices_report_request(devices, request_id: int) -> dict[str, Any]:
        if type(request_id) is not int or not 0 <= request_id < 1 << 64:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_request_identifier_range')))
        entries = [WRDialect._device_report(name, rate) for name, rate in devices]
        if not entries:
            raise S3ProtocolError("GPU report must contain at least one device")
        return {
            "id": request_id,
            "method": constant_text(FixedAlias('training.message.accelerator_metrics')),
            "params": entries,
        }

    @staticmethod
    def _device_report(gpu_name, training_rate):
        name = _text(gpu_name, "gpu_name", 256)
        if not name.strip() or not name.isprintable():
            raise S3ProtocolError("gpu_name must contain printable GPU identification")
        if isinstance(training_rate, bool) or not isinstance(training_rate, (int, float)):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.accelerator_throughput_range')))
        try:
            rate = float(training_rate)
        except OverflowError:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.accelerator_throughput_range'))) from None
        if not math.isfinite(rate) or rate < 0:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.accelerator_throughput_range')))
        return {constant_text('training_rate'): rate, "name": name}

    @staticmethod
    def accept(response: Mapping[str, Any]) -> bool:
        return (isinstance(response, Mapping) and "method" not in response
                and type(response.get("status")) is int and response["status"] == 1
                and response.get("result") is True and response.get("error") is None)

    @staticmethod
    def negotiate_gzip(auth_response: Mapping[str, Any]) -> bool:
        return False

    @staticmethod
    def classify(response: Mapping[str, Any]) -> str:
        """Classify replies, server seed updates and batch updates.

        Rejection replies can carry a boolean or a text reason. The session
        correlates replies by request ID before authenticating or crediting work.
        """
        if isinstance(response, Mapping) and isinstance(response.get("method"), str):
            raise UnknownWRRequest(constant_text(FixedAlias('training.validation.batch_server_method_unknown')))
        if not isinstance(response, Mapping) or set(response) - {"id", "status", "result", "error"}:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_server_fields_unknown')))
        request_id, status = response.get("id"), response.get("status")
        if (type(request_id) is not int or not -1 <= request_id < 1 << 64
                or type(status) is not int or status not in (0, 1)):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_response_envelope_unknown')))
        result = response.get("result")
        if request_id >= 0 and (type(result) is bool or (status == 0 and isinstance(result, str) and result)):
            return "reply"
        if (request_id != -1 or status != 1 or response.get("error") is not None
                or not isinstance(result, list) or not result):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_server_message_shape_unknown')))
        if constant_ascii_matches('server_seed', result[0]):
            if len(result) != 2:
                raise S3ProtocolError(constant_text(FixedAlias('training.validation.coordinator_seed_update_schema')))
            _hex_bytes(result[1], 8, constant_text('server_seed'))
            return result[0]
        if result[0] == constant_text(FixedAlias('training.message.batch_update')):
            # Full job validation is performed by parse_job, including version.
            if len(result) != 5 or not all(isinstance(field, str) for field in result):
                raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_update_schema')))
            return constant_text(FixedAlias('training.message.batch_update'))
        if isinstance(result[0], str):
            raise UnknownWRRequest(constant_text(FixedAlias('training.validation.batch_server_notification_unknown')))
        raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_notification_name_malformed')))

    @staticmethod
    def parse_server_seed(result: list) -> bytes:
        if not isinstance(result, list) or len(result) != 2 or not constant_ascii_matches('server_seed', result[0]):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.coordinator_seed_update_schema')))
        return _hex_bytes(result[1], 8, constant_text('server_seed'))

    @staticmethod
    def parse_job(result: list, configured_certificate_version: int | None = None,
                  device_seed: bytes | None = None) -> WRJob:
        if (not isinstance(result, list) or len(result) != 5
                or result[0] != constant_text(FixedAlias('training.message.batch_update'))
                or not all(isinstance(field, str) for field in result)):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_update_schema')))
        if result[4] != "3":
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_generation_unsupported')))
        if configured_certificate_version is not None and (
            type(configured_certificate_version) is not int or configured_certificate_version != 3
        ):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_generation_attestation_revision')))
        payload = _hex_bytes(result[1], 200, constant_text(FixedAlias('training.label.batch_update_payload')))
        training_threshold_bytes = _hex_bytes(result[2], 32, constant_text(FixedAlias('training.label.batch_acceptance_bound')))
        return WRJob(
            job_id=_text(result[3], constant_text(FixedAlias('training.field.batch_identifier')), 128),
            header_bytes=decode_update_payload(result[1]),
            # S7 transmits the uint256 in little-endian byte order,
            # unlike KEX's display-order hexadecimal training threshold.
            height=0, cert_version=3, training_threshold=int.from_bytes(training_threshold_bytes, "little"),
            encoded_training_threshold=result[2], update_payload=payload, device_seed=device_seed,
        )

    @staticmethod
    def submission_request(job: WRJob, submission: WRSubmission, request_id: int,
                       negotiated_gzip: bool = False) -> dict[str, Any]:
        if not isinstance(job, WRJob) or not isinstance(submission, WRSubmission):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_and_compact_attestation_required')))
        if negotiated_gzip:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_submission_compression_unsupported')))
        if job.device_seed is None or submission.device_seed != job.device_seed:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_submission_device_seed_mismatch')))
        return submission.request(request_id)
