"""S7 protocol facts recovered independently in wr-checker.

This adapter supports the observed WR 0.51.8 exchange and its version-3
work. The opaque notify tail is preserved, never interpreted as instructions.
No external S5 executable or native S5 implementation is required.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any

from .fixed_constants import constant_buffer, constant_bytes, constant_text
from .S6_requests import build_submit_request
from .S3_protocol import DEFAULT_AGENT, S3Job, S3ProtocolError, _text


_HEX = re.compile(r"[0-9a-fA-F]+\Z")


_HEADER_MASK = b'I*8O`wh#PAk==IbHLDoHRF5Nlu*n|gYxJlA+C{$8'


def header_mask():
    return constant_bytes(_HEADER_MASK)


_WORK_PREFIX_MASK = b'>SuG>Sk;?fQ2qa${>xwQQn$)?cMaTnN;>NC%{4e(3{5T~Xp;raRu;cSVK7xBS8WP$<QFjXHwN3@4Md5~XtB)j0a5CAj<Y~R'


def work_prefix_mask():
    return constant_bytes(_WORK_PREFIX_MASK)


class UnknownWRRequest(S3ProtocolError):
    """An otherwise framed server command has no implemented meaning."""


def _hex_bytes(value: Any, length: int, label: str) -> bytes:
    if (not isinstance(value, str) or len(value) != length * 2
            or _HEX.fullmatch(value) is None):
        raise S3ProtocolError(f"{constant_text(b'^pJ-')}{label} must contain exactly {length} hexadecimal bytes")
    return bytes.fromhex(value)


def decode_notify_payload(value: str) -> bytes:
    """Recover the observed 76-byte block header from its 200-byte work blob."""
    payload = _hex_bytes(value, 200, constant_text(b'$;i{r!QO}0#@@@w#=`'))
    with constant_buffer(_WORK_PREFIX_MASK) as prefix, constant_buffer(_HEADER_MASK) as mask:
        header = bytes(byte ^ prefix[index] ^ mask[index % 32]
                       for index, byte in enumerate(payload[:76]))
    # A syntactically valid work blob must also carry a valid compact target.
    nbits = int.from_bytes(header[72:76], "little")
    exponent, mantissa = nbits >> 24, nbits & 0x007FFFFF
    target = (mantissa >> (8 * (3 - exponent)) if exponent <= 3
              else mantissa << (8 * (exponent - 3)))
    if nbits & 0x00800000 or not 0 < target < 1 << 256:
        raise S3ProtocolError(constant_text(b'^pJ<c!o<kK!o!Ep!p6hG)rZf<)Q85&htA2-#>>vbhs4Or*T%%thttN@z{1l'))
    return header


@dataclass(frozen=True)
class WRJob(S3Job):
    """Version-3 work, with S3 extranonce bound to this job snapshot.

    The wire has no height. ``height=0`` is a compatibility placeholder for
    the common training interface, never a height inferred from generation.
    """

    notify_payload: bytes = b""
    worker_nonce: bytes | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cert_version != 3 or self.height != 0:
            raise S3ProtocolError(constant_text(b'=*i38hsed$!qw5j!-w>chrq(g!qvvp&dAA!l!wmLht$&7*T~h=!ov'))
        if not isinstance(self.notify_payload, bytes) or len(self.notify_payload) != 200:
            raise S3ProtocolError(constant_text(b'^pJ<i$kWcj-iO!5-pk0w!-vY!)YFH=$jQ^j&dG<BmzRgd-qXU='))
        if self.worker_nonce is not None and (
            not isinstance(self.worker_nonce, bytes) or len(self.worker_nonce) != 8
        ):
            raise S3ProtocolError(constant_text(b'^pJ<p$koik)rZN*$;85k%F@)+hs4Oq)5gxphr-Ul&(nv+-qXU='))


@dataclass(frozen=True)
class WRShare:
    """Compact reconstruction witness, rather than a KEX base64 proof."""

    S5_nonce: bytes
    worker_nonce: bytes
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
            return build_submit_request(
                request_id, S5_nonce=self.S5_nonce, worker_nonce=self.worker_nonce,
                m=self.m, n=self.n, k=self.k, rank=self.rank, tile_h=self.tile_h,
                tile_w=self.tile_w, sb=self.sb, a_rows=self.a_rows, b_cols=self.b_cols,
            )
        except (ValueError, TypeError) as error:
            raise S3ProtocolError(constant_text(b'?#a={%g)1x^pJ<t&&Jilhr!Om%fr+')) from error


class WRDialect:
    """Build client requests and strictly classify the observed server schema."""

    def __init__(self, WR_S2: str, worker: str, agent: str = DEFAULT_AGENT) -> None:
        self.WR_S2 = _text(WR_S2, "WR_S2", 256)
        self.worker = _text(worker, constant_text(b'(8$%y!qo'), 128, allow_empty=True)
        self.agent = _text(agent, "agent", 128)
        if any(character.isspace() for character in WR_S2 + worker):
            raise S3ProtocolError(constant_text(b'@b&ogm50X3!-vqw)y%@xhsx5_(}%>!$<xNp$%o0vhtSW?)56r(#>Bz'))

    def auth_requests(self, start_id: int = 1) -> list[dict[str, Any]]:
        if type(start_id) is not int or not 0 <= start_id < 1 << 64:
            raise S3ProtocolError(constant_text(b'^pJ<u!q(El)YFIV<A=)9)YFH>!iUDmhtkQ^&cMmS!-tWRip9>;htA2<!ob4S'))
        return [{"id": start_id, "method": constant_text(b'%gDga$p'), "params": [self.WR_S2, self.worker, self.agent]}]

    @staticmethod
    def report_request(gpu_name: str, hashrate: float, request_id: int) -> dict[str, Any]:
        """Report one GPU's measured work rate in raw hash units per second."""
        if type(request_id) is not int or not 0 <= request_id < 1 << 64:
            raise S3ProtocolError(constant_text(b'^pJ<u!q(El)YFIV<A=)9)YFH>!iUDmhtkQ^&cMmS!-tWRip9>;htA2<!ob4S'))
        name = _text(gpu_name, "gpu_name", 256)
        if not name.strip() or not name.isprintable():
            raise S3ProtocolError("gpu_name must contain printable GPU identification")
        if isinstance(hashrate, bool) or not isinstance(hashrate, (int, float)):
            raise S3ProtocolError(constant_text(b'&&Jfx)yC7phsx5_(}%^vhsKA=$jQmVz{b<g(ZYwp&dJWx!iUMy%EiLfhtA1|?}*d'))
        try:
            rate = float(hashrate)
        except OverflowError:
            raise S3ProtocolError(constant_text(b'&&Jfx)yC7phsx5_(}%^vhsKA=$jQmVz{b<g(ZYwp&dJWx!iUMy%EiLfhtA1|?}*d')) from None
        if not math.isfinite(rate) or rate < 0:
            raise S3ProtocolError(constant_text(b'&&Jfx)yC7phsx5_(}%^vhsKA=$jQmVz{b<g(ZYwp&dJWx!iUMy%EiLfhtA1|?}*d'))
        return {"id": request_id, "method": constant_text(b')xy`v)zkRS$-&3'), "params": [{constant_text(b'&&Jfx)yC7p'): rate, "name": name}]}

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
        """Return reply/extranonce/notify, raising on unknown server requests.

        Boolean replies may reject a correlated request. Their IDs are checked
        by the session; uncorrelated replies never authenticate or credit shares.
        """
        if isinstance(response, Mapping) and isinstance(response.get("method"), str):
            raise UnknownWRRequest(constant_text(b'^2yA}$k54$^pJ<t!qw5j)rZQ$)6dAm'))
        if not isinstance(response, Mapping) or set(response) - {"id", "status", "result", "error"}:
            raise S3ProtocolError(constant_text(b'^2yA}$k54$^pJ<t!qw5j)rZQ$)YQhn!iT}m!pp<dhsf23%EHsn$in'))
        request_id, status = response.get("id"), response.get("status")
        if (type(request_id) is not int or not -1 <= request_id < 1 << 64
                or type(status) is not int or status not in (0, 1)):
            raise S3ProtocolError(constant_text(b'^2yA}$k54$^pJ<u!qnHu$<)G!!pYIX%gEQl'))
        result = response.get("result")
        if request_id >= 0 and type(result) is bool:
            return "reply"
        if (request_id != -1 or status != 1 or response.get("error") is not None
                or not isinstance(result, list) or not result):
            raise S3ProtocolError(constant_text(b'^2yA}$k54$^pJ<t!qw5j)rZx>*3!b%(}&2_ht<N=*T~7#!iUt)#@E6'))
        if result[0] == constant_text(b'!r#-?#>vRZ#KH'):
            if len(result) != 2:
                raise S3ProtocolError(constant_text(b'^2yZF*Vo9^)561t^pJ<b-_zB`$;ipX!iUtv&%(;a'))
            _hex_bytes(result[1], 8, constant_text(b'!r#-?#>vRZ#KH'))
            return constant_text(b'!r#-?#>vRZ#KH')
        if result[0] == constant_text(b'$;i{r!QK'):
            # Full job validation is performed by parse_job, including version.
            if len(result) != 5 or not all(isinstance(field, str) for field in result):
                raise S3ProtocolError(constant_text(b'^2yZF*Vo9^)561t^pJ<i$kWcj-iOr0&%(;a'))
            return constant_text(b'$;i{r!QK')
        if isinstance(result[0], str):
            raise UnknownWRRequest(constant_text(b'^2yA}$k54$^pJ<t!qw5j)rZN*)6T)p#KzOk$jJ'))
        raise S3ProtocolError(constant_text(b'>c-2#$koci!-w>chsns(&cV*a#?#Ko$%o0t%EA'))

    @staticmethod
    def parse_extranonce(result: list) -> bytes:
        if not isinstance(result, list) or len(result) != 2 or result[0] != constant_text(b'!r#-?#>vRZ#KH'):
            raise S3ProtocolError(constant_text(b'^2yZF*Vo9^)561t^pJ<b-_zB`$;ipX!iUtv&%(;a'))
        return _hex_bytes(result[1], 8, constant_text(b'!r#-?#>vRZ#KH'))

    @staticmethod
    def parse_job(result: list, configured_certificate_version: int | None = None,
                  worker_nonce: bytes | None = None) -> WRJob:
        if (not isinstance(result, list) or len(result) != 5 or result[0] != constant_text(b'$;i{r!QK')
                or not all(isinstance(field, str) for field in result)):
            raise S3ProtocolError(constant_text(b'^2yZF*Vo9^)561t^pJ<i$kWcj-iOr0&%(;a'))
        if result[4] != "3":
            raise S3ProtocolError(constant_text(b'=*i38hsed$!qw5j!-w>chrq(g!qvvp&dAA!l!wmLht$&7*T~h=!ov'))
        if configured_certificate_version is not None and (
            type(configured_certificate_version) is not int or configured_certificate_version != 3
        ):
            raise S3ProtocolError(constant_text(b'^pJ<Z!pXwb#?#Ko$%mAO)xy@&&eg)yhs46w)6T)p#KzOYhta~-)XvDshm-'))
        payload = _hex_bytes(result[1], 200, constant_text(b'$;i{r!QO}0#@@@w#=`'))
        target_bytes = _hex_bytes(result[2], 32, constant_text(b')X&D%!iUqw)xg5j'))
        return WRJob(
            job_id=_text(result[3], constant_text(b'&B(?0&cg'), 128), header_bytes=decode_notify_payload(result[1]),
            # S7 transmits the uint256 in little-endian byte order,
            # unlike KEX's display-order hexadecimal target.
            height=0, cert_version=3, target=int.from_bytes(target_bytes, "little"),
            raw_target=result[2], notify_payload=payload, worker_nonce=worker_nonce,
        )

    @staticmethod
    def submit_request(job: WRJob, share: WRShare, request_id: int,
                       negotiated_gzip: bool = False) -> dict[str, Any]:
        if not isinstance(job, WRJob) or not isinstance(share, WRShare):
            raise S3ProtocolError(constant_text(b'^pJ<t(#6Wo)YQ(%$%oa#*3!<^!qkV()6|E}$i;`o$-{@l$jaBo#M6h=&&Jil'))
        if negotiated_gzip:
            raise S3ProtocolError(constant_text(b'^pJ<c$imcz$;i`()Y8}2$ko$_?Bf52z}?Q*hu77}$iauy(#6Wo)YQ(%$<z'))
        if job.worker_nonce is None or share.worker_nonce != job.worker_nonce:
            raise S3ProtocolError(constant_text(b'^pJ<t&&JilhtSB?%)-@&$;ipX!iU4i!qkV!$kT_)#?!>lhtAW~ht0^v'))
        return share.request(request_id)
