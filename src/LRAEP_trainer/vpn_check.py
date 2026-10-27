"""Check V2Ray links using tunneled pool connections and an HTTPS request.

The reported ping is a TCP/TLS connection measurement, not an ICMP echo.
Reports identify input line numbers and deliberately omit share-link secrets.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass
import ipaddress
import json
import math
from pathlib import Path
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from . import network
from .vpn import ManagedVPN, VPNError, parse_link


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


@dataclass
class LinkResult:
    line: int
    status: str = "INVALID"
    usable: bool = False
    pool_proof_level: str = "none"
    pool_attempts: int = 0
    pool_successes: int = 0
    pool_latency_ms: float | None = None
    pool_loss_percent: float | None = None
    pool_error: str | None = None
    https_status: str = "not_run"
    https_latency_ms: float | None = None
    exit_ip: str | None = None
    https_error: str | None = None
    error: str | None = None


def _error_kind(error: Exception) -> str:
    """Return an actionable category without copying a third-party message."""
    if isinstance(error, TimeoutError):
        return "connection timed out"
    if isinstance(error, ssl.SSLError):
        return "TLS verification or handshake failed"
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP status {error.code}"
    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        if isinstance(reason, (TimeoutError, ssl.SSLError)):
            return _error_kind(reason)
    return "connection failed"


def _probe_pool(pool: dict, proxy, timeout: float) -> float:
    started = time.monotonic()
    connection = None
    try:
        connection = network.create_connection((pool["host"], pool["port"]), timeout,
                                               proxy=proxy)
        if pool.get("tls", False):
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError()
            connection.settimeout(remaining)
            context = ssl.create_default_context(cafile=pool.get("ca_file"))
            connection = context.wrap_socket(connection, server_hostname=pool["host"])
        return (time.monotonic() - started) * 1000
    finally:
        if connection is not None:
            with suppress(OSError):
                connection.close()


def _probe_https(url: str, proxy, timeout: float) -> tuple[float, str | None]:
    # The URL is configurable, so validate again for direct API callers.
    try:
        parsed = urllib.parse.urlsplit(url)
        valid = (parsed.scheme == "https" and parsed.hostname is not None
                 and parsed.username is None and parsed.password is None
                 and not parsed.fragment and not any(c.isspace() for c in url))
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            valid = False
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("HTTPS test URL is invalid")
    started = time.monotonic()
    opener = network.http_opener(proxy, context=ssl.create_default_context(),
                                handlers=(_RejectRedirects(),))
    request = urllib.request.Request(url, headers={"User-Agent": "LRAEP-vpn-check/0.1",
                                                 "Accept": "text/plain"})
    with opener.open(request, timeout=timeout) as response:
        status = response.getcode()
        if status is not None and not 200 <= status < 300:
            raise urllib.error.HTTPError(url, status, "HTTP test failed", {}, None)
        body = response.read(4097)
    elapsed = (time.monotonic() - started) * 1000
    # A normal HTTPS page is useful as a reachability check too; an IP is
    # shown only when the complete small response is literally an IP address.
    exit_ip = None
    if len(body) <= 4096:
        try:
            exit_ip = str(ipaddress.ip_address(body.decode("ascii").strip()))
        except (ValueError, UnicodeError):
            pass
    return elapsed, exit_ip


def _check_link(line: int, link: str, configuration: dict, timeout: float,
                attempts: int) -> LinkResult:
    result = LinkResult(line=line)
    try:
        parse_link(link)
    except (VPNError, ValueError, TypeError):
        result.error = "invalid or unsupported link"
        return result
    result.status = "CORE_FAIL"
    try:
        with ManagedVPN(configuration.get("vpn", {}), link=link) as proxy:
            latencies = []
            for _ in range(attempts):
                result.pool_attempts += 1
                try:
                    latencies.append(_probe_pool(configuration["pool"], proxy, timeout))
                    result.pool_successes += 1
                except (OSError, ValueError, RuntimeError) as error:
                    result.pool_error = _error_kind(error)
            result.pool_loss_percent = round(
                100 * (attempts - result.pool_successes) / attempts, 1)
            if latencies:
                result.pool_latency_ms = round(sum(latencies) / len(latencies), 2)
                result.pool_proof_level = ("tls_verified" if configuration["pool"].get("tls", False)
                                           else "connect_only")
                result.status = "POOL_OK" if result.pool_proof_level == "tls_verified" else "UNVERIFIED"
            else:
                result.status = "POOL_FAIL"
            url = configuration.get("vpn", {}).get("check_url", "https://api.ipify.org")
            if url is None:
                result.https_status = "disabled"
            else:
                try:
                    latency, result.exit_ip = _probe_https(url, proxy, timeout)
                    result.https_latency_ms = round(latency, 2)
                    result.https_status = "ok"
                except (OSError, ValueError, RuntimeError) as error:
                    result.https_status = "failed"
                    result.https_error = _error_kind(error)
            result.usable = result.pool_successes > 0 and (
                result.pool_proof_level == "tls_verified" or result.https_status == "ok")
            if result.usable and result.pool_proof_level == "connect_only":
                result.status = "CONNECT_OK"
            elif result.pool_successes and not result.usable:
                result.error = "TCP CONNECT alone is unverified; a successful HTTPS check is required"
    except (VPNError, OSError, ValueError, RuntimeError):
        # Neither core diagnostics nor share-link strings are suitable for a
        # result file: both may contain credentials from the supplied link.
        result.status = "CORE_FAIL"
        result.usable = False
        result.error = "proxy core failed to start or stopped unexpectedly"
    return result


def run_vpn_check(configuration: dict, links_file: str | Path = "links.txt", *,
                  timeout: float = 5.0, attempts: int = 3,
                  output: str | Path | None = None) -> int:
    """Test every noncomment link; return zero if any link is usable.

    A verified pool TLS handshake proves the tunnel works. A plain SOCKS
    CONNECT may be acknowledged before dialing, so plain TCP also requires a
    successful HTTPS probe to count as usable. All probes use the endpoint
    supplied by the managed core, never a direct fallback.
    """
    if type(attempts) is not int or not 1 <= attempts <= 20:
        raise ValueError("VPN test attempts must be an integer from 1 to 20")
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 0 < timeout <= 60):
        raise ValueError("VPN test timeout must be positive and at most 60 seconds")
    try:
        lines = Path(links_file).expanduser().read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError, TypeError, ValueError):
        print("Could not read the VPN links file.", file=sys.stderr)
        return 1
    links = [(number, value.strip()) for number, value in enumerate(lines, 1)
             if value.strip() and not value.lstrip().startswith("#")]
    if not links:
        print("The VPN links file contains no configurations.", file=sys.stderr)
        return 1
    print("Line  Pool        Avg ms   Loss      HTTPS       Exit IP")
    results = []
    for number, link in links:
        result = _check_link(number, link, configuration, float(timeout), attempts)
        results.append(result)
        latency = f"{result.pool_latency_ms:.1f}" if result.pool_latency_ms is not None else "-"
        loss = f"{result.pool_loss_percent:.1f}%" if result.pool_loss_percent is not None else "-"
        http_status = {"not_run": "-", "disabled": "disabled", "ok": "OK",
                       "failed": "FAIL"}[result.https_status]
        print(f"{number:4}  {result.status:<10}  {latency:>7}  {loss:>6}    "
              f"{http_status:<8}  {result.exit_ip or '-'}", flush=True)
        details = []
        if result.error:
            details.append(result.error)
        if result.pool_error:
            details.append(f"pool: {result.pool_error}")
        if result.https_error:
            details.append(f"HTTPS: {result.https_error}")
        if details:
            print("      " + "; ".join(details), flush=True)
    working = sum(result.usable for result in results)
    print(f"{working}/{len(results)} configurations are usable. "
          "Ping measures tunneled TCP/TLS connection time. "
          "Plain TCP CONNECT requires a successful HTTPS check and does not verify the pool destination.")
    if output is not None:
        report = {"working": working, "total": len(results),
                  "measurement": "tunneled TCP/TLS connection time",
                  "results": [asdict(result) for result in results]}
        try:
            path = Path(output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError, TypeError):
            print("Could not write the VPN test report.", file=sys.stderr)
            return 1
    return 0 if working else 1
