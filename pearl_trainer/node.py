"""Direct JSON-RPC connection to a PEARL node and official block helpers."""

from __future__ import annotations

import base64
import json
import math
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class NodeError(RuntimeError):
    """A node connection, protocol, or submission failure."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Credentials belong to the selected node; never forward them elsewhere.
        return None


class NodeClient:
    """Talk to pearld directly; HTTPS always verifies the server certificate.

    HTTP is supported for a node explicitly configured without TLS. A remote
    node must be selected explicitly; the default endpoint is loopback HTTPS.
    """

    MAX_RESPONSE_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        url: str = "https://127.0.0.1:44107",
        username: str = "",
        password: str = "",
        ca_file: str | None = None,
        timeout: float = 30,
    ) -> None:
        if not isinstance(url, str):
            raise NodeError("RPC URL must be a string")
        try:
            parsed = urllib.parse.urlsplit(url)
            valid = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.fragment
                and not any(character.isspace() for character in url)
            )
            _ = parsed.port  # Validate the port even though urllib parses it later.
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise NodeError("RPC URL must use http or https and contain no credentials or fragment")
        if not isinstance(username, str) or not isinstance(password, str):
            raise NodeError("RPC username and password must be strings")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise NodeError("RPC timeout must be a positive finite number")
        if ":" in username:
            raise NodeError("RPC username cannot contain a colon")

        self.url = url
        self.timeout = timeout
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        self._authorization = f"Basic {token}"
        self._secrets = (self._authorization, token, password, username)
        self._request_id = 0
        self._id_lock = threading.Lock()
        self._context = None
        if parsed.scheme == "https":
            try:
                self._context = ssl.create_default_context(cafile=ca_file)
            except (OSError, ssl.SSLError):
                raise NodeError("Could not load RPC certificate authority file") from None
        elif ca_file is not None:
            raise NodeError("A certificate authority file requires an https RPC URL")
        self._opener = urllib.request.build_opener(
            _RejectRedirects(), urllib.request.HTTPSHandler(context=self._context)
        )

    def _safe_message(self, value: Any) -> str:
        message = str(value)
        for secret in self._secrets:
            if secret:
                message = message.replace(secret, "[redacted]")
        return message[:500]

    def rpc(self, method: str, params: list[Any] | None = None) -> Any:
        """Make one authenticated JSON-RPC request, raising on any error."""
        if not isinstance(method, str) or not method:
            raise NodeError("RPC method must be a nonempty string")
        if params is None:
            params = []
        if not isinstance(params, list):
            raise NodeError("RPC params must be a list")
        with self._id_lock:
            self._request_id += 1
            request_id = self._request_id
        try:
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise NodeError("RPC parameters cannot be encoded as JSON") from None

        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": self._authorization,
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(self.MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise NodeError(f"RPC HTTP error {exc.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise NodeError(f"RPC connection failed: {self._safe_message(exc)}") from None
        if len(raw) > self.MAX_RESPONSE_BYTES:
            raise NodeError("RPC response exceeds the size limit")
        try:
            reply = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise NodeError("RPC returned invalid JSON") from None
        if not isinstance(reply, dict):
            raise NodeError("RPC response must be an object")
        if reply.get("id") != request_id or isinstance(reply.get("id"), bool):
            raise NodeError("RPC response ID does not match the request")
        if reply.get("error") is not None:
            raise NodeError(f"Node RPC error: {self._safe_message(reply['error'])}")
        if "result" not in reply:
            raise NodeError("RPC response is missing its result")
        return reply["result"]

    def raw_template(self) -> dict[str, Any]:
        """Fetch a SegWit block template using PEARL's official request format."""
        result = self.rpc(
            "getblocktemplate",
            [{"capabilities": ["coinbasevalue", "workid", "coinbase/append"], "rules": ["segwit"]}],
        )
        if not isinstance(result, dict):
            raise NodeError("Node block template must be an object")
        return result

    def get_template(self, address: str) -> Any:
        """Construct a template using the pinned official PEARL helper files."""
        from pearl_gateway.comm.dataclasses import BlockTemplate
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        raw = self.raw_template()
        try:
            # The official helper stores an already serialized coinbase script
            # in Script.script[0]; TxInput serializes that raw field specially.
            return BlockTemplate.from_get_block_template(
                GetBlockTemplateResponse.model_validate(raw), mining_address=address
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise NodeError(f"Invalid block template: {self._safe_message(exc)}") from None

    def best_hash(self) -> str:
        """Return the current chain tip as a validated display-order hash."""
        result = self.rpc("getbestblockhash")
        if not isinstance(result, str) or len(result) != 64:
            raise NodeError("Node returned an invalid best block hash")
        try:
            bytes.fromhex(result)
        except ValueError:
            raise NodeError("Node returned an invalid best block hash") from None
        return result

    def submit(self, block_bytes: bytes) -> None:
        """Submit a serialized block; only a null node result means acceptance."""
        if not isinstance(block_bytes, bytes) or not block_bytes:
            raise NodeError("A nonempty serialized block is required")
        result = self.rpc("submitblock", [block_bytes.hex()])
        if result is not None:
            raise NodeError(f"Node rejected block: {self._safe_message(result)}")


def build_block(proof: Any, template: Any) -> bytes:
    """Certify and serialize a solution using the official version-aware prover."""
    from pearl_gateway.proof_generator import ProofGenerator

    return ProofGenerator.generate_block(proof, template, debug_mode=True).serialize()
