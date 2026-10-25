"""Strict JSON configuration for the custom PEARL GPU trainer.

This module only validates settings. Pool settings do not imply a particular
wire protocol, and a documentation path never changes executable behavior.
Official source revision checks remain in ``load_official_helpers`` when the
trainer uses those helpers.
"""

import ipaddress
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit


class ConfigError(RuntimeError):
    """The trainer configuration could not be read or validated."""


_TOP_LEVEL = {"version", "mode", "official_source", "device", "training", "pool", "node"}
_TRAINING_DEFAULTS = {
    "m": 256, "n": 256, "k": 2048, "tile_h": 8, "tile_w": 8,
    "max_batches": 0, "max_seconds": 0, "output_dir": "artifacts",
}
_POOL_DEFAULTS = {
    "host": "prl.kryptex.network", "port": 7048, "tls": False, "ca_file": None,
    "wallet": "", "worker": "rig-01", "password": "x", "connect_timeout": 15,
    "read_timeout": 60, "reconnect_delay": 3,
    "protocol": "auto", "compression": "none", "certificate_version": None,
    "agent": "custom-pearl-trainer/0.1",
    "protocol_documentation": "https://pool.kryptex.com/en/prl",
}
_NODE_DEFAULTS = {
    "rpc_url": "https://127.0.0.1:44107", "username": "", "password": "",
    "ca_file": None, "poll_interval": 2, "refresh_interval": 15,
    "rpc_timeout": 15, "address": "",
}

# Unresolved paths make this suitable for a portable example JSON file. The
# loader copies each section and resolves paths relative to the selected file.
DEFAULT_CONFIG = {
    "version": 1, "mode": "pool", "official_source": ".official/pearl",
    "device": "cuda:0", "training": dict(_TRAINING_DEFAULTS),
    "pool": dict(_POOL_DEFAULTS), "node": dict(_NODE_DEFAULTS),
}


def _object_without_duplicates(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ConfigError("configuration contains a duplicate field")
        result[name] = value
    return result


def _reject_constant(_):
    raise ConfigError("configuration numbers must be finite")


def _section(value, allowed, section):
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be an object")
    if value.keys() - allowed:
        # Unknown field names can accidentally contain a pasted credential.
        raise ConfigError(f"{section} contains an unknown field")
    return value


def _string(value, field, *, nonempty=False, controls=False):
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string")
    if nonempty and not value.strip():
        raise ConfigError(f"{field} must be a nonempty string")
    if not controls and any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ConfigError(f"{field} must not contain control characters")
    return value


def _integer(value, field, minimum, maximum=None):
    if type(value) is not int:
        raise ConfigError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(f"{field} is outside the supported range")
    return value


def _number(value, field, *, zero=False):
    if type(value) not in (int, float):
        raise ConfigError(f"{field} must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (not zero and value == 0):
        adjective = "nonnegative" if zero else "positive"
        raise ConfigError(f"{field} must be {adjective} and finite")
    return value


def _path(value, field, directory, *, optional=False):
    if optional and value is None:
        return None
    value = _string(value, field, nonempty=True)
    try:
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = directory / resolved
        return str(resolved.resolve())
    except (OSError, RuntimeError, ValueError):
        raise ConfigError(f"{field} is not a valid filesystem path") from None


def _http_url(value, field, *, allow_fragment=False):
    value = _string(value, field, nonempty=True)
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"} and bool(parsed.hostname)
            and parsed.username is None and parsed.password is None
            and (allow_fragment or not parsed.fragment)
            and not any(char.isspace() for char in value)
        )
        port = parsed.port
        if port is not None and not 1 <= port <= 65535:
            valid = False
    except ValueError:
        valid = False
    if not valid:
        restrictions = "credentials" if allow_fragment else "credentials or a fragment"
        raise ConfigError(f"{field} must be an HTTP or HTTPS URL without {restrictions}")
    return value


def _host(value):
    value = _string(value, "pool.host", nonempty=True)
    if any(char.isspace() for char in value) or any(char in value for char in "/@?#\\"):
        raise ConfigError("pool.host must be a hostname or IP address without a URL or credentials")
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass
    # DNS names may have one final dot; a port must be supplied separately.
    if len(value.rstrip(".")) > 253:
        raise ConfigError("pool.host is not a valid hostname or IP address")
    labels = value.rstrip(".").split(".")
    if not all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
               for label in labels):
        raise ConfigError("pool.host is not a valid hostname or IP address")
    return value


def _documentation(value, directory):
    if value is None:
        return None
    value = _string(value, "pool.protocol_documentation", nonempty=True)
    if "://" in value or value.startswith(("http:", "https:")):
        return _http_url(value, "pool.protocol_documentation", allow_fragment=True)
    return _path(value, "pool.protocol_documentation", directory)


def load_config(path) -> dict:
    """Read version-1 JSON, fill defaults, and resolve paths beside the file.

    Missing payout addresses and credentials are allowed during loading. The
    selected runtime validates the settings it needs before starting work.
    Error messages identify schema fields without repeating user values.
    """
    if not isinstance(path, (str, Path)):
        raise ConfigError("configuration path must be a string or Path")
    try:
        config_path = Path(path).expanduser().resolve()
        raw = config_path.read_text(encoding="utf-8")
    except (OSError, RuntimeError, ValueError, UnicodeError):
        raise ConfigError("could not read the configuration file") from None
    try:
        config = json.loads(raw, object_pairs_hook=_object_without_duplicates,
                            parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise ConfigError("configuration must contain valid JSON") from None
    config = _section(config, _TOP_LEVEL, "configuration")
    if "version" not in config or type(config["version"]) is not int or config["version"] != 1:
        raise ConfigError("configuration.version must be the integer 1")
    mode = config.get("mode", "pool")
    if not isinstance(mode, str) or mode not in {"pool", "solo"}:
        raise ConfigError("configuration.mode must be pool or solo")
    device = _string(config.get("device", "cuda:0"), "device", nonempty=True)
    if re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
        raise ConfigError("device must select CUDA, for example cuda:0")
    directory = config_path.parent
    official_source = _path(config.get("official_source", ".official/pearl"),
                            "official_source", directory)

    training = dict(_TRAINING_DEFAULTS)
    training.update(_section(config.get("training", {}), set(training), "training"))
    for field in ("m", "n"):
        _integer(training[field], f"training.{field}", 128, 4096)
    _integer(training["k"], "training.k", 2048, 65536)
    if training["k"] % 128:
        raise ConfigError("training.k must be a multiple of 128")
    for field in ("tile_h", "tile_w"):
        _integer(training[field], f"training.{field}", 2, 128)
        if training[field] % 2:
            raise ConfigError(f"training.{field} must be even")
    if not 32 <= training["tile_h"] * training["tile_w"] <= 256:
        raise ConfigError("training tile area must be between 32 and 256")
    if training["m"] % training["tile_h"] or training["n"] % training["tile_w"]:
        raise ConfigError("training matrix dimensions must be divisible by their tile dimensions")
    if (training["tile_h"] + training["tile_w"]) * training["k"] > 1 << 22:
        raise ConfigError("training proof inputs exceed PEARL's 4 MiB worker-input limit")
    _integer(training["max_batches"], "training.max_batches", 0)
    _number(training["max_seconds"], "training.max_seconds", zero=True)
    training["output_dir"] = _path(training["output_dir"], "training.output_dir", directory)

    pool = dict(_POOL_DEFAULTS)
    pool.update(_section(config.get("pool", {}), set(pool), "pool"))
    pool["host"] = _host(pool["host"])
    _integer(pool["port"], "pool.port", 1, 65535)
    if type(pool["tls"]) is not bool:
        raise ConfigError("pool.tls must be true or false")
    pool["ca_file"] = _path(pool["ca_file"], "pool.ca_file", directory, optional=True)
    if pool["ca_file"] is not None and not pool["tls"]:
        raise ConfigError("pool.ca_file requires pool.tls to be true")
    for field in ("wallet", "worker", "password"):
        _string(pool[field], f"pool.{field}", controls=field == "password")
    if not isinstance(pool["protocol"], str) or pool["protocol"] not in {"auto", "kryptex", "pearlhash"}:
        raise ConfigError("pool.protocol must be auto, kryptex, or pearlhash")
    if not isinstance(pool["compression"], str) or pool["compression"] not in {"none", "gzip"}:
        raise ConfigError("pool.compression must be none or gzip")
    if pool["certificate_version"] is not None:
        _integer(pool["certificate_version"], "pool.certificate_version", 1, 3)
    _string(pool["agent"], "pool.agent", nonempty=True)
    if len(pool["agent"]) > 128:
        raise ConfigError("pool.agent must contain at most 128 characters")
    for field in ("connect_timeout", "read_timeout", "reconnect_delay"):
        _number(pool[field], f"pool.{field}")
    pool["protocol_documentation"] = _documentation(pool["protocol_documentation"], directory)

    node = dict(_NODE_DEFAULTS)
    node.update(_section(config.get("node", {}), set(node), "node"))
    node["rpc_url"] = _http_url(node["rpc_url"], "node.rpc_url")
    for field in ("username", "password", "address"):
        _string(node[field], f"node.{field}", controls=field == "password")
    if ":" in node["username"]:
        raise ConfigError("node.username must not contain a colon")
    node["ca_file"] = _path(node["ca_file"], "node.ca_file", directory, optional=True)
    if node["ca_file"] is not None and urlsplit(node["rpc_url"]).scheme != "https":
        raise ConfigError("node.ca_file requires an HTTPS node.rpc_url")
    for field in ("poll_interval", "refresh_interval", "rpc_timeout"):
        _number(node[field], f"node.{field}")

    return {"version": 1, "mode": mode, "official_source": official_source,
            "device": device, "training": training, "pool": pool, "node": node}
