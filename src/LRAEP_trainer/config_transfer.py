"""Export readable WR settings with only S_3 encrypted."""

import copy
import json
from pathlib import Path

from .config import ConfigError, _atomic_write, _config_path, _validate_config, read_config, resolve_S3_settings
from .config_export import DEFAULT_DECRYPT_KEY, ExportError, encode_config_document


def prepare_wr_config(source_path, *, key=None):
    """Select WR and include CNN settings without changing the source."""
    values = copy.deepcopy(read_config(source_path, key=key))
    values["mode"] = "S3"
    values.setdefault("S3", {})["protocol"] = "WR"
    runtime = _validate_config(values, _config_path(source_path))
    if not runtime["vpn"]["enabled"]:
        raise ConfigError("WR export requires vpn.enabled to be true")
    S3 = resolve_S3_settings(runtime["S3"], "WR")
    from .S3_training import validate_S3_endpoint, validate_S3_identity

    validate_S3_endpoint(S3)
    validate_S3_identity(S3, "WR", validate_address=False)
    # Materialize the shared dataset default inside the exported copy;
    # the standalone source intentionally has no repository URL default.
    values.setdefault("side_job", {}).setdefault("repo_url", runtime["side_job"]["repo_url"])
    return values


def export_wr_config(source_path, destination_path, *, key=DEFAULT_DECRYPT_KEY, source_key=None):
    """Write a WR copy with only S_3 encrypted, without creating a key file."""
    source = _config_path(source_path)
    destination = _config_path(destination_path)
    if destination == source or destination.name == ".env":
        raise ConfigError("the WR export must use a separate configuration path")
    values = prepare_wr_config(source, key=source_key)
    _validate_config(values, destination)
    try:
        envelope = encode_config_document(values, DEFAULT_DECRYPT_KEY if key is None else key)
    except ExportError as exc:
        raise ConfigError(str(exc)) from None
    payload = json.dumps(envelope, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, UnicodeError):
        raise ConfigError("could not prepare the WR export destination") from None
    _atomic_write(destination, payload)
    return Path(destination)
