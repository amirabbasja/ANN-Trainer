"""Export a plain authoring config for the standalone WR runtime."""

import copy
import json
from pathlib import Path

from .config import ConfigError, _atomic_write, _config_path, _validate_config, read_config, resolve_pool_settings
from .config_export import encode_config_document, write_open_text


def prepare_wr_config(source_path):
    """Select WR without changing the source document or resolving its paths."""
    values = copy.deepcopy(read_config(source_path))
    values["mode"] = "pool"
    values.setdefault("pool", {})["protocol"] = "WR"
    runtime = _validate_config(values, _config_path(source_path))
    if not runtime["vpn"]["enabled"]:
        raise ConfigError("WR export requires vpn.enabled to be true")
    pool = resolve_pool_settings(runtime["pool"], "WR")
    from .pool_training import validate_pool_endpoint, validate_pool_identity

    validate_pool_endpoint(pool)
    validate_pool_identity(pool, "WR", validate_address=False)
    return values


def export_wr_config(source_path, destination_path, *, open_text, save_key=True):
    """Write an encoded WR copy and, optionally, its adjacent dotenv text key."""
    source = _config_path(source_path)
    destination = _config_path(destination_path)
    if destination == source or destination.name == ".env":
        raise ConfigError("the WR export must use a separate configuration path")
    if type(save_key) is not bool:
        raise ConfigError("save_key must be true or false")
    values = prepare_wr_config(source)
    _validate_config(values, destination)
    envelope = encode_config_document(values, open_text)
    payload = json.dumps(envelope, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    try:
        original = None
        if destination.exists():
            with destination.open("r", encoding="utf-8", newline="") as stream:
                original = stream.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, UnicodeError):
        raise ConfigError("could not prepare the WR export destination") from None
    _atomic_write(destination, payload)
    if save_key:
        try:
            write_open_text(destination.with_name(".env"), open_text)
        except (OSError, ValueError):
            if original is None:
                destination.unlink(missing_ok=True)
            else:
                _atomic_write(destination, original)
            raise ConfigError("could not save the WR decode key; export was rolled back") from None
    return Path(destination)
