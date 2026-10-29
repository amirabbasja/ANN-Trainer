"""Command line entry point for the LRAEP GPU miner."""

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import logging
import math
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .gpu import TrainingCancelled, gpu_transcripts
from .gpu_pacing import GpuPacer
from .effort import EffortSchedule, normalize_effort
from .node import NodeClient, NodeError, build_block
from .protocol import (TrainingConfig, adjusted_target, load_official_helpers,
                       matching_tiles, prepare_batch, search_batch)
from . import native_cuda
from .paths import DEFAULT_SOURCE, artifacts_dir, default_config_path

LOG = logging.getLogger("LRAEP_trainer")


class TipWatcher:
    """Cancel obsolete GPU work when the node advances or becomes unreachable."""

    def __init__(self, node: NodeClient, interval: float, *, schedule=None,
                 stopped=None, deadline=None):
        self.node = node
        self.interval = interval
        self.schedule = schedule
        self.deadline = deadline
        self._rpc_timeout = node.timeout
        self.tip = None
        self.failed = True
        self.stopped = stopped if stopped is not None else threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stopped.is_set():
            boundary = math.inf
            if self.schedule is not None:
                if not self.schedule.wait(self.stopped, deadline=self.deadline):
                    break
                percentage, boundary = self.schedule.current()
                if percentage <= 0:
                    continue
            if self.deadline is not None:
                boundary = min(boundary, self.deadline)
                if time.monotonic() >= self.deadline:
                    break
            remaining = boundary - time.monotonic()
            if remaining <= 0:
                continue
            self.node.timeout = min(self._rpc_timeout, remaining)
            try:
                self.tip = self.node.best_hash()
                self.failed = False
            except NodeError:
                self.failed = True
            self.stopped.wait(max(0, min(self.interval, boundary - time.monotonic())))

    def stale(self, parent: str) -> bool:
        return self.failed or self.tip != parent


def require_cuda(device: str) -> None:
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("select an available CUDA GPU, for example --device cuda:0")
    name = torch.cuda.get_device_name(torch.device(device))
    LOG.info("GPU: %s", name)


def self_test(args) -> int:
    """Exercise all supported certificate versions without contacting a network."""
    from LRAEP_trainer import primitives as pm
    from miner_base.noisy_gemm import NoisyGemm

    require_cuda(args.device)
    cfg = TrainingConfig(m=128, n=128)
    use_native = native_cuda.available(args.device)
    # With this synthetic target and work factor, almost every tile qualifies.
    # It is used only locally and is never sent to a node.
    header = pm.IncompleteBlockHeader(
        version=1, prev_block=bytes(32), merkle_root=bytes(32),
        timestamp=0, nbits=0x1E7FFFFF,
    )
    base_target = 0x7FFFFF << (8 * (0x1E - 3))
    rng = np.random.default_rng(20261025)
    for version in (1, 2, 3):
        batch = prepare_batch(header, version, cfg, rng, device=args.device, gpu_signals=True,
                              native=use_native)
        if use_native:
            transcripts = native_cuda.transcripts_device(batch.a_noised, batch.b_noised_t)
            transcripts = transcripts.cpu().numpy().view(np.uint32)
        else:
            transcripts = gpu_transcripts(batch.a_noised, batch.b_noised,
                                          cfg.tile_h, cfg.tile_w, args.device)
        # Compare a complete k=2048 tile against the official CPU reference.
        reference = NoisyGemm(noise_rank=128, hash_tile_h=cfg.tile_h,
                              hash_tile_w=cfg.tile_w)
        _, _, official = reference._process_output_tile(
            batch.a_noised.cpu(), batch.b_noised.cpu(), cfg.k, 0, cfg.tile_h, 0, cfg.tile_w,
        )
        expected = np.array(official[0][0].data, dtype=np.uint32)
        if not np.array_equal(transcripts[0, 0], expected):
            raise RuntimeError(f"V{version} GPU transcript differs from official reference")
        target = adjusted_target(base_target, cfg)
        row, col, _ = next(matching_tiles(transcripts, batch.commitment.noise_seed_A, target))
        proof = batch.plain_proof(row, col)
        valid, message = pm.verify_plain_proof_for_cert_version(version, header, proof)
        if not valid:
            raise RuntimeError(f"V{version} official proof verification failed: {message}")
        LOG.info("V%d: GPU transcript matches official CPU reference; plain proof verified", version)
        if args.zk and version == 3:
            LOG.info("Creating a local V3 ZK certificate using the official proof helper")
            zk = pm.generate_proof_for_cert_version(version, header, proof)
            valid, message = pm.verify_proof_for_cert_version(version, header, zk)
            if not valid:
                raise RuntimeError(f"V3 ZK proof verification failed: {message}")
            LOG.info("V3 ZK proof verified (%d proof bytes)", len(zk.proof_data))
    LOG.info("Self-test passed; no network connection or block submission was made")
    return 0


def train(args) -> int:
    from LRAEP_trainer import primitives as pm

    stopped = threading.Event()
    started = time.monotonic()
    schedule = EffortSchedule(args.main_effort, started=started)
    deadline = started + args.max_seconds if args.max_seconds else None
    watcher = None
    batches = attempts = accepted = 0
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, lambda *_: stopped.set())

    def finished():
        return (stopped.is_set() or schedule.ended()
                or (args.max_batches > 0 and batches >= args.max_batches)
                or (deadline is not None and time.monotonic() >= deadline))

    with schedule.stopping(stopped):
        try:
            # Pauses and an immediate exit must not initialize a mining backend.
            if not schedule.wait(stopped, deadline=deadline) or finished():
                return 0
            require_cuda(args.device)
            if not args.address:
                raise ValueError("set LRAEP_TRAINING_ADDRESS or provide --address (a PEARL Taproot payout address)")
            username = getattr(args, "rpc_username", os.environ.get("LRAEPD_RPC_USER", ""))
            password = getattr(args, "rpc_password", os.environ.get("LRAEPD_RPC_PASSWORD", ""))
            if not username or not password:
                raise ValueError("set node.username and node.password in --config, or LRAEPD_RPC_USER and LRAEPD_RPC_PASSWORD")
            cfg = TrainingConfig(args.m, args.n, args.k, args.tile_h, args.tile_w)
            pacer = GpuPacer(args.device, 100)
            LOG.info("GPU effort schedule: %s", args.main_effort)
            use_native = ((cfg.tile_h, cfg.tile_w) == (8, 8) and native_cuda.available(args.device))
            from LRAEP_gateway.blockchain_utils.blockchain_utils import get_script_pubkey_from_p2tr_address
            get_script_pubkey_from_p2tr_address(args.address)
            node_options = dict(url=args.rpc_url, username=username, password=password,
                                ca_file=args.rpc_ca, timeout=args.rpc_timeout)
            node = NodeClient(**node_options)
            watcher = TipWatcher(NodeClient(**node_options), args.poll_interval,
                                 schedule=schedule, stopped=stopped, deadline=deadline)
            watcher.thread.start()
            rng = np.random.default_rng()
            output = Path(args.output_dir).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)

            def stage_deadline(boundary):
                return boundary if deadline is None else min(boundary, deadline)

            def rpc(method, *values):
                # Existing requests remain synchronous and bounded by the next
                # effort boundary, rather than continuing in detached threads.
                percentage, boundary = schedule.current()
                if stopped.is_set() or percentage <= 0:
                    raise TrainingCancelled("GPU effort is paused or stopped")
                remaining = stage_deadline(boundary) - time.monotonic()
                if remaining <= 0:
                    raise TrainingCancelled("GPU effort interval ended")
                node.timeout = min(args.rpc_timeout, max(0.001, remaining))
                return method(*values)

            while not finished():
                if not schedule.wait(stopped, deadline=deadline) or finished():
                    break
                try:
                    template = rpc(node.get_template, args.address)
                    if schedule.current()[0] <= 0 or finished():
                        continue
                    cert = int(template.required_cert_version)
                    if cert not in (1, 2, 3):
                        raise ValueError(f"unsupported certificate version {cert}; update official helpers")
                    header = template.header.incomplete_header
                    parent = bytes(header.prev_block).hex()
                    target = adjusted_target(template.target, cfg)
                    expires = time.monotonic() + args.refresh_interval
                    LOG.info("Training height=%d certificate=V%d shape=%dx%dx%d",
                             template.height, cert, cfg.m, cfg.n, cfg.k)
                    watcher.tip = rpc(node.best_hash)
                    watcher.failed = False

                    def template_stale():
                        return finished() or watcher.stale(parent) or time.monotonic() >= expires

                    while not template_stale():
                        if not schedule.wait(stopped, deadline=deadline) or template_stale():
                            break
                        percentage, boundary = schedule.current()
                        if percentage <= 0:
                            continue
                        pacer.set_utilization(percentage)

                        def cancelled():
                            return (template_stale() or time.monotonic() >= boundary
                                    or schedule.current()[0] <= 0)

                        def candidate_cancelled():
                            # A completed final batch remains eligible unless
                            # a schedule transition, stop, or chain tip invalidates it.
                            return (stopped.is_set() or watcher.stale(parent)
                                    or time.monotonic() >= boundary
                                    or schedule.current()[0] <= 0)

                        pacer.begin()
                        batch = prepare_batch(header, cert, cfg, rng, device=args.device, gpu_signals=True,
                                              native=use_native)
                        if cancelled():
                            pacer.finish(cancelled=True)
                            pacer.wait(stopped, deadline=stage_deadline(boundary))
                            continue
                        try:
                            matches = search_batch(batch, target, args.device, cancelled)
                        except TrainingCancelled:
                            pacer.finish(cancelled=True)
                            pacer.wait(stopped, deadline=stage_deadline(boundary))
                            continue
                        pacer.finish()
                        batches += 1
                        attempts += (cfg.m // cfg.tile_h) * (cfg.n // cfg.tile_w)
                        for row, col, _ in matches:
                            if candidate_cancelled():
                                break
                            if rpc(node.best_hash) != parent or candidate_cancelled():
                                break
                            proof = batch.plain_proof(row, col)
                            if candidate_cancelled():
                                break
                            valid, message = pm.verify_plain_proof_for_cert_version(cert, header, proof)
                            if not valid:
                                raise RuntimeError(f"GPU candidate rejected locally: {message}")
                            if candidate_cancelled():
                                break
                            LOG.info("Training candidate verified; creating its ZK certificate")
                            block_bytes = build_block(proof, template)
                            if candidate_cancelled():
                                break
                            name = f"candidate-{template.height}-{hashlib.sha256(block_bytes).hexdigest()[:16]}"
                            (output / f"{name}.bin").write_bytes(block_bytes)
                            (output / f"{name}.json").write_text(json.dumps({
                                "height": template.height, "certificate_version": cert,
                                "incomplete_header_hex": bytes(header.to_bytes()).hex(),
                                "plain_proof_base64": proof.to_base64(),
                            }, indent=2) + "\n")
                            if args.no_submit:
                                LOG.info("Verified block saved to %s (submission disabled)", output / f"{name}.bin")
                            elif rpc(node.best_hash) == parent and not candidate_cancelled():
                                rpc(node.submit, block_bytes)
                                accepted += 1
                                LOG.info("Node accepted the training result at height %d", template.height)
                            else:
                                LOG.info("Chain advanced during certification; saved candidate is stale")
                            break
                        elapsed = max(time.monotonic() - started, 0.001)
                        LOG.info("Training batches=%d tile_attempts=%d attempts/s=%.1f accepted=%d",
                                 batches, attempts, attempts / elapsed, accepted)
                        if not finished():
                            pacer.wait(stopped, deadline=stage_deadline(boundary))
                except TrainingCancelled:
                    continue
                except NodeError as exc:
                    LOG.warning("%s; retrying in 3 seconds", exc)
                    _, boundary = schedule.current()
                    stopped.wait(max(0, min(3, stage_deadline(boundary) - time.monotonic())))
        finally:
            stopped.set()
            if watcher is not None:
                watcher.stopped.set()
                watcher.thread.join()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    LOG.info("Training stopped: batches=%d tile_attempts=%d accepted=%d", batches, attempts, accepted)
    return 0


def pool_check(configuration) -> int:
    """Check the advertised pool transport without guessing its login format."""
    from .config import resolve_pool_settings
    from .pool_transport import PoolTransport
    from .pool_training import validate_pool_endpoint

    pool = resolve_pool_settings(configuration["pool"])
    validate_pool_endpoint(pool)
    with PoolTransport(
        host=pool["host"], port=pool["port"], tls=pool["tls"],
        ca_file=pool["ca_file"], connect_timeout=pool["connect_timeout"],
        read_timeout=pool["read_timeout"],
    ):
        LOG.info("Pool %s connection established to %s:%d",
                 "TLS" if pool["tls"] else "TCP", pool["host"], pool["port"])
    LOG.info("Transport check complete; login, job receipt, and share acceptance were not tested")
    return 0


def pool_train(configuration) -> int:
    """Run our GPU worker directly against the configured pool."""
    from .pool_training import run_pool_training

    return run_pool_training(configuration)


def _add_training_arguments(parser):
    """Allow explicit command-line options to override configuration values."""
    for name in ("m", "n", "k", "tile-h", "tile-w", "max-batches"):
        parser.add_argument(f"--{name}", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-seconds", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max-gpu-utilization", type=int, default=argparse.SUPPRESS,
                        help="legacy constant GPU effort, 1-100 (100 disables pacing)")
    parser.add_argument("--main-effort", default=argparse.SUPPRESS,
                        help='JSON schedule of [seconds, percentage] pairs; 0 pauses and -1 exits')
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)


def _apply_settings(args, configuration):
    """Config values replace defaults; explicit command options take priority."""
    from .config import DEFAULT_CONFIG

    training = configuration["training"] if configuration else {}
    node = configuration["node"] if configuration else {}
    if hasattr(args, "main_effort") and hasattr(args, "max_gpu_utilization"):
        raise ValueError("use only one of --main-effort and --max-gpu-utilization")
    if hasattr(args, "main_effort"):
        try:
            effort = json.loads(args.main_effort)
        except (TypeError, ValueError):
            raise ValueError("main-effort must be a JSON list of [seconds, percentage] pairs") from None
    elif hasattr(args, "max_gpu_utilization"):
        if type(args.max_gpu_utilization) is not int or not 1 <= args.max_gpu_utilization <= 100:
            raise ValueError("max-gpu-utilization must be an integer between 1 and 100")
        effort = [[60, args.max_gpu_utilization]]
    else:
        effort = training.get("main_effort", DEFAULT_CONFIG["training"]["main_effort"])
    args.main_effort = normalize_effort(effort, field="main_effort")
    args.decoy_effort = normalize_effort(training.get("decoy_effort", []),
                                         field="decoy_effort", allow_empty=True)
    defaults = dict(
        official_source=configuration["official_source"] if configuration else
        os.environ.get("LRAEP_OFFICIAL_SOURCE", str(DEFAULT_SOURCE)),
        device=configuration["device"] if configuration else "cuda:0",
        m=training.get("m", 256), n=training.get("n", 256),
        k=training.get("k", 2048), tile_h=training.get("tile_h", 8),
        tile_w=training.get("tile_w", 8), max_batches=training.get("max_batches", 0),
        max_seconds=training.get("max_seconds", 0),
        output_dir=training.get("output_dir", str(artifacts_dir())),
        address=node.get("address") if configuration else os.environ.get("LRAEP_TRAINING_ADDRESS"),
        rpc_url=node.get("rpc_url", "https://127.0.0.1:44107") if configuration else
        os.environ.get("LRAEPD_RPC_URL", "https://127.0.0.1:44107"),
        rpc_ca=node.get("ca_file") if configuration else os.environ.get("LRAEPD_RPC_CA"),
        rpc_timeout=node.get("rpc_timeout", 15), poll_interval=node.get("poll_interval", 2),
        refresh_interval=node.get("refresh_interval", 15),
    )
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    if configuration:
        # A config-driven run gets all RPC settings from the selected file.
        args.rpc_username = node["username"]
        args.rpc_password = node["password"]
    return args


@contextmanager
def _network_scope(configuration):
    settings = (configuration or {}).get("vpn", {})
    if not settings.get("enabled", False):
        yield
        return
    from .network import routed
    from .vpn import ManagedVPN

    with ManagedVPN(settings) as proxy, routed(proxy):
        LOG.info("VPN routing enabled through Xray; direct fallback disabled")
        yield


def _execute(args, configuration):
    if args.command in ("train", "pool") and args.main_effort[0][1] == -1:
        LOG.info("GPU effort schedule requests an immediate exit")
        return 0
    if args.command == "vpn-check":
        from .config import DEFAULT_CONFIG, resolve_pool_settings
        from .vpn_check import run_vpn_check

        settings = configuration if configuration is not None else copy.deepcopy(DEFAULT_CONFIG)
        settings = dict(settings, pool=resolve_pool_settings(settings["pool"], getattr(args, "miner", None)))
        links = args.links if args.links is not None else settings["vpn"]["links_file"]
        return run_vpn_check(settings, links, timeout=args.timeout,
                             attempts=args.attempts, output=args.output)
    if args.command in ("pool", "pool-check", "pool-login-check"):
        if configuration is None:
            raise ValueError("pool commands require --config; edit config.json with your settings")
        from .config import resolve_pool_settings
        from .pool_training import validate_pool_endpoint, validate_pool_identity

        pool_settings = resolve_pool_settings(configuration["pool"], getattr(args, "miner", None))
        configuration = dict(configuration, pool=pool_settings)
        validate_pool_endpoint(pool_settings)
        if args.command != "pool-check":
            validate_pool_identity(pool_settings, validate_address=False)
    network_configuration = configuration if args.command != "self-test" else None
    with _network_scope(network_configuration):
        if args.command in ("pool", "pool-check", "pool-login-check"):
            if configuration is None:
                raise ValueError("pool commands require --config; edit config.json with your settings")
            configuration = dict(configuration, device=args.device,
                                 official_source=args.official_source)
            if args.command == "pool-check":
                return pool_check(configuration)
            if args.command == "pool-login-check":
                from .pool_training import run_pool_training

                return run_pool_training(configuration, login_only=True)
            return pool_train(configuration)
        if args.command == "train":
            for name in ("poll_interval", "refresh_interval", "rpc_timeout"):
                value = getattr(args, name)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name.replace('_', '-')} must be positive and finite")
            if args.max_batches < 0 or args.max_seconds < 0 or not np.isfinite(args.max_seconds):
                raise ValueError("training limits must be finite and nonnegative")
        load_official_helpers(args.official_source)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        return self_test(args) if args.command == "self-test" else train(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LRAEP GPU miner for the PRL network")
    parser.add_argument("--config", help="plain JSON configuration; defaults to the application config.json")
    parser.add_argument("--official-source", default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--miner", choices=("KEX", "WR"), default=argparse.SUPPRESS,
                        help="select KEX or WR and its connection settings from the shared configuration")
    commands = parser.add_subparsers(dest="command")
    check = commands.add_parser("self-test", help="compare GPU math and official proof verification offline")
    check.add_argument("--zk", action="store_true", help="also generate and verify a V3 ZK certificate")
    run = commands.add_parser("train", help="train against your PEARL node and submit certified results")
    run.add_argument("--address", default=argparse.SUPPRESS)
    run.add_argument("--rpc-url", default=argparse.SUPPRESS)
    run.add_argument("--rpc-ca", default=argparse.SUPPRESS)
    run.add_argument("--rpc-timeout", type=float, default=argparse.SUPPRESS)
    run.add_argument("--poll-interval", type=float, default=argparse.SUPPRESS)
    run.add_argument("--refresh-interval", type=float, default=argparse.SUPPRESS)
    _add_training_arguments(run)
    run.add_argument("--no-submit", action="store_true", help="save certified blocks locally")
    for name, help_text in (
        ("pool", "train directly against a supported PEARL pool"),
        ("pool-check", "check TCP/TLS transport from --config without sending a login"),
        ("pool-login-check", "authenticate and receive one pool job without GPU work"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--miner", choices=("KEX", "WR"), default=argparse.SUPPRESS,
                             help="select KEX or WR and its connection settings from the shared configuration")
    vpn = commands.add_parser("vpn-check", help="test V2Ray links through their tunnels without mining")
    vpn.add_argument("--miner", choices=("KEX", "WR"), default=argparse.SUPPRESS,
                     help="test the selected miner's pool connection settings")
    vpn.add_argument("--links", help="one V2Ray link per line; defaults to vpn.links_file or links.txt")
    vpn.add_argument("--attempts", type=int, default=3, help="pool connectivity attempts per link (1–20)")
    vpn.add_argument("--timeout", type=float, default=5, help="seconds per connection attempt (up to 60)")
    vpn.add_argument("--output", help="optional JSON report; share-link credentials are omitted")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        from .config import load_config

        selected_config = args.config
        if selected_config is None and args.command != "self-test":
            canonical = default_config_path()
            if canonical.is_file():
                selected_config = canonical
        configuration = load_config(selected_config) if selected_config else None
        if args.command is None:
            if configuration is None:
                parser.error("specify a command, or use --config to select the configured mode")
            args.command = "pool" if configuration["mode"] == "pool" else "train"
            args.no_submit = False
        _apply_settings(args, configuration)
        return _execute(args, configuration)
    except KeyboardInterrupt:
        LOG.info("Stopped")
        return 130
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
