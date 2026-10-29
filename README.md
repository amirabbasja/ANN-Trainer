# LRAEP GPU trainer

LRAEP is a Python client for the PEARL network with a native CUDA search loop
and selectable Kryptex or WildRig/PearlHash pool connections. A bundled subset of PEARL's official
helpers supplies deterministic noise, commitments, Merkle proofs, and proof
verification. The helper packages use LRAEP names; their upstream authorship
and license notices are retained. Solo operation also supports a `pearld`
node. No external trainer application is required or launched.

For brevity, source code, filenames, configuration values and CLI choices use
**KEX** for **Kryptex** and **WR** for **WildRig**. Full names are kept in README
files for explanation and source attribution. Use `pool.protocol: "KEX"` or
`"WR"`, or `pool --miner KEX` / `pool --miner WR`. Older full-name protocol
values and filenames have been replaced; automatic selection by hostname has
been removed.

Bundled source and public configuration templates contain no real pool
addresses. The `pool.KEX.host` and `pool.WR.host` fields are blank: supply the host, port, identity
and TLS settings in your own private configuration. Existing private endpoint
and account settings are retained. The endpoint table below is a human
reference; it is not loaded by the program.

Here, **neural network training** names the PEARL work loop. The generated
workload is integer matrix multiplication; it does not fit a neural network,
update model weights, or run a model optimizer. Supplying useful model work
would require an additional model integration.

## Plain source configuration and WR export

Maintain the app settings in `config.json`. The trainer finds this file beside
`ann_trainer.py`, even when launched from another directory. `--config` remains
available if you store the shared file elsewhere. `config.example.json` is the
single public template; it contains both miner profiles with blank hosts.
The private `trainer/config.json` stays plain JSON so you can edit it directly.
It is ignored by git because it can contain credentials. The shared trainer can
use this source file for either KEX or WR. The standalone WR trainer uses an
encoded export at `LRAEP-WR-Trainer/config.json`, produced by the notebook.
Relative paths remain relative to each selected config file.

### Edit and export with the notebook

Open [`config_manager.ipynb`](config_manager.ipynb) and select the **LRAEP
trainer** kernel, or the project's `.venv/bin/python` interpreter in VS Code.
Run the setup and helper-definition cells, then call `show_config()` to print
the source on command, `update_from_dict(plain_dict)` to merge nested settings,
or `replace_from_dict(plain_dict)` to replace the whole source. These operations
save plain JSON. All example execution lines are commented, so running all
cells does not read or change either config or prompt for a key.

Call `export_wr_config()` when you want to update the standalone WR config.
First fill in `pool.WR`, `pool.WR_wallet`, and the enabled VPN settings
(`vpn.enabled: true` and `vpn.link`) in the plain source; export validates them.
It prompts for arbitrary text with `getpass`, uses that exact UTF-8 text as the
decode key, and writes the encoded document to the sibling
`LRAEP-WR-Trainer/config.json`. The exported copy selects pool mode and the WR
profile while preserving the source settings and relative paths. The source
`trainer/config.json` remains plain and unchanged.

By default, the same call writes the text to the lowercase variable
`open_text` in `LRAEP-WR-Trainer/.env`, preserving unrelated entries. WR reads
that adjacent file to decode its config. For example, the `.env` entry has
this form; the notebook handles quoting and escaping for the text you enter:

```dotenv
open_text="your chosen decode text"
```

The encoded JSON contains format metadata and ciphertext; its field values
are decoded in memory by WR. Enter the same text exactly, including spaces and
letter case, whenever you recreate its `.env` entry. To export elsewhere, pass
the destination path to `export_wr_config(output_path=...)`. Setting
`save_key=False` leaves `.env` unchanged, so that destination must already have
the matching `open_text` entry before WR starts.

The notebook prints only output paths during export. `show_config()` explicitly
prints credentials, and notebook editors may autosave that output; clear it
before saving or sharing. Setup disables IPython history logging, and the key
prompt keeps the entered text out of saved cell source.

Install the project requirements:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

On another machine, register the named Python 3.12+ kernel if it is not available:

```bash
uv pip install --python .venv/bin/python ipykernel
.venv/bin/python -m ipykernel install --user --name lraep-trainer --display-name 'LRAEP trainer'
```

The Python API also edits the plain source and preserves relative paths:

```python
from LRAEP_trainer.config import load_config, read_config, save_config, update_config

settings = load_config("config.json")  # Validated runtime values in memory.
plain_dict = {"pool": {"worker": "Rig01"}}
update_config("config.json", plain_dict)  # Preserve unspecified nested fields.
# save_config("config.json", complete_dict) replaces the entire document.
# read_config("config.json") returns the original dictionary without defaults.
```

Source writes validate before atomically replacing the file with owner-only
permissions. They do not create decode keys or modify an existing WR export.
Run the notebook export again after changing source settings that WR should use.
Configuration instructions for both trainers live in this README. The standalone
WR directory has no README or code comments; upstream license notices are retained.

KEX and WR share the `device`, `training`, `vpn` and `node` sections. The `pool`
section keeps both identities and separate connection settings:

```json
"pool": {
  "protocol": "KEX",
  "KEX_uname": "krxYOURID",
  "WR_wallet": "prl1YOUR_PEARL_ADDRESS",
  "worker": "Rig01",
  "KEX": {"host": "", "port": 8048, "tls": true},
  "WR": {"host": "", "port": 9000, "tls": false, "certificate_version": 3}
}
```

`--miner KEX` selects `pool.KEX` and `pool.KEX_uname`; `--miner WR` selects
`pool.WR` and `pool.WR_wallet`. Without that option, `pool.protocol` selects
the default miner. The worker, password, timeouts and unknown-request settings
are shared. Each connection profile can set `host`, `port`, `tls`, `ca_file`,
`compression`, `certificate_version` and `protocol_documentation`.
A missing selected profile or host produces an error before networking starts;
the other miner's connection settings are never used as a fallback.

Run either miner with the same file:

```bash
.venv/bin/python ann_trainer.py pool --miner KEX
.venv/bin/python ann_trainer.py pool --miner WR
```

KEX uses the Kryptex protocol. WR implements the recovered WildRig/PearlHash
protocol in this Python/CUDA client and does not launch WildRig. `KEX_uname`
accepts your Kryptex Mining Username or a PEARL receiving address. `WR_wallet`
takes your PEARL receiving address for PearlHash. Keep each identity separate
from `pool.worker`; the client constructs the appropriate login.

These endpoints are human references for your private profile settings:

| Profile | Reference host | Port | TLS |
|---|---|---|---|
| KEX global | `prl.kryptex.network` | `8048` | Yes |
| KEX global TCP | `prl.kryptex.network` | `7048` | No |
| KEX Hong Kong | `prl-hk.kryptex.network` | `8048` | Yes |
| WR/PearlHash captured endpoint | `pool.pearlhash.xyz` | `9000` | No |

See the [Kryptex pool page](https://pool.kryptex.com/en/prl) and
[WR work and unknown-request guide](docs/wr-training.md). The public template
contains no real pool hostname. The former `pool.wallet` field is replaced by
`KEX_uname` and `WR_wallet`; older flat connection settings still load, while
the shared template and private config use profiles for switching miners.

Start with these checks from `/workspace/trainer`:

```bash
/workspace/trainer/.venv/bin/python -m LRAEP_trainer --config /workspace/trainer/config.json pool-check
/workspace/trainer/.venv/bin/python -m LRAEP_trainer --config /workspace/trainer/config.json pool-login-check
```

`pool-check` opens and closes the configured TCP/TLS connection. For Kryptex,
`pool-login-check` authenticates with your configured identity and receives one
job; it performs no GPU work and submits no shares. Then start the custom GPU
client with:

```bash
/workspace/trainer/.venv/bin/python -m LRAEP_trainer --config /workspace/trainer/config.json pool
```

With `mode` set to `"pool"`, omitting the subcommand starts the same pool
training loop. A full node and node RPC credentials are unnecessary in pool
mode. The client uses the received header and certificate version, applies
PEARL's official share-target adjustment, and verifies candidates locally.
Kryptex receives native proofs; PearlHash receives compact nonce/index witnesses.
A matching reply with literal `result: true` and no error is
counted as accepted. A missing reply is not counted as acceptance.
PearlHash also requires `status: 1` on a matching submission acknowledgement.
Login and telemetry acknowledgements never count as accepted shares.

The miner choice selects the matching connection profile and identity in
the shared configuration:

```bash
.venv/bin/python ann_trainer.py pool --miner WR
.venv/bin/python ann_trainer.py pool --miner KEX
```

`training.main_effort` controls mining effort over time in both trainers. Each
entry is `[seconds, percentage]`; the default schedule is:

```json
"training": {
  "main_effort": [[60, 80], [120, 70], [180, 10]],
  "decoy_effort": []
}
```

This mines for 60 seconds at 80%, then 120 seconds at 70%, then 180 seconds at
10%, and repeats from the first entry until stopped. Durations must be positive,
finite numbers; an exit entry may use zero seconds. Percentages must be integers
from `-1` to `100`:

- `1`–`99` targets that average GPU duty cycle.
- `100` runs without pacing.
- `0` pauses mining for the entry's duration.
- `-1` stops all mining immediately when the entry begins and runs normal
  connection and VPN cleanup; its duration is ignored.

For example, `[[60, 80], [30, 0], [0, -1]]` mines at 80% for one minute,
pauses for 30 seconds, then exits. `main_effort` must contain at least one
entry. `decoy_effort` uses the same entry format and validation, but is reserved
for decoy mining and defaults to an empty list.

Edit the plain source directly, or use the notebook or `update_config` API:

```python
from LRAEP_trainer.config import update_config
update_config("config.json", {"training": {
    "main_effort": [[60, 80], [120, 70], [180, 10]],
    "decoy_effort": [],
}})
```

The miner finishes each GPU batch, synchronizes CUDA work, and rests to target
the active entry's average duty cycle. It submits completed shares before
resting. Waits respond to schedule changes, `Ctrl+C`, and time limits. A single
kernel can still show 100% utilization; other processes using the GPU are not
controlled. The pool status GPU percentage is a device-wide `nvidia-smi` sample,
so it can differ from the effort percentage.

The schedule follows elapsed wall time, including pauses and reconnections.
Transitions are checked between computation stages; an in-flight CUDA kernel
or synchronous verification finishes before the miner pauses or exits.

Older configurations with `training.max_gpu_utilization` remain readable. When
`main_effort` is absent, a legacy value such as `70` becomes `[[60, 70]]`, which
repeats at a constant 70%. An explicit `main_effort` takes precedence. New
configurations should use the schedule fields. Solo mode also accepts a JSON
schedule, for example `train --main-effort '[[60,80],[30,0],[0,-1]]'`.

`training.max_seconds` bounds a pool session using a stop timer, including
authentication and share-acknowledgement waits. CPU/GPU work already in a
computation stage finishes that stage before stopping. `training.max_batches`
stops the next GPU batch while allowing valid results from the completed batch
to be submitted. `Ctrl+C` stops the session and closes its connection.

During pool mining, one status row refreshes in place about once per second:

```text
Hashrate(avg) 700.00 MH/s | GPU 5% | accepted=0 rejected=0 | submitted=0 batches=100
```

The local average hashrate counts completed PEARL matrix work as
`batches * m * n * k / elapsed_seconds`, with automatic H/s through EH/s
units. One work unit is an int8 multiply-accumulate at the supported rank
128; tile attempts remain a separate counter. This follows the official
helper's `inner_hash_per_matmul` counter and PEARL's target scaling. It is
a local work rate; credited pool hashrate depends on accepted shares.

GPU utilization is the selected CUDA device's overall busy percentage,
sampled in the background using `nvidia-smi`. It shows `N/A` when unavailable.
Normal connection, share, and error messages clear the status row before
being logged. The row fits the terminal width to prevent wrapping; wider
terminals show additional counters. Redirected output writes only the final
status, rather than a line for every batch. `Ctrl+C` finishes the status row
and stops the sampler along with the miner.

For WR/PearlHash, the client also reports the selected GPU's name and measured
rate using `report_info`. It registers the GPU after login, updates the rate
after the first completed batch, then about every 10 seconds during the first
two minutes and every two minutes afterward. Each report logs the rate sent
and its request ID so it can be compared with the dashboard.
The wire rate uses raw H/s and the same average as the console. These reports
supply connected-worker GPU information; their acknowledgements never count
as accepted mining shares. Pool credit still depends on accepted shares.

Pool protocol options:

| Field | Default | Meaning |
|---|---|---|
| `protocol` | `"KEX"` | Explicitly select `"KEX"` or `"WR"`; `"pearlhash"` is an alias for `"WR"`. Hostnames do not select an adapter. |
| `KEX_uname` | `""` | KEX Mining Username or PEARL receiving address, used only by KEX. |
| `WR_wallet` | `""` | PEARL receiving address, used only by WR/PearlHash. |
| `KEX.compression` / `WR.compression` | `"none"` | Native uncompressed proof bytes; `"gzip"` requests the pool's optional gzip session. |
| `KEX.certificate_version` / `WR.certificate_version` | `null` | Use the explicit job field. An integer `1`, `2`, or `3` is a fallback only when a job omits it. Unknown versions fail. |
| `agent` | `"LRAEP-trainer/0.1"` | Client identifier sent during authentication. |
| `stop_on_unknown_request` | `true` | Record unknown PearlHash messages, stop work, close the connection, and report an error without reconnecting. |
| `unknown_request_dir` | `null` | Diagnostic directory, relative to the config file. Defaults to `training.output_dir/unknown-pool-requests`. |

Gzip is used only after the pool confirms `type: "v2"` in a successful
authentication reply. Otherwise proofs remain uncompressed. This session
version does not change PEARL's native proof serialization. Each profile's
`protocol_documentation` setting records a reference; it is never executed.

Kryptex uses static difficulty `2097152` by default. Optional password `d=N`
selects a value from `2097152` through `4294967296`. The actual share target
comes from each job; the client does not reconstruct it from this password.
The implemented dot-separated login follows captured Kryptex wire messages.

The payout wallet is blank in the shareable templates. The local configuration
contains the user's wallet, and verified TLS authentication and live V3 jobs
have been tested. Mock-pool tests and offline GPU/proof checks establish local
behavior. No live accepted share, credited balance, or payout is claimed here.

To use the existing solo trainer entirely from the configuration file, set
`mode` to `"solo"` and fill in `node.username`, `node.password`, `node.address`,
and the node URL/CA fields. Then run:

```bash
/workspace/trainer/.venv/bin/python -m LRAEP_trainer --config /workspace/trainer/config.json
```

An explicit `train` command selects solo training. Existing command-line
options override corresponding configuration values. With `--config`, node
credentials come from the file; without it, the existing environment variables
continue to work. No default payout address is included.

## VPN routing and link checks

LRAEP can route pool TCP/TLS and node HTTP/HTTPS connections through a V2Ray
share link using a managed local Xray core. Target DNS resolution also goes
through the proxy, and connection failures never trigger a direct retry.
The managed core receives its derived configuration through standard input;
VPN credentials are not written to a plaintext temporary config file.

Linux x86_64 includes the official Xray-core `v26.10.21` executable at
`src/LRAEP_trainer/bin/linux-x86_64/xray`; no separate VPN-core installation
is needed. Use the config notebook to set `vpn.link` and `vpn.enabled: true`
in your plain source config, then run:

```bash
.venv/bin/LRAEP --config config.json pool
```

Leave `vpn.core_path` as `null` to select the bundled core first, followed by
an executable on `PATH` or in the application data directory when no bundle
supports the platform. An explicit `vpn.core_path` selects your chosen core.
The optional `setup_vpn.py` installer remains available for Linux arm64 or a
separate installation; other platforms need a compatible executable selected
with `vpn.core_path`.

To compare links, put one configuration per line in the ignored `links.txt`
file, then run:

```bash
.venv/bin/LRAEP --config config.json vpn-check --links links.txt --attempts 3 --timeout 5
```

The checker reports connection latency, loss, HTTPS reachability, and exit IP.
It identifies links by line number and sends no pool login or mining shares.
Its “ping” measures tunneled TCP/TLS connections; it is not an ICMP ping.
See [VPN setup, supported links, and report interpretation](docs/vpn.md).

## Ready in this workspace

The local `.venv` and bundled `src/LRAEP_helpers` helper subset are installed.
The Linux x86_64 Xray VPN core, its original MPL-2.0 license, and checksum
manifest are included under `src/LRAEP_trainer/bin/` and packaged with the app.
The pool client automatically selects the native CUDA DP4A backend for 8×8
hash tiles. Startup reports `Mining backend: native CUDA DP4A`. The first run
builds this project's CUDA sources with `nvcc`; later runs reuse the app's
cache at `~/.cache/LRAEP/native/`.
No external miner executable is loaded. If CUDA compilation is unavailable,
startup explains the failure and selects the PyTorch reference backend.

The application source is in `src/LRAEP_trainer`. Required upstream Python
modules and Rust proof-helper sources are bundled in `src/LRAEP_helpers`;
an upstream Git checkout and the old `.official` directory are unnecessary.
The helper manifest records the reviewed upstream revision and source-file
checksums. Startup verifies these files before importing them.

Generated CUDA files use `~/.cache/LRAEP/native/`, and logs, benchmarks, and
saved results use `~/.local/share/LRAEP/artifacts/`. `XDG_CACHE_HOME` and
`XDG_DATA_HOME` are respected. Set `LRAEP_CACHE_DIR` or `LRAEP_DATA_DIR` to
override the app's cache or data root; the `native` and `artifacts` subdirectories
are added automatically. An explicit `training.output_dir` overrides the
result-output location.

From `/workspace/trainer`, run the offline GPU check:

```bash
.venv/bin/python -m LRAEP_trainer self-test
```

To also create and verify a complete local V3 ZK certificate:

```bash
.venv/bin/python -m LRAEP_trainer self-test --zk
```

Self-tests use synthetic headers and targets. They make no network requests.

## Set up on another machine

```bash
python setup_helpers.py
.venv/bin/python -m LRAEP_trainer self-test
```

Linux x86_64 installations already include the optional VPN core. Its source
and release attribution are recorded in [third-party notices](THIRD_PARTY_NOTICES.md);
platform alternatives are described in the [VPN guide](docs/vpn.md).

After setup, `.venv/bin/LRAEP` is an equivalent entry point:

```bash
.venv/bin/LRAEP --config config.json pool
```

No gateway service, vLLM application, or downloaded trainer executable is
started. Selected official Python modules are imported as bundled helper files.

## Train against your node

First run a synced official PEARL node and obtain your own PEARL Taproot payout
address. See the official repository's node and wallet setup instructions.
This client does not create a wallet, handle private keys, or start a node.

Set the RPC credentials and payout address. Use your actual values in place of
the placeholders:

```bash
export LRAEPD_RPC_URL='https://127.0.0.1:44107'
export LRAEPD_RPC_USER='your-rpc-user'
export LRAEPD_RPC_PASSWORD='your-rpc-password'
export LRAEPD_RPC_CA='/absolute/path/to/your/pearld/rpc.cert'
export LRAEP_TRAINING_ADDRESS='your-prl1p-taproot-address'

.venv/bin/python -m LRAEP_trainer train
```

HTTPS verifies the node certificate. For a localhost node explicitly configured
with `--notls`, set `LRAEPD_RPC_URL='http://127.0.0.1:44107'` and unset
`LRAEPD_RPC_CA`. These LRAEP application variables supply credentials to the
official node; they do not rename the upstream node software or wire protocol.

Useful options:

```bash
# Bound the session to 10 complete matrix batches.
.venv/bin/python -m LRAEP_trainer train --max-batches 10

# Stop starting new work after 60 seconds.
.venv/bin/python -m LRAEP_trainer train --max-seconds 60

# Verify and save any certified result, without submitting it.
.venv/bin/python -m LRAEP_trainer train --max-batches 10 --no-submit

# Select another CUDA device; global options precede the subcommand.
.venv/bin/python -m LRAEP_trainer --device cuda:1 train --m 512 --n 512
```

`Ctrl+C` stops work between slabs. The client polls the chain tip and discards
obsolete work, refreshes templates periodically, verifies every candidate with
the official verifier, creates its ZK certificate, then calls `submitblock`.
Certified blocks and their plain proof metadata are saved under
`~/.local/share/LRAEP/artifacts/` by default, or `training.output_dir` when set.
Only a null `submitblock` result is logged as accepted.

Sessions default to continuous work until interrupted. `--max-batches` counts
completed GPU batches, so a disconnected node can keep retrying until you stop
it; use `--max-seconds` to bound retry time as well. Certification already in
progress finishes before a time limit takes effect. RPC calls have a timeout.

## Arithmetic and protocol compatibility

- Rank is fixed at 128. Signals lie in `[-64, 63]`. Native CUDA noise generation
  matches the pinned official helper's keyed BLAKE3 bytes, uniform factors,
  and permutation indices exactly.
- Mining generates signals with an isolated CUDA random generator. The native
  path computes exact keyed BLAKE3 Merkle roots on the GPU and keeps signals
  there. Official CPU proof trees are built only when a candidate needs them;
  each resulting CPU root must match its GPU commitment before submission.
- Sparse noise application and signal addition use a fused integer CUDA kernel,
  retaining contiguous noisy A and transposed B on the device.
- One native kernel uses packed signed INT8 DP4A operations, cumulative INT32
  cells, tile XOR reduction, and rotate/XOR transcript accumulation. It avoids
  intermediate full-matrix products and float conversion.
- After every slab, each tile's cumulative int32 values are XOR-reduced. The
  16-word transcript uses the official rotate-left-by-13/XOR recurrence.
- Native final keyed BLAKE3 hashing and the inclusive little-endian target
  comparison run on the GPU. Only matching coordinates and digests are copied
  back. Full candidate buffers are retried without dropping matches. Target
  adjustment comes from the official binding.
- The PyTorch reference path remains available for unsupported tile shapes or
  unavailable native compilation. It preserves exact float32 rank-128 products
  with TF32 disabled and INT32 accumulation, plus independent CPU hashing.
- Certificates V1, V2, and V3 are selected from the node's
  `requiredcertversion`. V3 uses dimension-salted noise seeds. Unknown versions
  fail explicitly.
- Defaults: `m=n=256`, `k=2048`, tile `8×8`. This implementation requires
  `m,n >=128`, divisible tile dimensions, and `k` divisible by 128. Larger
  matrices consume more GPU and host memory.

The GTX 1650 was also checked with `m=n=4096`, `k=16384`, tile `8×8`,
including a complete GPU-to-local-pool exchange with native proof verification.
The local workspace configuration selects these larger batches; portable
example configurations keep the conservative defaults. Longer dot products
can improve throughput but increase preparation time and job-change latency.

The native pipeline retains Python orchestration and copies commitment roots
back for the official seed derivation. Candidate proof construction remains on
the CPU. Larger matrix dimensions increase signal storage and per-job latency.
The optimization tests compare against unchanged official noise/proof helpers
and an independent transcript oracle, including signed arithmetic, Merkle
padding, chunk counters, target boundaries, and transcript slot reuse.
Successful offline verification establishes arithmetic and proof compatibility;
live acceptance still requires a current job and a timely valid result. No
reward rate or live accepted share is implied by local hashrate measurements.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m LRAEP_trainer self-test
```

The unit tests compare GPU transcripts against an independent integer oracle,
including negative values and transcript slot reuse. Node tests exercise RPC
requests, certificate handling, rejection handling, and response errors.
The offline self-test compares against PEARL's official CPU reference and
verifies real plain proofs for all three certificate versions.

## Sources and permitted protocol references

Matrix arithmetic and proof helpers are pinned to PEARL commit
`0798903bced203de8cfcb49441f44fd9b51b18eb`. The bundled subset has renamed
modules and build paths; its source checksums are recorded in
`src/LRAEP_helpers/manifest.json`. The cryptographic domains, address format,
proof serialization, and Kryptex messages retain upstream-compatible bytes.
The user additionally authorized **"Allow protocol references only"** for
implementing pool networking. Public wire captures and Kryptex's published gzip
note inform the custom client. Official PEARL proof helpers and their required
proof-system dependencies are bundled; no external miner's GPU implementation
or executable is used. Exact protocol references and limits are recorded in
`docs/kex-protocol-review.md`.

The bundled helpers retain their upstream license notices in
`src/LRAEP_helpers/`, including the PEARL ISC notice and the proof-system
dependency licenses. See `THIRD_PARTY_NOTICES.md` and
`src/LRAEP_helpers/README.md` for attribution and vendoring details.

## Pool reference links

Full upstream names and addresses are confined to README documentation.
The abbreviated source documents refer here when an original URL contains
a full name. These links are references, not runtime endpoint defaults.

- [Upstream pool reference 1](https://github.com/xXGuilasXx/spark-pearl-miner/blob/65365633e3bff61b0f29d72109622dd22ff5a379/docs/protocol/kryptex.md)
- [Upstream pool reference 2](https://github.com/xXGuilasXx/spark-pearl-miner/blob/65365633e3bff61b0f29d72109622dd22ff5a379/tests/fixtures/capture-kryptex-8048-authorize.jsonl)
- [Upstream pool reference 3](https://pool.kryptex.com/articles/pearl-fp8-fork-en)
- [Upstream pool reference 4](https://pool.kryptex.com/openapi.yaml)

## External research workspace

WR research source labels refer to the sibling `/workspace/wildrig-checker`
workspace. For example, `wr-checker/captures/...` in fixture provenance means
that workspace's `captures/...` directory. It is not a runtime dependency.
