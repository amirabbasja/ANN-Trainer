# LRAEP GPU trainer

LRAEP is a Python client for the S6 network with a native CUDA search loop
and selectable Kryptex or WildRig/S7 pool connections. A bundled subset of S6's official
helpers supplies deterministic noise, commitments, Merkle proofs, and proof
verification. The helper packages use LRAEP names; their upstream authorship
and license notices are retained. Solo operation also supports a `S6d`
node. No external trainer application is required or launched.

For brevity, source code, filenames, configuration values and CLI choices use
**KEX** for **Kryptex** and **WR** for **WildRig**. Full names are kept in README
files for explanation and source attribution. Use `pool.protocol: "KEX"` or
`"WR"`, or `pool --S5 KEX` / `pool --S5 WR`. Older full-name protocol
values and filenames have been replaced; automatic selection by hostname has
been removed.

Bundled source and public configuration templates contain no real pool
addresses. The `pool.KEX.host` and `pool.WR.host` fields are blank: supply the host, port, identity
and TLS settings in your own private configuration. Existing private endpoint
and account settings are retained. The endpoint table below is a human
reference; it is not loaded by the program.

Here, **neural network training** names the S6 work loop. The generated
workload is integer matrix multiplication; it does not fit a neural network,
update model weights, or run a model optimizer. Supplying useful model work
would require an additional model integration.

The shared trainer now includes a separate [CNN side-job](side-job/README.md)
for Tibetan page orientation. It runs during `job-1` schedule intervals or with
`.venv/bin/python side-job/train.py`. Its `side_job` config section controls
the dataset, layer sizes and independent VPN (disabled by default). All its data
and checkpoints live under `data/side-job/`.

## Plain source configuration and WR export

Set `training.generate_artifacts` to `false` for training without local output,
terminal logs, or Telegram updates. Candidate files, unknown-request captures,
and CNN checkpoints are skipped. The CNN reads dataset shards in memory, and
native backends reuse existing libraries or use their reference implementations.
The default is `true`. For the installed `LRAEP` command or `python -m LRAEP_trainer`,
set `PYTHONDONTWRITEBYTECODE=1` to suppress Python bytecode during startup as well.

### Periodic Telegram updates

Set `trainer_name` to the name that should appear at the start of each update.
Telegram settings come from the selected configuration file:

```json
{
  "trainer_name": "my-trainer",
  "telegram": {
    "enabled": true,
    "tgBot_key": "YOUR_BOT_TOKEN",
    "tgChatID": "YOUR_CHAT_ID",
    "update_interval_seconds": 300
  }
}
```

The interval is a positive number of seconds; the first update arrives after
one interval. Reports contain elapsed time, completed batches, GPU utilization,
average hashrate and share counters. Solo reports contain batches, attempts and
accepted blocks. Updates continue during pool reconnects and schedule pauses.
`telegram.enabled: false` disables reporting, as does omitting the section.
Connection and login checks do not send updates.

Delivery uses the [Telegram Bot API](https://core.telegram.org/bots/api#sendmessage)
on a background thread and follows the trainer's configured VPN route. A delivery
failure logs a warning and retries at the next interval without stopping training.
Bot tokens are omitted from delivery warnings. Keep credentials in the private
config; the public example contains blank credentials. Start the bot in Telegram
before expecting it to message your private chat.

WR exports retain `trainer_name` and `telegram` inside the encoded configuration.
Set the name in the source config before exporting if you want the WR copy to
use a different name.

### Source and exported files

Both trainers use readable JSON configuration. Only the top-level `S_3`
value is encrypted, using the default decryption key `qwe`. No `.env` or
separate key file is needed. Training, CNN, Telegram, VPN, and other settings
remain ordinary JSON fields. The config editing API decrypts `S_3` into an
`S3` dictionary. The standalone WR training entry points use the separate
login flow below.
Config files remain ignored by git and are saved with owner-only permissions.
Relative paths remain relative to the selected config file.

### WR login address in memory

The standalone WR CLI and training notebook read `config.json` through a
short-lived helper. At startup it returns settings with `WR_S2` and
`KEX_uname` removed. For each login or reconnect, a fresh helper decrypts
the private config, validates `WR_S2`, and builds the encoded login frame.
Only that frame crosses into the trainer. The helper exits before the trainer
sends it. The trainer holds it in a mutable buffer and overwrites it
immediately after sending, including when the send fails. Restart training
after changing the WR endpoint.

The login address is processed briefly in memory during authentication.
This limits its lifetime in the application. Python, TLS, and operating
system buffers can still hold transient copies outside that buffer's
control. Config editing and display helpers still return decrypted values
and can keep them in the notebook kernel. Restart a kernel that previously
loaded the full config to discard its old process memory.

### Edit and save with the notebook

Open [`config_manager.ipynb`](config_manager.ipynb) and run its setup and
helper-definition cells. Its `CONFIG_PATH` points directly to
`LRAEP-WR-Trainer/config.json`. Call `show_config()` to display the decrypted
settings, `update_from_dict(plain_dict)` to merge changes, or
`replace_from_dict(plain_dict)` to replace the configuration. Both save helpers
write directly to that standalone directory, encrypting only `S_3` with `qwe`.
Use `S3` for the connection section in the dictionaries passed to these helpers.
All example calls are commented, so running every cell does not change files.

For example, run this explicitly after the setup cells:

```python
update_from_dict({"training": {"job_sequence": [{"title": "job-0", "time": 600, "max_gpu_utilization": 100}, {"title": "job-1", "time": 120, "max_gpu_utilization": 100}]}})
update_from_dict({"S3": {"worker": "device-01"}})
```

`export_wr_config()` is an optional import action: it reads
`trainer/config.json`, selects the WR profile, and replaces
`LRAEP-WR-Trainer/config.json`. It requires the WR endpoint, WR identity,
and enabled VPN settings. Ordinary notebook edits do not require an export.
The import defaults to `qwe`; it does not prompt for a key or write `.env`.
For a separate output, pass `output_path=...`. A custom key can be supplied
as `decode_key=...`; consumers must then pass the same key explicitly.

`show_config()` prints decrypted settings only when called. Clear its output
before sharing the notebook. Export and save helpers print only destination
paths. Setup disables IPython history logging.

Install the project requirements:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

On another machine, register the named Python 3.12+ kernel if it is not available:

```bash
uv pip install --python .venv/bin/python ipykernel
.venv/bin/python -m ipykernel install --user --name lraep-trainer --display-name 'LRAEP trainer'
```

The Python API also edits either config and preserves relative paths:

```python
from LRAEP_trainer.config import load_config, read_config, save_config, update_config

settings = load_config("config.json")  # Validated runtime values in memory.
plain_dict = {"S3": {"worker": "device-01"}}
update_config("config.json", plain_dict)  # Preserve unspecified nested fields.
# save_config("config.json", complete_dict) replaces the entire document.
# read_config("config.json") returns decrypted settings without defaults.
# load_config/save_config/update_config accept key="custom key" when needed.
```

Writes validate before atomically replacing the selected file with owner-only
permissions. They encrypt only `S_3` and never create key files. The notebook
updates the standalone file directly; a shared trainer import is optional.
Configuration instructions for both trainers live in this directory. See
[`docs/standalone-wr.md`](docs/standalone-wr.md) for notebook controls and worker
lifecycle details. Keep README files, mining-related comments and docstrings,
and explanations of these conventions out of `LRAEP-WR-Trainer/`. Maintain
that documentation, upstream source attribution and license notices in this
shared trainer directory.

The standalone trainer requires no cryptocurrency-specific packages, SDKs, or
external S5 installations. Its protocol and proof support is bundled
locally. Its installation requirements are general-purpose Python packages:
`numpy`, `torch`, `blake3`, `pydantic`, `loguru`, `cryptography`, `ipykernel`,
`Pillow`, and `pyarrow`. In particular, `cryptography` supplies configuration
encryption and `blake3` supplies general-purpose hashing; neither is a
cryptocurrency SDK. The notebook setup installs missing packages from that
requirements list.

The standalone `config.json` keeps CNN settings and the dataset repository URL
readable. Only `S_3` is encrypted. Its default decryption key is `qwe`, and
startup never reads a `.env` file. The standalone source has no dataset URL
default; an active CNN interval requires `side_job.repo_url` in the config.

Both trainers compute during `job-0` intervals and train the CNN during `job-1`
intervals, preserving model, optimizer, and partial epoch state between them.
Use an interval percentage of `100` to disable intentional GPU pacing.
Standalone checkpoints live in `LRAEP-WR-Trainer/data/side-job/`; see the linked
worker guide for checkpoint transfer and restart behavior.

Start the combined application from `LRAEP-WR-Trainer/train.ipynb` by running
its setup and training cells. The notebook defines and calls ordinary Python
training code directly in the kernel, and its training cell stays busy for the
session. The application loads that directory's `config.json` using the
WR login flow above and passes the readable `training.job_sequence` to the
scheduler. The notebook contains no
separate schedule, and the worker does not read the shared authoring config at
runtime. Each run reads the current config; rerun
the training cell after saving changes from the config manager.

The sequence repeats indefinitely: after the final interval, the next interval
is the first one again. Job-0 and CNN state persist across cycles. For continuous
operation, use `training.max_batches: 0`, `training.max_seconds: 0`, and
`side_job.epochs: 0`, with no `-1` exit entries. Zero here means unlimited.
Existing pause entries still
pause for their configured duration before the repeating sequence advances.

Progress, epoch summaries, connection messages, and errors appear as ordinary
notebook output, just as they do in a console. Training runs in the kernel.
Use the notebook's **Interrupt** button
to stop manually. Resource cleanup closes the CNN inputs, pool connection,
telemetry, and VPN. Restart the kernel when you want a complete reset of its
CUDA context and GPU memory. Other cells wait until training finishes.

The notebook includes the training entry point directly and does not import or
execute `train.py`. That file remains available for separate command-line use.

KEX and WR share the `device`, `training`, `vpn` and `node` sections. The `pool`
section keeps both identities and separate connection settings:

```json
"pool": {
  "protocol": "KEX",
  "KEX_uname": "krxYOURID",
  "WR_wallet": "prl1YOUR_S6_ADDRESS",
  "worker": "Rig01",
  "KEX": {"host": "", "port": 8048, "tls": true},
  "WR": {"host": "", "port": 9000, "tls": false, "certificate_version": 3}
}
```

`--S5 KEX` selects `pool.KEX` and `pool.KEX_uname`; `--S5 WR` selects
`pool.WR` and `pool.WR_wallet`. Without that option, `pool.protocol` selects
the default S5. The worker, password, timeouts and unknown-request settings
are shared. Each connection profile can set `host`, `port`, `tls`, `ca_file`,
`compression`, `certificate_version` and `protocol_documentation`.
A missing selected profile or host produces an error before networking starts;
the other S5's connection settings are never used as a fallback.

Run either S5 with the same file:

```bash
.venv/bin/python ann_trainer.py pool --S5 KEX
.venv/bin/python ann_trainer.py pool --S5 WR
```

KEX uses the Kryptex protocol. WR implements the recovered WildRig/S7
protocol in this Python/CUDA client and does not launch WildRig. `KEX_uname`
accepts your Kryptex Mining Username or a S6 receiving address. `WR_wallet`
takes your S6 receiving address for S7. Keep each identity separate
from `pool.worker`; the client constructs the appropriate login.

These endpoints are human references for your private profile settings:

| Profile | Reference host | Port | TLS |
|---|---|---|---|
| KEX global | `prl.kryptex.network` | `8048` | Yes |
| KEX global TCP | `prl.kryptex.network` | `7048` | No |
| KEX Hong Kong | `prl-hk.kryptex.network` | `8048` | Yes |
| WR/S7 captured endpoint | `pool.pearlhash.xyz` | `9000` | No |

See the [Kryptex pool page](https://pool.kryptex.com/en/prl) and
[WR work and unknown-request guide](docs/wr-training.md). The public template
contains no real pool hostname. The former `pool.wallet` field is replaced by
`KEX_uname` and `WR_wallet`; older flat connection settings still load, while
the shared template and private config use profiles for switching S5s.

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
S6's official share-target adjustment, and verifies candidates locally.
Kryptex receives native proofs; S7 receives compact nonce/index witnesses.
A matching reply with literal `result: true` and no error is
counted as accepted. A missing reply is not counted as acceptance.
S7 also requires `status: 1` on a matching submission acknowledgement.
Login and telemetry acknowledgements never count as accepted shares.

The S5 choice selects the matching connection profile and identity in
the shared configuration:

```bash
.venv/bin/python ann_trainer.py pool --S5 WR
.venv/bin/python ann_trainer.py pool --S5 KEX
```

`training.job_sequence` is an array of dictionaries. Every dictionary must
contain exactly `title`, `time`, and `max_gpu_utilization`:

```json
"training": {
  "job_sequence": [
    {"title": "job-0", "time": 60, "max_gpu_utilization": 80},
    {"title": "job-1", "time": 30, "max_gpu_utilization": 20},
    {"title": "job-0", "time": 10, "max_gpu_utilization": 0},
    {"title": "job-0", "time": 0, "max_gpu_utilization": -1}
  ]
}
```

`job-0` runs the primary computation; `job-1` trains the page-orientation CNN.
`time` is a positive, finite number of seconds; an exit entry may use zero.
`max_gpu_utilization` must be an integer from `-1` to `100`. Values `1`–`99`
set the GPU duty-cycle target, `100` runs without intentional pacing, `0`
pauses for that interval, and `-1` exits when the entry begins. There is no
tolerance or randomized target. Array entries and extra fields are rejected.

The example runs job-0 at 80% for one minute, job-1 at 20% for 30 seconds,
pauses for ten seconds, then exits. Without an exit entry, the sequence repeats
from the beginning. Both jobs share one monotonic wall-clock timeline and the
same pause, stop, and duration rules. Model and partial epoch state survive
job switches. The default schedule runs job-0 for 60 seconds at 80%, 120
seconds at 70%, and 180 seconds at 10%, then repeats.

The config manager saves edits directly to the standalone directory:

```python
update_from_dict({"training": {"job_sequence": [
    {"title": "job-0", "time": 60, "max_gpu_utilization": 80},
    {"title": "job-1", "time": 30, "max_gpu_utilization": 20},
]}})
```

GPU work is paced between batches. The standalone CNN additionally paces
between CUDA operations during forward, backward, and optimizer work. Fresh
hardware readings above the configured `max_gpu_utilization` delay new WR
work until the sample falls to that value or the interval ends. There is no
additional utilization margin. Missing or stale readings use time-based pacing;
`100` bypasses intentional pacing. An in-flight operation can finish after a
schedule boundary, and measured device-wide utilization can briefly exceed
the configured target.

A legacy `training.max_gpu_utilization` value still supplies a repeating
job-0 dictionary when `job_sequence` is absent. The CLI accepts the same JSON
dictionary format through `--job-sequence`, for example:

```bash
.venv/bin/python ann_trainer.py train --job-sequence '[{"title":"job-0","time":60,"max_gpu_utilization":80},{"title":"job-1","time":30,"max_gpu_utilization":20}]'
```

`training.max_seconds` bounds a pool session using a stop timer, including
authentication and share-acknowledgement waits. CPU/GPU work already in a
computation stage finishes that stage before stopping. `training.max_batches`
stops the next GPU batch while allowing valid results from the completed batch
to be submitted. `Ctrl+C` stops the session and closes its connection.

Both jobs use the same compact status row in the terminal and periodic
Telegram training stats:

```text
1 | job-0 | 80% | 00:01:00 | 123.45
2 | job-1 | 50% | 00:00:30 | 4
```

The fields are the cumulative interval number, job title, measured GPU
utilization, remaining interval time, and speed. Job-0 speed is the compute
rate without a unit; job-1 speed is the current epoch number. Telegram sends
these periodic updates through `telegram.training_stats_tgBot_key`. Lifecycle
events continue to use `telegram.tgBot_key` and the shared `tgChatID`.
`update_interval_seconds` controls the training-stats interval.

GPU utilization is the selected CUDA device's overall busy percentage,
sampled in the background using `nvidia-smi`. It shows `N/A` when unavailable.
Normal connection, training, and error messages clear the status row before
being logged. The row fits the terminal width to prevent wrapping. Redirected
output writes only the final status, rather than a line for every batch.
`Ctrl+C` finishes the status row
and stops the sampler along with the S5.

For WR/S7, the client also reports the selected GPU's name and measured
rate using `report_info`. It registers the GPU after login, updates the rate
after the first completed batch, then about every 10 seconds during the first
two minutes and every two minutes afterward. Each report logs the rate sent
and its request ID at `DEBUG` so it can be compared with the dashboard.
The wire rate uses raw H/s calculated from completed matrix work as
`batches * m * n * k / elapsed_seconds`. One work unit is an int8
multiply-accumulate at the supported rank 128. This follows the official
helper's `inner_hash_per_matmul` counter and S6's target scaling. These reports
supply connected-worker GPU information; their acknowledgements never count
as accepted mining shares. Pool credit still depends on accepted shares.

Pool protocol options:

| Field | Default | Meaning |
|---|---|---|
| `protocol` | `"KEX"` | Explicitly select `"KEX"` or `"WR"`; `"S7"` is an alias for `"WR"`. Hostnames do not select an adapter. |
| `KEX_uname` | `""` | KEX Mining Username or S6 receiving address, used only by KEX. |
| `WR_wallet` | `""` | S6 receiving address, used only by WR/S7. |
| `KEX.compression` / `WR.compression` | `"none"` | Native uncompressed proof bytes; `"gzip"` requests the pool's optional gzip session. |
| `KEX.certificate_version` / `WR.certificate_version` | `null` | Use the explicit job field. An integer `1`, `2`, or `3` is a fallback only when a job omits it. Unknown versions fail. |
| `agent` | `"LRAEP-trainer/0.1"` | Client identifier sent during authentication. |
| `stop_on_unknown_request` | `true` | Record unknown S7 messages, stop work, close the connection, and report an error without reconnecting. |
| `unknown_request_dir` | `null` | Diagnostic directory, relative to the config file. Defaults to `training.output_dir/unknown-pool-requests`. |

Gzip is used only after the pool confirms `type: "v2"` in a successful
authentication reply. Otherwise proofs remain uncompressed. This session
version does not change S6's native proof serialization. Each profile's
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
in your config, then run:

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
hash tiles. Startup reports `Compute backend: native CUDA DP4A`. The first run
builds this project's CUDA sources with `nvcc`; later runs reuse the app's
cache at `~/.cache/LRAEP/native/`.
No external S5 executable is loaded. If CUDA compilation is unavailable,
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

First run a synced official S6 node and obtain your own S6 Taproot payout
address. See the official repository's node and wallet setup instructions.
This client does not create a wallet, handle private keys, or start a node.

Set the RPC credentials and payout address. Use your actual values in place of
the placeholders:

```bash
export LRAEPD_RPC_URL='https://127.0.0.1:44107'
export LRAEPD_RPC_USER='your-rpc-user'
export LRAEPD_RPC_PASSWORD='your-rpc-password'
export LRAEPD_RPC_CA='/absolute/path/to/your/S6d/rpc.cert'
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

Python and CUDA code use `training_threshold` for the exact 256-bit batch
acceptance bound, `base_training_threshold` before scaling, and
`encoded_training_threshold` for its received hexadecimal form. Rate helpers
use `S3_training_rate` and `format_training_rate`. The standalone WR trainer
uses the same names. Serialized protocol field names retain their wire format.

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
The offline self-test compares against S6's official CPU reference and
verifies real plain proofs for all three certificate versions.

`tests/test_local_dependencies.py` checks both trainers for cryptocurrency SDK
imports and package requirements. It also imports both runtimes with those
SDKs and network access blocked, verifying that their bundled protocol code
does not require a cryptocurrency package installation.

## Sources and permitted protocol references

Matrix arithmetic and proof helpers are pinned to S6 commit
`0798903bced203de8cfcb49441f44fd9b51b18eb`. The bundled subset has renamed
modules and build paths; its source checksums are recorded in
`src/LRAEP_helpers/manifest.json`. The cryptographic domains, address format,
proof serialization, and Kryptex messages retain upstream-compatible bytes.
The user additionally authorized **"Allow protocol references only"** for
implementing pool networking. Public wire captures and Kryptex's published gzip
note inform the custom client. Official S6 proof helpers and their required
proof-system dependencies are bundled; no external S5's GPU implementation
or executable is used. Exact protocol references and limits are recorded in
`docs/kex-protocol-review.md`.

The bundled helpers retain their upstream license notices in
`src/LRAEP_helpers/`, including the S6 ISC notice and the proof-system
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


## Abbreviations

| Original | Abbreviation |
| --- | --- |
| mining | S1 |
| wallet | S2 |
| pool | S3 |
| stratum | S4 |
| miner | S5 |
| pearl | S6 |
| pearlhash | S7 |

The standalone `LRAEP-WR-Trainer` uses `S4` for the Stratum-compatible KEX
wire protocol, including the `wire_request` and `notification_method` adapter
functions. Its endpoint-required wire methods retain their original spellings.

These substitutions apply case-insensitively to local names, filenames, imports,
configuration keys, and CLI commands (for example, `pool` becomes `S3` and
`WR_wallet` becomes `WR_S2`, and `--miner` becomes `--S5`). Existing README
examples retain the original S1–S4 spellings; substitute those abbreviations
when using them. S5–S7 examples use the renamed spellings; the longer S7 term
is replaced before S6. Dynamic protocol names use the
[native word codec and placeholder table](#native-word-codec) below. Legacy
encoded fixed byte constants are also decoded in that compiled library,
preserving network and proof bytes. The previous S7 protocol alias remains
accepted through its decoded spelling. Upstream URLs, endpoint hostnames,
source attribution, and license text retain their required spellings.

## Native message buffers

Pool sessions in both repositories use `wire_io.NativeChannel` for incoming
and outgoing S3/WR messages. The author source is `native/wire_codec.cpp`;
the standalone receives `libwire_codec.so` and the Python bridge, without the
decoder C++ source. The native layer owns the receive, JSON token, serialization
and WR framing buffers. It translates the listed protocol names to placeholders
before Python parses the resulting neutral JSON. Sending performs the reverse
translation in native memory. Message values are preserved; account addresses,
job data and other values needed by Python are still ordinary Python data.

The channel uses `recv_into` and `sendall` with scoped memory views of native
buffers, including on TLS sockets. Native allocations are explicitly zeroed
when discarded, resized or closed, including failure paths. Transport setup,
TLS verification and SOCKS routing retain their existing behavior. The original
raw transport and S6 codec APIs remain available for explicit compatibility use;
they are not the production session JSON path.

The trainer's node RPC client also reads response bodies into native buffers
and sends scoped native request buffers. Its `raw_template()` returns neutral
field names. RPC models retain neutral schemas with no decoded aliases.
`Model.model_validate_wire(stream)` accepts a `readinto` stream;
`with model.wire_buffer() as data:` exposes a scoped native view for sending.
Default `model_validate_json()` also translates through native code. Explicit
Pydantic alias/name policies and strict JSON options keep the compatibility
parser to preserve their semantics. Prefer the scoped native wire-buffer API
when sending a model. Explicit APIs returning
wire dictionaries or strings, including `model_dump(by_alias=True)` and
`model_dump_json()`, necessarily give the caller plaintext and do not promise
physical erasure. Supplying an already decoded Python string or bytes to a
parser cannot erase the caller's original allocation.

This reduces ownership and lifetime of original protocol text; it does not
provide host-level confidentiality. TLS, HTTP, Python and OS internals may hold
other copies. Unknown fields and unrecognized data values pass through and can
remain in Python. The host administrator can observe data while it is used.
The change adds no GPU synchronization; a hashrate improvement or percentage
cost requires a representative workload benchmark.

Rebuild and distribute both native libraries and bridges with
`python3 tools/build_word_codec.py`, then verify with
`python3 tools/build_word_codec.py --check`. Restart running application or
notebook processes to load the new modules and shared libraries.

<!-- native-word-codec:start -->
## Native word codec

The author checkout's `native/word_codec.cpp` defines the exact aliases used
by both trainers. Training-related names identify internal constants. The
codec resolves them to the original text or bytes wherever the peer protocol,
proof format, diagnostics, or regression fixtures require those values.
The table is reversible and the original values are public below.

```python
from LRAEP_trainer.word_codec import FixedAlias, encode_fixed, fixed_buffer, transform_word

transform_word("legacy_protocol", "decode")
transform_word("pearlhash", "encode")
assert encode_fixed(b"target") == FixedAlias("training.field.acceptance_bound")
with fixed_buffer(FixedAlias("training.field.acceptance_bound")) as temporary:
    consume(temporary)
```

The corresponding standalone import is `workrunner.native_text_codec`.
`decode_word` and `encode_word` match structural protocol aliases exactly and
pass unlisted text through unchanged. `encode_fixed` and `fixed_buffer` with
a `FixedAlias` perform exact lookup across the full table. An unknown fixed
alias or value raises `ValueError`. Fixed values can include arbitrary binary
bytes. The codec performs no substring replacement.

### Placeholder table (decode direction)

| Training alias | Decoded value |
| --- | --- |
| `account` | `wallet` |
| `legacy_protocol` | `pearlhash` |
| `training_rate` | `hashrate` |
| `server_seed` | `extranonce` |
| `local_seed` | `miner_nonce` |
| `device_seed` | `worker_nonce` |
| `training.subscribe` | `mining.subscribe` |
| `training.authorize` | `mining.authorize` |
| `training.submit` | `mining.submit` |
| `training.notify` | `mining.notify` |
| `training.reconnect` | `mining.reconnect` |
| `noise_seed_domain` | `pearl/cert-v3/noise-seed/` |
| `identity_prefix` | `prl1` |
| `identity_mode_prefix` | `solo:` |
| `device_default` | `rig-01` |
| `record_id` | `txid` |
| `parent_digest` | `previousblockhash` |
| `size_limit` | `vsizelimit` |
| `records` | `transactions` |
| `poll_id` | `longpollid` |
| `counter_range` | `noncerange` |
| `reward_metadata` | `coinbaseaux` |
| `reward_amount` | `coinbasevalue` |
| `default_attestation` | `default_witness_commitment` |
| `required_version` | `requiredcertversion` |
| `template_request` | `getblocktemplate` |
| `tip_request` | `getbestblockhash` |
| `submit_record` | `submitblock` |
| `training.format.batch_frame_coefficients` | `1e62c4909baa47273a9650ad471b61ac63360c3c590e48592245bdbb2a05010408030263e81ef40609040bab3763667b` (48 bytes, hex) |
| `training.capability.append_attestation_record` | `coinbase/append` |
| `training.capability.separate_attestation_payload` | `segwit` |
| `training.config.coordinator_section` | `S3` |
| `training.config.legacy_batch_protocol` | `S7` |
| `training.field.acceptance_bound` | `target` |
| `training.field.attestation_revision` | `cert_version` |
| `training.field.attestation_version` | `certificate_version` |
| `training.field.batch_identifier` | `job_id` |
| `training.field.batch_seed` | `nonce` |
| `training.field.compact_acceptance_bound` | `nbits` |
| `training.field.coordinator_work_identifier` | `workid` |
| `training.field.dense_attestation` | `plain_proof` |
| `training.field.input_row_index_label` | `a_rows index` |
| `training.field.input_row_indices` | `a_rows` |
| `training.field.matrix_tile_height` | `tile_h` |
| `training.field.matrix_tile_width` | `tile_w` |
| `training.field.parent_record_digest` | `prev_block` |
| `training.field.signal_baseline` | `sb` |
| `training.field.signal_baseline_parameters` | `sb_p` |
| `training.field.tensor_commitment_root` | `merkle_root` |
| `training.field.tensor_tail_mode` | `tail_mode` |
| `training.field.training_worker` | `worker` |
| `training.field.weight_column_index_label` | `b_cols index` |
| `training.field.weight_column_indices` | `b_cols` |
| `training.fixture.byte_sweep` | `000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff` (256 bytes, hex) |
| `training.fixture.empty_input` | `""` |
| `training.fixture.text_sample` | `abc` |
| `training.format.attestation_payload_tag` | `aa21a9ed` |
| `training.format.batch_header_mask` | `9d2f73c411a85be0367ad14e920cbf65f32884da176eb941cc530fa67de21974` (32 bytes, hex) |
| `training.format.batch_update_prefix_mask` | `4dc0d47dff723cf8f75a583b596cf848f5106dd0d0aa7bdded9d4d566a929ffaabea8985cf34a26af1b118e2c697f283f0caadd643b0975390a17c79aae32e69cf166b57a6f64dd02914e7e4` (76 bytes, hex) |
| `training.format.empty_parent_output_index` | `ffffffff` |
| `training.label.attestation_integer` | `proof integer` |
| `training.label.attestation_version` | `certificate version` |
| `training.label.attestation_version_config_path` | `{}.certificate_version` |
| `training.label.batch_acceptance_bound` | `share target` |
| `training.label.batch_update_payload` | `notify payload` |
| `training.label.legacy_batch_protocol_prefix` | `"S7 "` |
| `training.label.tensor_commitment_tree` | `Merkle` |
| `training.log.accelerator_duty_cycle` | `%s effort GPU duty-cycle target: %d%%` |
| `training.log.accelerator_metrics_acknowledged` | `S7 acknowledged GPU information report` |
| `training.log.accelerator_metrics_acknowledgement_missing` | `S7 discarded an unacknowledged GPU report` |
| `training.log.accelerator_metrics_rejected` | `S7 rejected GPU information report` |
| `training.log.accelerator_metrics_sent` | `S7 GPU report sent: %s (request=%d)` |
| `training.log.accepted_batch_tile_location` | `"Found valid block at "` |
| `training.log.attestation_record_assembly` | `Generating block from PlainProof` |
| `training.log.attestation_record_generated` | `Generated block` |
| `training.log.batch_generation` | `S7 job=%s generation=%d` |
| `training.log.batch_header_verified_no_compute` | `Authenticated job verified: header=%d bytes, target=%d bits; no GPU work performed` |
| `training.log.batch_header_verified_no_submissions` | `Authenticated job verified: header=%d bytes, target=%d bits; no GPU work or shares submitted` |
| `training.log.batch_solution_verified` | `S1 solution verified successfully` |
| `training.log.batch_submission_accepted` | `S3 accepted training share job=%s` |
| `training.log.batch_submission_rejected` | `S3 rejected training share job=%s: %s` |
| `training.log.coordinator_batch_revision` | `S3 job=%s height=%d certificate=V%d` |
| `training.log.coordinator_session_authenticated` | `S3 authenticated at %s:%d; proof compression=%s` |
| `training.log.training_run_summary` | `S3 training stopped: batches=%d submitted=%d accepted=%d rejected=%d stale=%d` |
| `training.log.zero_knowledge_attestation_generated` | `Generated ZK proof` |
| `training.log.zero_knowledge_attestation_verified` | `verified ZK proof` |
| `training.log.zero_knowledge_attestation_verifying` | `verifying ZK proof` |
| `training.message.accelerator_metrics` | `report_info` |
| `training.message.batch_update` | `notify` |
| `training.message.coordinator_reconnect` | `client.reconnect` |
| `training.message.open_session` | `login` |
| `training.message.submit_batch_result` | `submit` |
| `training.metadata.client_agent` | `LRAEP-trainer/0.1` |
| `training.metadata.trainer_title` | `LRAEP-WR-Trainer` |
| `training.path.unrecognized_batch_messages` | `unknown-S3-requests` |
| `training.tensor.input_noise_domain` | `415f74656e736f72000000000000000000000000000000000000000000000000` (32 bytes, hex) |
| `training.tensor.left_operand_label` | `A` |
| `training.tensor.right_operand_label` | `B` |
| `training.tensor.weight_noise_domain` | `425f74656e736f72000000000000000000000000000000000000000000000000` (32 bytes, hex) |
| `training.thread.batch_message_reader` | `WR-S3-reader` |
| `training.thread.batch_session_stop_monitor` | `WR-S3-stop` |
| `training.validation.accelerator_attestation_digest_length` | `GPU share digest must be exactly 32 bytes` |
| `training.validation.accelerator_batch_digest_mismatch` | `GPU S7 digest differs from independently reconstructed CPU work` |
| `training.validation.accelerator_metrics_acknowledgement_timeout` | `S7 GPU report acknowledgement timed out` |
| `training.validation.accelerator_metrics_authentication_required` | `S7 session must authenticate before reporting GPU information` |
| `training.validation.accelerator_metrics_send_failed` | `S7 GPU report send failed` |
| `training.validation.accelerator_reference_digest_mismatch` | `GPU share digest differs from the official CPU reference` |
| `training.validation.accelerator_required` | `an available CUDA GPU is required for S3 training` |
| `training.validation.accelerator_tensor_commitment_mismatch` | `native GPU Merkle root differs from the official CPU tree` |
| `training.validation.accelerator_throughput_range` | `hashrate must be a nonnegative finite number in H/s` |
| `training.validation.acceptance_bound_bit_length` | `target must fit in 256 bits` |
| `training.validation.acceptance_bound_hex_format` | `target must be a hexadecimal string` |
| `training.validation.acceptance_bound_interval_prefix` | `pow_target must be in the range pow_target=` |
| `training.validation.acceptance_bound_positive_hex_range` | `target must be a positive 256-bit hexadecimal string` |
| `training.validation.acceptance_bound_scaling_overflow` | `target scaling overflows; choose a harder test-network target` |
| `training.validation.account_identity_required_suffix` | `" must contain a valid S6 Taproot address"` |
| `training.validation.attestation_commitment_length_prefix` | `"Proof commitment must be "` |
| `training.validation.attestation_commitment_mismatch` | `Proof commitment mismatch` |
| `training.validation.attestation_commitment_missing` | `Proof commitment is not set` |
| `training.validation.attestation_digest_outside_bound` | `proof digest does not meet difficulty target` |
| `training.validation.attestation_input_size_limit` | `training proof inputs exceed S6's 4 MiB worker-input limit` |
| `training.validation.attestation_matrix_rows_invalid` | `invalid proof matrix rows` |
| `training.validation.attestation_pattern_offset_invalid` | `invalid proof pattern offset` |
| `training.validation.attestation_rank_or_inner_dimension` | `unsupported proof rank or common dimension` |
| `training.validation.attestation_revision_too_old` | `proof requires a later certificate version` |
| `training.validation.attestation_revision_unsupported` | `unsupported certificate version` |
| `training.validation.attestation_selected_tile_size_limit` | `selected proof rows and columns exceed S6's 4 MiB worker-input limit` |
| `training.validation.attestation_tensor_signal_range` | `proof signal values must lie in [-64, 64]` |
| `training.validation.attestation_tile_shape_invalid` | `invalid proof tile shape` |
| `training.validation.attestation_truncated` | `truncated proof` |
| `training.validation.attestation_vector_length_invalid` | `invalid proof vector length` |
| `training.validation.attestation_version_prefix` | `"unsupported S6 certificate version "` |
| `training.validation.attestation_version_template` | `unsupported S6 certificate version {}` |
| `training.validation.authenticated_batch_required` | `no authenticated S3 job was received before stopping` |
| `training.validation.batch_acceptance_bound_range` | `S7 final target must be a positive uint256` |
| `training.validation.batch_account_identity_whitespace` | `WR_S2 and worker must contain no whitespace` |
| `training.validation.batch_and_compact_attestation_required` | `S7 submission requires its job and compact share` |
| `training.validation.batch_attestation_revision_required` | `job is missing cert_version; configure it explicitly or update the S3` |
| `training.validation.batch_candidate_digest_length` | `S7 candidate digest must contain exactly 32 bytes` |
| `training.validation.batch_candidate_outside_acceptance_bound` | `S7 candidate does not meet the exact S3 target` |
| `training.validation.batch_compression_unsupported` | `S7 compression must be none` |
| `training.validation.batch_device_seed_length` | `S7 worker nonce must contain eight bytes` |
| `training.validation.batch_dimensions_or_signal_mode` | `Unsupported S7 work dimensions or signal mode` |
| `training.validation.batch_generation_attestation_revision` | `S7 generation 3 requires certificate version 3` |
| `training.validation.batch_generation_three_required` | `S7 training requires an observed generation-3 job` |
| `training.validation.batch_generation_unsupported` | `Only observed S7 generation 3 is supported` |
| `training.validation.batch_header_compact_bound_invalid` | `S7 decoded header has an invalid compact target` |
| `training.validation.batch_header_length` | `S7 header must contain exactly 76 bytes` |
| `training.validation.batch_notification_name_malformed` | `Malformed S7 notification name` |
| `training.validation.batch_replaced_before_submission` | `S3 replaced this training job before submission` |
| `training.validation.batch_request_identifier_range` | `S7 request ID must be an unsigned 64-bit integer` |
| `training.validation.batch_request_unknown` | `Unknown S7 request; training stopped` |
| `training.validation.batch_response_envelope_unknown` | `Unknown S7 response envelope` |
| `training.validation.batch_seed_pair_length` | `S7 nonces must each contain exactly eight bytes` |
| `training.validation.batch_server_fields_unknown` | `Unknown S7 server message fields or method` |
| `training.validation.batch_server_message_shape_unknown` | `Unknown S7 server request or response shape` |
| `training.validation.batch_server_method_unknown` | `Unknown S7 server method` |
| `training.validation.batch_server_notification_unknown` | `Unknown S7 server notification` |
| `training.validation.batch_session_authentication_rejected` | `S7 rejected authentication` |
| `training.validation.batch_submission_compression_unsupported` | `S7 does not support KEX gzip proof submissions` |
| `training.validation.batch_submission_device_seed_mismatch` | `S7 share worker nonce does not match its job` |
| `training.validation.batch_submission_dimensions_type` | `S7 share dimensions must be integers` |
| `training.validation.batch_submission_fields_invalid` | `Invalid S7 share fields` |
| `training.validation.batch_submission_outside_acceptance_bound` | `training share does not meet the exact S3 target` |
| `training.validation.batch_tile_coordinate_range` | `S7 tile coordinates are out of range` |
| `training.validation.batch_tile_index_order_and_seed_row` | `S7 tile indices must be sorted, unique and outside the nonce row` |
| `training.validation.batch_update_payload_length` | `S7 notify payload must contain 200 bytes` |
| `training.validation.batch_update_schema` | `Unsupported S7 notify schema` |
| `training.validation.compact_acceptance_bound_invalid` | `Invalid compact target` |
| `training.validation.compact_attestation_required` | `S7 verification requires a compact share` |
| `training.validation.coordinator_acceptance_bound_kind` | `S3 target kind must be base or final` |
| `training.validation.coordinator_acceptance_bound_range` | `node target must be a positive 256-bit integer` |
| `training.validation.coordinator_acceptance_bound_value` | `S3 target must be a positive 256-bit integer` |
| `training.validation.coordinator_authentication_rejected` | `S3 rejected authentication` |
| `training.validation.coordinator_authentication_timeout` | `S3 authentication timed out` |
| `training.validation.coordinator_batch_timeout` | `S3 sent no usable training job before the job timeout` |
| `training.validation.coordinator_closed_before_acknowledgement` | `S3 closed before acknowledging the share` |
| `training.validation.coordinator_compression_unsupported` | `S3 compression must be none or gzip` |
| `training.validation.coordinator_config_prefix` | `fill S3.` |
| `training.validation.coordinator_host_config_prefix` | `fill S3.host for a flat configuration, or S3.` |
| `training.validation.coordinator_hostname_suffix` | `.host in the shared configuration, with your S3 hostname or IP address` |
| `training.validation.coordinator_identity_whitespace_template` | `S3.{} and S3.worker must not contain whitespace` |
| `training.validation.coordinator_pending_submission_limit` | `Too many pending S3 submissions` |
| `training.validation.coordinator_protocol_explicit_selection` | `S3.protocol must explicitly select KEX or WR; automatic detection is unavailable` |
| `training.validation.coordinator_protocol_unsupported` | `S3.protocol must select KEX or WR` |
| `training.validation.coordinator_reconnect_requested` | `S3 requested reconnection to the configured endpoint` |
| `training.validation.coordinator_seed_required` | `S7 training requires the S3's eight-byte extranonce` |
| `training.validation.coordinator_seed_update_schema` | `Unsupported S7 extranonce schema` |
| `training.validation.coordinator_session_already_stopped` | `S3 session is stopped` |
| `training.validation.coordinator_session_closed` | `S3 session closed` |
| `training.validation.coordinator_session_not_open` | `S3 session is not open` |
| `training.validation.coordinator_session_opening` | `S3 session is already opening` |
| `training.validation.coordinator_session_stopped` | `S3 session stopped` |
| `training.validation.coordinator_submission_acknowledgement_timeout` | `S3 share acknowledgement timed out` |
| `training.validation.coordinator_submission_attestation_version` | `unsupported S6 certificate version for S3 share` |
| `training.validation.coordinator_submission_rejected` | `S3 rejected share` |
| `training.validation.coordinator_submission_send_failed` | `S3 share send failed` |
| `training.validation.coordinator_submission_verification_prefix` | `"S3 share rejected by official verification: "` |
| `training.validation.coordinator_tile_coordinates_type` | `S3 share tile coordinates must be integers` |
| `training.validation.coordinator_tls_authority_unreadable` | `Could not load RPC certificate authority file` |
| `training.validation.coordinator_tls_url_required` | `A certificate authority file requires an https RPC URL` |
| `training.validation.coordinator_wait_timeout_finite` | `S3 wait timeout must be a finite number` |
| `training.validation.coordinator_wait_timeout_positive` | `S3 wait timeout must be a positive finite number` |
| `training.validation.coordinator_worker_identity_whitespace` | `KEX_uname and worker must contain no whitespace` |
| `training.validation.cpu_attestation_tile_incomplete` | `official CPU reference produced no complete share tile` |
| `training.validation.cpu_batch_tile_incomplete` | `CPU reference produced no complete S7 tile` |
| `training.validation.dense_attestation_canonical_format` | `plain_proof must be nonempty canonical base64` |
| `training.validation.dense_attestation_encoding_invalid` | `invalid plain proof encoding` |
| `training.validation.dense_attestation_input_format` | `plain proof must be a bounded base64 string` |
| `training.validation.dense_attestation_size_limit` | `plain_proof exceeds the 4 MiB decoded size limit` |
| `training.validation.dense_attestation_trailing_data` | `unsupported or trailing plain proof data` |
| `training.validation.dense_model_attestation_required` | `MoE proofs are rejected by the dense-only rule` |
| `training.validation.final_acceptance_bound_range` | `final S3 target must be a positive 256-bit integer` |
| `training.validation.local_attestation_generation_unavailable` | `Local ZK certificate generation is unavailable; WR and plain S3 shares do not require it` |
| `training.validation.local_attestation_verification_unavailable` | `Local ZK certificate verification is unavailable` |
| `training.validation.mixture_of_experts_attestation_unavailable` | `MoE plain proofs are unavailable` |
| `training.validation.tensor_commitment_header_mismatch` | `Merkle proof commitment does not match header and configuration` |
| `training.validation.tensor_commitment_input_empty` | `Merkle data must not be empty` |
| `training.validation.tensor_commitment_leaf_count_mismatch` | `Merkle tree leaf count differs from declared dimensions` |
| `training.validation.tensor_commitment_leaf_count_positive` | `Merkle tree requires a positive leaf count` |
| `training.validation.tensor_commitment_leaf_order` | `Merkle leaf indices must be sorted, unique and in range` |
| `training.validation.tensor_commitment_leaf_range` | `Merkle leaf indices are out of range` |
| `training.validation.tensor_commitment_matrix_coverage` | `Merkle leaves do not cover required matrix bytes` |
| `training.validation.tensor_commitment_slice_range` | `invalid Merkle extraction range` |
| `training.validation.transcript_digest_parameters_invalid` | `invalid transcript hash key or target` |
| `training.validation.worker_identity_type` | `S3.worker must be a string` |
| `training.validation.worker_identity_whitespace_suffix` | `" and S3.worker must not contain whitespace"` |

Text values with leading or trailing spaces are quoted to show the exact
spacing. Raw binary values are shown in hex. RPC fields and methods use the
structural entries selected by the native wire profile. Explicit constant
aliases are resolved by their callers and do not rewrite arbitrary message
content. The aliases describe internal names and preserve the original peer
protocol and proof behavior.

### Updating and distributing mappings

1. Add or change a `make_entry("alias", "decoded value")` row in
   `native/word_codec.cpp`, or use `make_binary_entry("alias", "hex")` for
   binary bytes. Both columns must remain unique. For an explicit constant,
   add the row to `kConstants` and use `FixedAlias` in Python. For a new
   network field or command, also select it in the appropriate `key_entry` or
   `Parser::translation` profile in `native/wire_codec.cpp`. Profiles distinguish
   RPC fields, pool fields, command names and ordinary string data; adding a
   word-table entry alone does not rewrite arbitrary message values.
2. Run `python3 tools/build_word_codec.py`. It validates the table, compiles the
   word and wire libraries, regenerates this table in both READMEs, and atomically
   installs identical libraries and Python bridges. This includes
   `native_text_codec.py`, `wire_io.py` and `rpc_fields.py`. The default standalone
   checkout is the sibling `LRAEP-WR-Trainer`; use `--standalone PATH` to choose
   another checkout or `--local-only` for an author-only build.
3. Run `python3 tools/build_word_codec.py --check` and the protocol/native tests.
   The check detects stale README content, source fingerprints, differing
   binaries, or differing Python bridges. Restart running trainer/notebook
   processes to load a newly installed library; loaded libraries are not
   hot-swapped in place.

The current build targets the host's Linux architecture (`x86_64` or
`aarch64`) and requires a C++17 compiler. The output is
`src/LRAEP_trainer/bin/<platform>/libword_codec.so` and, when distributing,
`src/workrunner/bin/<platform>/libword_codec.so`. The native message layer is
installed alongside it as `libwire_codec.so`. Rebuild on each target
architecture. No compiler or C++ decoder source is needed in the standalone
at runtime. A missing or incompatible library produces an explicit error.

### Buffer lifetime

`word_buffer` and `fixed_buffer` copy values into mutable buffers and clear
the buffers on normal and exception exits. `word_matches` compares ASCII
protocol values with a scoped buffer. The native table retains the original
bytes so later calls can resolve the same alias.

`transform_word`, `decode_word`, and `encode_word` return ordinary immutable
Python strings. Callers must release those references after use; their physical
bytes cannot be guaranteed erased, and serializers, sockets, exception text,
or caller-retained objects may create copies. The framing-only
`customEncoder`/`customDecoder` API implements the peer's frame format.
<!-- native-word-codec:end -->

## CUDA temporary work buffers

The native CUDA Python interface now scopes its privately allocated 32-byte
work-key and target tensors in `_temporary_words`. After their last operation,
cleanup overwrites each tensor on the same CUDA stream and synchronizes that
stream before returning or propagating an error. The complete candidate-capacity
retry loop stays inside this scope. Previously allocated keys are also cleared
when a later allocation fails. `_words` clears its owned CPU staging array after
the explicitly blocking host-to-device copy, including failure paths.

Borrowed matrices, transcripts, returned results, and the caller's CPU key bytes
and target integer are preserved. Those original job inputs remain available
while the job needs them. The added stream synchronization can reduce overlap;
no zero-overhead guarantee is made. If CUDA cannot perform cleanup, a successful
operation becomes an explicit cleanup failure; an existing operation exception
is preserved and receives a cleanup-failure note. Fatal CUDA context errors can
prevent wiping.

This cleanup covers owned tensor allocations and CPU staging. It does not erase
fixed labels embedded in loaded CUDA instructions or guarantee clearing
compiler-managed registers, local spills, or other copies outside these buffers.
