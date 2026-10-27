# LRAEP GPU trainer

LRAEP is a Python client for the PEARL network with a native CUDA search loop
and a Kryptex pool connection. A bundled subset of PEARL's official
helpers supplies deterministic noise, commitments, Merkle proofs, and proof
verification. The helper packages use LRAEP names; their upstream authorship
and license notices are retained. Solo operation also supports a `pearld`
node. No external trainer application is required or launched.

Here, **neural network training** names the PEARL work loop. The generated
workload is integer matrix multiplication; it does not fit a neural network,
update model weights, or run a model optimizer. Supplying useful model work
would require an additional model integration.

## Configuration file and Kryptex connection

The Python entry point is `ann_trainer.py`. Start the miner from this directory:

```bash
.venv/bin/python ann_trainer.py --config config.json pool
```

It accepts the same configuration, VPN settings, and commands as `LRAEP` and
`python -m LRAEP_trainer`, including `vpn-check`, `pool-login-check`, and `self-test`.

Kryptex job reception and share submission are implemented in this project's
Python client. Pearlhash currently supports TCP/TLS diagnostics only: its
custom job/share protocol remains unknown. To inspect your own WildRig session
against Pearlhash, see [`docs/wildrig-wire-probe.md`](docs/wildrig-wire-probe.md).

Edit `config.json` to set the pool endpoint, payout wallet, worker name,
password, TCP/TLS settings, timeouts, GPU, and matrix sizes. A shareable template
is in `config.example.json`. The local `config.json` is ignored by git because
it may contain account or RPC credentials. Relative paths resolve from the
configuration file's directory. Run the commands below from `/workspace/trainer`;
the configuration file itself may be stored elsewhere.

The [Kryptex PEARL pool page](https://pool.kryptex.com/en/prl) lists
`prl.kryptex.network:7048` for TCP and port `8048` for TLS. The main configuration
uses verified TLS on port `8048`. The portable `config.example.json` uses TCP;
for a TLS template, use `config.tls.example.json`, or set `pool.tls` to `true`
and `pool.port` to `8048`. Certificate and hostname verification remain enabled.

Endpoint configurations are available for Kryptex's global and Hong Kong servers:

| Configuration | Endpoint | TLS |
|---|---|---|
| Local `config.json` | `prl.kryptex.network:8048` | Yes |
| `config.example.json` | `prl.kryptex.network:7048` | No |
| `config.tls.example.json` | `prl.kryptex.network:8048` | Yes |
| `config.hk.example.json` | `prl-hk.kryptex.network:7048` | No |
| `config.hk-tls.example.json` | `prl-hk.kryptex.network:8048` | Yes |
| `config.global.example.json` | `prl.kryptex.network:7048` | No |

Choose a configuration with `--config` and fill in `pool.wallet` and
`pool.worker`. Keep the wallet address and worker label separate; the client
constructs the pool login. `config.kryptex.example.json` selects verified TLS
and the Kryptex protocol explicitly. The host contains only the hostname;
the `tls` and `port` fields represent `stratum+ssl://` connection settings.

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
PEARL's official share-target adjustment, verifies candidate proofs, and submits
them to the pool. A matching reply with literal `result: true` and no error is
counted as accepted. A missing reply is not counted as acceptance.

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

Pool protocol options:

| Field | Default | Meaning |
|---|---|---|
| `protocol` | `"auto"` | Select the known host's adapter, or explicitly select `"kryptex"` or `"pearlhash"`. Pearlhash training is unsupported. |
| `compression` | `"none"` | Native uncompressed proof bytes; `"gzip"` requests the pool's optional gzip session. |
| `certificate_version` | `null` | Use the explicit job field. An integer `1`, `2`, or `3` is a fallback only when a job omits it. Unknown versions fail. |
| `agent` | `"LRAEP-trainer/0.1"` | Client identifier sent during authentication. |

Gzip is used only after the pool confirms `type: "v2"` in a successful
authentication reply. Otherwise proofs remain uncompressed. This session
version does not change PEARL's native proof serialization. The pool's
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

Install the pinned core once, paste your share link into `vpn.link` in
`config.json`, and set `vpn.enabled` to `true`:

```bash
.venv/bin/python setup_vpn.py
.venv/bin/LRAEP --config config.json pool
```

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

Requirements: Python 3.12+, stable Rust/cargo, a C build toolchain, an NVIDIA
CUDA GPU and driver, and CUDA-enabled PyTorch. Native kernels additionally
require Linux, a compatible CUDA toolkit containing `nvcc`, and compute capability
6.1 or later. Default matrices fit a 4 GB card. The native backend has been
checked on a GTX 1650 (SM75). AMD/ROCm and
Apple GPU operation have not been implemented or tested.

```bash
python setup_helpers.py
.venv/bin/python -m LRAEP_trainer self-test
```

The setup script builds the bundled Rust helper at
`src/LRAEP_helpers/rust/py-LRAEP-mining/Cargo.toml` with
`maturin --release --locked`, installs its renamed `LRAEP_mining` extension,
and installs this app in editable mode. It does not clone or fetch an upstream
Git checkout. Python packages and Rust registry dependencies may be downloaded
during the build. Other packages in `requirements.txt` are general Python
libraries. An existing CUDA-enabled system PyTorch installation is reused by
the local virtual environment. If PyTorch reports CUDA unavailable, install
a CUDA-enabled PyTorch build before running the self-test.

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
`docs/kryptex-protocol-review.md`.

- [PEARL protocol website](https://pearlresearch.ai/network)
- [Official repository and node setup](https://github.com/pearl-research-labs/pearl/tree/0798903bced203de8cfcb49441f44fd9b51b18eb)
- [Python proof helper](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/py-pearl-mining/README.md)
- [Reference noisy matrix arithmetic and transcript](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/miner/miner-base/src/miner_base/noisy_gemm.py)
- [Commitments and V3 salts](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/miner/miner-base/src/miner_base/commitment_hash.py)
- [Consensus limits and target adjustment](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/zk-pow/src/api/sanity_checks.rs)
- [Official block proof construction](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/miner/pearl-gateway/src/pearl_gateway/proof_generator.py)

The bundled helpers retain their upstream license notices in
`src/LRAEP_helpers/`, including the PEARL ISC notice and the proof-system
dependency licenses. See `THIRD_PARTY_NOTICES.md` and
`src/LRAEP_helpers/README.md` for attribution and vendoring details.
