# Custom PEARL GPU neural network trainer

Python client with its own PyTorch GPU search loop and direct Kryptex pool
connection. PEARL's official helpers supply deterministic noise, commitments,
Merkle proofs, and proof verification. Solo operation also supports a `pearld`
node. No external trainer application is required or launched.

Here, **neural network training** names the PEARL work loop. The generated
workload is integer matrix multiplication; it does not fit a neural network,
update model weights, or run a model optimizer. Supplying useful model work
would require an additional model integration.

## Configuration file and Kryptex connection

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
uses TCP. For TLS, use `config.tls.example.json`, or set `pool.tls` to `true`
and `pool.port` to `8048`. Certificate and hostname verification remain enabled.

Endpoint configurations are available for Kryptex's global and Hong Kong servers:

| Configuration | Endpoint | TLS |
|---|---|---|
| `config.json` / `config.example.json` | `prl.kryptex.network:7048` | No |
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
/workspace/trainer/.venv/bin/python -m pearl_trainer --config /workspace/trainer/config.json pool-check
/workspace/trainer/.venv/bin/python -m pearl_trainer --config /workspace/trainer/config.json pool-login-check
```

`pool-check` opens and closes the configured TCP/TLS connection. For Kryptex,
`pool-login-check` authenticates with your configured identity and receives one
job; it performs no GPU work and submits no shares. Then start the custom GPU
client with:

```bash
/workspace/trainer/.venv/bin/python -m pearl_trainer --config /workspace/trainer/config.json pool
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
| `agent` | `"custom-pearl-trainer/0.1"` | Client identifier sent during authentication. |

Gzip is used only after the pool confirms `type: "v2"` in a successful
authentication reply. Otherwise proofs remain uncompressed. This session
version does not change PEARL's native proof serialization. The pool's
`protocol_documentation` setting records a reference; it is never executed.

Kryptex uses static difficulty `2097152` by default. Optional password `d=N`
selects a value from `2097152` through `4294967296`. The actual share target
comes from each job; the client does not reconstruct it from this password.
The implemented dot-separated login follows captured Kryptex wire messages.

The payout wallet is blank in the shareable templates. Mock-pool tests and
offline GPU/proof checks establish local behavior; authentication, credited
shares, and payouts on your real Kryptex account still require your configured
wallet and a live run. No live share acceptance is claimed here.

To use the existing solo trainer entirely from the configuration file, set
`mode` to `"solo"` and fill in `node.username`, `node.password`, `node.address`,
and the node URL/CA fields. Then run:

```bash
/workspace/trainer/.venv/bin/python -m pearl_trainer --config /workspace/trainer/config.json
```

An explicit `train` command selects solo training. Existing command-line
options override corresponding configuration values. With `--config`, node
credentials come from the file; without it, the existing environment variables
continue to work. No default payout address is included.

## Ready in this workspace

The local `.venv` and pinned `.official/pearl` helper checkout are installed.
The pool client automatically selects the native CUDA DP4A backend for 8×8
hash tiles. Startup reports `Mining backend: native CUDA DP4A`. The first run
builds this project's CUDA sources with `nvcc`; later runs reuse `.native/`.
No external miner executable is loaded. If CUDA compilation is unavailable,
startup explains the failure and selects the PyTorch reference backend.

From `/workspace/trainer`, run the offline GPU check:

```bash
.venv/bin/python -m pearl_trainer self-test
```

To also create and verify a complete local V3 ZK certificate:

```bash
.venv/bin/python -m pearl_trainer self-test --zk
```

Self-tests use synthetic headers and targets. They make no network requests.

## Set up on another machine

Requirements: Python 3.12+, git, stable Rust/cargo, a C build toolchain, an NVIDIA
CUDA GPU and driver, and CUDA-enabled PyTorch. Native kernels additionally
require Linux, a compatible CUDA toolkit containing `nvcc`, and compute capability
6.1 or later. Default matrices fit a 4 GB card. The native backend has been
checked on a GTX 1650 (SM75). AMD/ROCm and
Apple GPU operation have not been implemented or tested.

```bash
python setup_helpers.py
.venv/bin/python -m pearl_trainer self-test
```

The setup script fetches **only** the pinned official repository and builds its
`py-pearl-mining` helper with `maturin --release --locked`. Other packages in
`requirements.txt` are general Python libraries. An existing CUDA-enabled
system PyTorch installation is reused by the local virtual environment. If
PyTorch reports CUDA unavailable, install a CUDA-enabled PyTorch build before
running the self-test.

No `pearl-gateway` service, vLLM application, or downloaded trainer executable
is started. Selected official Python modules are imported as helper files.

## Train against your node

First run a synced official PEARL node and obtain your own PEARL Taproot payout
address. See the official repository's node and wallet setup instructions.
This client does not create a wallet, handle private keys, or start a node.

Set the RPC credentials and payout address. Use your actual values in place of
the placeholders:

```bash
export PEARLD_RPC_URL='https://127.0.0.1:44107'
export PEARLD_RPC_USER='your-rpc-user'
export PEARLD_RPC_PASSWORD='your-rpc-password'
export PEARLD_RPC_CA='/absolute/path/to/your/pearld/rpc.cert'
export PEARL_TRAINING_ADDRESS='your-prl1p-taproot-address'

.venv/bin/python -m pearl_trainer train
```

HTTPS verifies the node certificate. For a localhost node explicitly configured
with `--notls`, set `PEARLD_RPC_URL='http://127.0.0.1:44107'` and unset
`PEARLD_RPC_CA`. The RPC credential variables match PEARL's official examples.

Useful options:

```bash
# Bound the session to 10 complete matrix batches.
.venv/bin/python -m pearl_trainer train --max-batches 10

# Stop starting new work after 60 seconds.
.venv/bin/python -m pearl_trainer train --max-seconds 60

# Verify and save any certified result, without submitting it.
.venv/bin/python -m pearl_trainer train --max-batches 10 --no-submit

# Select another CUDA device; global options precede the subcommand.
.venv/bin/python -m pearl_trainer --device cuda:1 train --m 512 --n 512
```

`Ctrl+C` stops work between slabs. The client polls the chain tip and discards
obsolete work, refreshes templates periodically, verifies every candidate with
the official verifier, creates its ZK certificate, then calls `submitblock`.
Certified blocks and their plain proof metadata are saved under `artifacts/`.
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
.venv/bin/python -m pearl_trainer self-test
```

The unit tests compare GPU transcripts against an independent integer oracle,
including negative values and transcript slot reuse. Node tests exercise RPC
requests, certificate handling, rejection handling, and response errors.
The offline self-test compares against PEARL's official CPU reference and
verifies real plain proofs for all three certificate versions.

## Sources and permitted protocol references

Matrix arithmetic and proof helpers are pinned to PEARL commit
`0798903bced203de8cfcb49441f44fd9b51b18eb`. The helper checkout must remain clean.
The user additionally authorized **"Allow protocol references only"** for
implementing pool networking. Public wire captures and Kryptex's published gzip
note inform the custom client; no third-party GPU/proof algorithm or executable
is used. Exact protocol references and limits are recorded in
`docs/kryptex-protocol-review.md`.

- [PEARL protocol website](https://pearlresearch.ai/network)
- [Official repository and node setup](https://github.com/pearl-research-labs/pearl/tree/0798903bced203de8cfcb49441f44fd9b51b18eb)
- [Python proof helper](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/py-pearl-mining/README.md)
- [Reference noisy matrix arithmetic and transcript](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/miner/miner-base/src/miner_base/noisy_gemm.py)
- [Commitments and V3 salts](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/miner/miner-base/src/miner_base/commitment_hash.py)
- [Consensus limits and target adjustment](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/zk-pow/src/api/sanity_checks.rs)
- [Official block proof construction](https://github.com/pearl-research-labs/pearl/blob/0798903bced203de8cfcb49441f44fd9b51b18eb/miner/pearl-gateway/src/pearl_gateway/proof_generator.py)

The official checkout retains its upstream license notices. See
`THIRD_PARTY_NOTICES.md` for attribution of the reference arithmetic.
