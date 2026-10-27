# WR/PearlHash in the custom trainer

Choose `pool.protocol: "WR"` (or `"pearlhash"`) to use the PearlHash
adapter. KEX keeps its own protocol and work path. This option runs the
trainer's Python/CUDA code; it does not invoke the WR executable or load
its libraries. The external WR research workspace supplies evidence and is
not a runtime dependency; its location is documented in the main README.

Use the app's single `config.json`. Set `pool.WR.host`, `pool.WR.port`,
`pool.WR_wallet` and `pool.worker`, and run from the trainer directory:

```bash
.venv/bin/python ann_trainer.py pool-login-check --miner WR
.venv/bin/python ann_trainer.py pool --miner WR
```

`pool-login-check` authenticates and validates one generation-3 job and the
connection's eight-byte extranonce. It performs no GPU search or submission.
Use `pool-check` to test TCP/TLS without authentication. The example endpoint
has a blank host. Set your private endpoint explicitly; the captured endpoint
is listed in the main README.
The application codec obfuscates frames; it does not provide TLS encryption.
When an endpoint supports TLS, `pool.WR.tls` enables certificate and hostname
verification through the same transport as KEX.

The CLI `--miner WR` selects both the WR adapter and `pool.WR` connection
profile from the shared file. WR uses `pool.WR_wallet`; KEX uses
`pool.KEX_uname` and `pool.KEX`. VPN, device, training settings and worker name
remain shared. No separate WR configuration file is needed.
KEX's optional gzip proof transport is unavailable on PearlHash.

## GPU count and connected-worker hashrate

WR sends PearlHash's `report_info` message after authentication to register the
selected GPU. It sends the measured work rate after the first completed batch,
then about every 10 seconds during the connection's first two minutes, and
about every 120 seconds afterward, at a batch boundary. Each report contains
one device object with its actual GPU name and a numeric rate in H/s.
For example, a console rate of 4 TH/s corresponds to about `4000000000000`
H/s in the report. Login-only checks send no GPU reports.

Each report prints the rate sent and its request ID in the terminal. Compare
the newest `PearlHash GPU report sent` line with the website. The live console
average continues changing between reports, and the website may refresh later.
Early batches can run slower during allocation and GPU warmup; the frequent
startup updates replace that initial low sample promptly.

The rate uses the same completed matrix work and elapsed wall-clock time as
the console's local average. The initial report can be zero before work has
completed. A new connection registers the GPU again. A connected worker alone
does not provide this device information: without reports the dashboard can
show an online worker with zero GPUs and zero connected-worker hashrate.

Report acknowledgements are tracked separately from share acknowledgements.
They do not increment submitted or accepted share counters. Reported device
rate is distinct from pool credit, which depends on accepted mining shares.

## Deterministic work

The PearlHash path builds the captured `sb: 1`, `tail_mode: 1` work: constant
signal value `64`, with the last sixteen bytes of A derived from
`BLAKE3(worker_nonce || miner_nonce)`. Each of those digest bytes maps to
`byte % 129 - 64`. A reserved final row carries the tail and is excluded from
search. Thus `training.m` searchable rows submit dimension `m + 1`; `n` and
`k` keep their configured values. The full matrix dimensions bind the V3
noise seeds before the reserved row is removed from the search view.

The existing trainer supplies commitments, deterministic noise and integer
matrix search. Before submission, an independent CPU path reconstructs the
work from the compact witness, recomputes the selected tile digest and checks
the exact pool target. Random KEX matrices are never sent as PearlHash
work. The original header and its network difficulty remain unchanged.

## Unknown requests

`pool.stop_on_unknown_request` defaults to `true`. An unfamiliar incoming
method or notification, invalid frame, unsupported job layout or generation
causes the trainer to invalidate its current work, save the received data,
close the pool connection, cancel further GPU work, and exit with an error.
It does not reconnect automatically after a compatibility failure. A GPU
stage already running finishes before control returns to the cancellation
check; its result cannot be submitted after invalidation.

The terminal prints a critical notice with the saved file's path, and rings
the terminal bell when attached to a terminal. A separate private JSON file
preserves the UTC timestamp, pool endpoint, diagnostic reason, exact incoming
frame bytes in hex, and decoded JSON when available. No configured password,
wallet, outgoing login or proof is added. Remote messages themselves may contain
private data; files are created with mode `0600`, new directories with `0700`.

Set `pool.unknown_request_dir` to choose where to save records. Relative paths
resolve beside the configuration file. With `null`, records go under
`training.output_dir/unknown-pool-requests`. Recording failures also stop the
trainer and print an error. Files use unique names and survive later runs.

Setting `pool.stop_on_unknown_request` to `false` permits unfamiliar extension
methods and named notifications to be recorded, warned about and ignored.
Malformed frames, invalid jobs and unsupported generations still stop the
trainer because they cannot be used to construct work safely.

## Protocol evidence

The independent codec implements the observed hex frame, four-byte seed and
48-byte repeating key. Jobs use the captured two-layer header mask and a
little-endian 32-byte share target. The generation field is not a block height;
only captured generation `3` is supported. Each extranonce replacement
invalidates the previous work and binds subsequent shares to the new nonce.

Submissions contain the miner nonce, worker nonce, matrix dimensions, rank,
tile size, signal byte and selected row/column indices. They contain no job ID
or base64 proof. Only a matching submission reply with integer `status: 1`,
literal `result: true` and no error counts as accepted. Authentication and
telemetry replies never count as shares.

Offline fixtures preserve captured frames, decoded jobs and three accepted
WR submissions from the October 26–27 capture. The request builder
reproduces those submissions byte for byte. Live accepted shares and payouts
from this custom trainer must be assessed separately; a successful login is
not evidence of credited work.
