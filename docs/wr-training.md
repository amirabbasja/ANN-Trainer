# WR/S7 in the custom trainer

Choose `S3.protocol: "WR"` (or `"S7"`) to use the S7
adapter. KEX keeps its own protocol and work path. This option runs the
trainer's Python/CUDA code; it does not invoke the WR executable or load
its libraries. The external WR research workspace supplies evidence and is
not a runtime dependency; its location is documented in the main README.

Use the app's single `config.json`. Set `S3.WR.host`, `S3.WR.port`,
`S3.WR_S2` and `S3.worker`, and run from the trainer directory:

```bash
.venv/bin/python ann_trainer.py S3-login-check --S5 WR
.venv/bin/python ann_trainer.py S3 --S5 WR
```

`S3-login-check` authenticates and validates one generation-3 job and the
connection's eight-byte extranonce. It performs no GPU search or submission.
Use `S3-check` to test TCP/TLS without authentication. The example endpoint
has a blank host. Set your private endpoint explicitly; the captured endpoint
is listed in the main README.
The application codec obfuscates frames; it does not provide TLS encryption.
When an endpoint supports TLS, `S3.WR.tls` enables certificate and hostname
verification through the same transport as KEX.

The CLI `--S5 WR` selects both the WR adapter and `S3.WR` connection
profile from the shared file. WR uses `S3.WR_S2`; KEX uses
`S3.KEX_uname` and `S3.KEX`. VPN, device, training settings and worker name
remain shared. No separate WR configuration file is needed.
KEX's optional gzip proof transport is unavailable on S7.

## GPU count and connected-worker hashrate

WR sends S7's `report_info` message after authentication to register the
selected GPU. It sends the measured work rate after the first completed batch,
then about every 10 seconds during the connection's first two minutes, and
about every 120 seconds afterward, at a batch boundary. Each report contains
one device object with its actual GPU name and a numeric rate in H/s.
For example, a diagnostic rate of 4 TH/s corresponds to about `4000000000000`
H/s in the report. Login-only checks send no GPU reports.

Each report records the rate sent and its request ID at the `DEBUG` log level.
Compare the newest `S7 GPU report sent` diagnostic with the website.
The completed-work average continues changing between reports, and the website
may refresh later.
Early batches can run slower during allocation and GPU warmup; the frequent
startup updates replace that initial low sample promptly.

The rate uses completed matrix work and elapsed wall-clock time. The initial
report can be zero before work has completed. A new connection registers the GPU
again. A connected worker alone
does not provide this device information: without reports the dashboard can
show an online worker with zero GPUs and zero connected-worker hashrate.

Report acknowledgements are tracked separately from share acknowledgements.
They do not increment submitted or accepted share counters. Reported device
rate is distinct from S3 credit, which depends on accepted S1 shares.

The normal terminal row uses training-style progress: `Main | Step 100 |
elapsed=00:02:30 | GPU 42%`. A configured main-batch limit adds a total and
percentage. Scheduled CNN work shows its actual epoch, optimizer steps,
percentage, running loss and accuracy instead. Hashrate and submission counters
stay out of the normal display; detailed counters and successful submission
messages remain available at `DEBUG`. GPU reports and submission behavior are
unchanged, and rejected work still logs a warning.

## Deterministic work

The S7 path builds the captured `sb: 1`, `tail_mode: 1` work: constant
signal value `64`, with the last sixteen bytes of A derived from
`BLAKE3(worker_nonce || S5_nonce)`. Each of those digest bytes maps to
`byte % 129 - 64`. A reserved final row carries the tail and is excluded from
search. Thus `training.m` searchable rows submit dimension `m + 1`; `n` and
`k` keep their configured values. The full matrix dimensions bind the V3
noise seeds before the reserved row is removed from the search view.

The existing trainer supplies commitments, deterministic noise and integer
matrix search. Before submission, an independent CPU path reconstructs the
work from the compact witness, recomputes the selected tile digest and checks
the exact S3 target. Random KEX matrices are never sent as S7
work. The original header and its network difficulty remain unchanged.

## Unknown requests

`S3.stop_on_unknown_request` defaults to `true`. An unfamiliar incoming
method or notification, invalid frame, unsupported job layout or generation
causes the trainer to invalidate its current work, save the received data,
close the S3 connection, cancel further GPU work, and exit with an error.
It does not reconnect automatically after a compatibility failure. A GPU
stage already running finishes before control returns to the cancellation
check; its result cannot be submitted after invalidation.

The terminal prints a critical notice with the saved file's path, and rings
the terminal bell when attached to a terminal. A separate private JSON file
preserves the UTC timestamp, S3 endpoint, diagnostic reason, exact incoming
frame bytes in hex, and decoded JSON when available. No configured password,
S2, outgoing login or proof is added. Remote messages themselves may contain
private data; files are created with mode `0600`, new directories with `0700`.

Set `S3.unknown_request_dir` to choose where to save records. Relative paths
resolve beside the configuration file. With `null`, records go under
`training.output_dir/unknown-S3-requests`. Recording failures also stop the
trainer and print an error. Files use unique names and survive later runs.

Setting `S3.stop_on_unknown_request` to `false` permits unfamiliar extension
methods and named notifications to be recorded, warned about and ignored.
Malformed frames, invalid jobs and unsupported generations still stop the
trainer because they cannot be used to construct work safely.

## Protocol evidence

The independent codec implements the observed hex frame, four-byte seed and
48-byte repeating key. Jobs use the captured two-layer header mask and a
little-endian 32-byte share target. The generation field is not a block height;
only captured generation `3` is supported. Each extranonce replacement
invalidates the previous work and binds subsequent shares to the new nonce.

Submissions contain the S5 nonce, worker nonce, matrix dimensions, rank,
tile size, signal byte and selected row/column indices. They contain no job ID
or base64 proof. Only a matching submission reply with integer `status: 1`,
literal `result: true` and no error counts as accepted. Authentication and
telemetry replies never count as shares.

Offline fixtures preserve captured frames, decoded jobs and three accepted
WR submissions from the October 26–27 capture. The request builder
reproduces those submissions byte for byte. Live accepted shares and payouts
from this custom trainer must be assessed separately; a successful login is
not evidence of credited work.
