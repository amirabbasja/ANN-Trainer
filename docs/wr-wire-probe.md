# Inspect WR's Pearlhash wire exchange

`tools/wr_wire_probe.py` is a **local TCP relay**, not a Pearlhash protocol
implementation. It forwards bytes unchanged between a WR instance you
control and the selected pool, while printing timestamped JSON-line events for
both directions. No packet capture permissions, TLS interception certificate,
or WR source code are needed.

The shared `config.json` stores the WR connection under `pool.WR`.
Verify its host, port and TLS settings before using the commands below.
`config.example.json` is the single public template and leaves hosts blank.

From `/workspace/trainer`:

```bash
# Optional: test that the real endpoint accepts a TCP connection; no login sent.
python3 tools/wr_wire_probe.py \
  --upstream-host <POOL_HOST> --upstream-port 9000 --check-only

# In one terminal, run the relay (bound only to 127.0.0.1:19000).
python3 tools/wr_wire_probe.py \
  --upstream-host <POOL_HOST> --upstream-port 9000
```

Then, **in your existing WR command/config**, change only the pool endpoint
to `stratum+tcp://127.0.0.1:19000` (or `127.0.0.1:19000`, depending on its
configuration format). Keep the wallet/worker/password you already use; don't
copy private credentials into the probe's command line. If WR has a separate
SSL/TLS setting, disable it **only for its local loopback connection**. Leave
WR running to observe subsequent job updates and, if it finds a share,
submission and acknowledgement. Stop the probe with Ctrl+C. Nothing changes in
the trainer's `config.json`.

If the **real** pool endpoint uses TLS, pass `--upstream-tls` (and, if needed,
`--ca-file /path/to/ca.pem`) to **both** probe commands, but keep the WR
connection to the local relay plain TCP. The probe verifies the upstream
certificate and hostname; it does not accept miner-side TLS. The loopback hop
is unencrypted, so do not bind or forward the probe to a public interface.

Each output line contains `connection`, `event`, and `time`. A `json` event also
contains `direction`, `operation`, and `message`. `operation` on a response is
looked up by its request ID; unsolicited server notifications show their own
method. This lets you compare login, job delivery, and share submission without
assuming a particular JSON-RPC dialect. `upstream-connected` proves TCP/TLS
connectivity, **not** successful login or accepted shares. A `non-json` event
means the data was not a newline-delimited JSON message; it shows a byte count
instead of trying to decode unknown binary data. Lines over 8 MiB are omitted
from the display but still forwarded.

By default, common credential fields and positional login parameters are
redacted, the first parameter of a `submit` call is redacted, and long JSON
strings are abbreviated. These are best-effort safeguards: a protocol with
unexpected credential fields may still put identifying data in the output. Do
not publish captures without reviewing them. For an **unredacted** local trace
(including non-JSON bytes as hex), explicitly add `--show-raw` and keep the
output private. `--field-limit N` changes the display truncation length, not
the forwarded data. A pool may send no new job for a while, and WR will
not submit a share until it finds one.
