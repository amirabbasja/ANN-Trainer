# LRAEP client for KEX and PearlHash

The LRAEP Python client implements KEX authentication, incoming jobs,
GPU work, official PEARL proof verification, and share submission. It uses
Python libraries and a renamed subset of PEARL's official helper files bundled
in `src/LRAEP_helpers`. Their original license and provenance notices remain
included. It launches no external
trainer application. The WR/PearlHash adapter uses the independently
recovered exchange in the external WR research workspace; see [its training guide](wr-training.md).

## Configure and run KEX

Edit the existing `config.json`, setting `pool.KEX_uname` to your KEX Mining
Username or public PEARL payout address, and `pool.worker` to your worker name.
The single `config.example.json` template includes both KEX and WR profiles.
Keep credentials in the private shared `config.json`.

Use `pool.protocol: "KEX"` and supply `pool.KEX.host` in your private configuration.
Public templates leave the host blank; reference endpoints are documented in
the main README. TLS verifies the server certificate and hostname. `pool.KEX.ca_file` can
specify an operator-provided certificate authority; certificate verification
is never disabled.

Run from `/workspace/trainer`:

```bash
.venv/bin/python -m LRAEP_trainer --config config.json self-test
.venv/bin/python -m LRAEP_trainer --config config.json pool-check
.venv/bin/python -m LRAEP_trainer --config config.json pool-login-check
.venv/bin/python -m LRAEP_trainer --config config.json pool
```

- `self-test`: offline GPU arithmetic and official proof checks.
- `pool-check`: opens and closes TCP/TLS, with no authentication or GPU work.
- `pool-login-check`: authenticates and receives one KEX job, with no GPU
  work or share submission.
- `pool`: receives jobs and submits verified results using the custom GPU loop.

Pool mode does not require a full node or node RPC credentials. `KEX_uname`
is the KEX payout identity; `WR_wallet` is the separate WR payout address.
The client formats KEX login as `KEX_uname.worker` and sends the WR address
and worker as separate login parameters. Only the selected identity is used.

## Protocol options

`pool.protocol` accepts `KEX`, `WR`, or `pearlhash` (an alias for `WR`).
The protocol is explicit and is not inferred from the host. The CLI
`pool --miner KEX` or `pool --miner WR` selects that adapter's connection
profile and identity from the same `config.json`, found automatically.

`pool.KEX.compression` defaults to `none`. Set it to `gzip` to request KEX's
published gzip session. Only a successful authentication reply explicitly
confirming `type: "v2"` enables gzip. This changes transport compression,
not the native PEARL PlainProof structure.

`pool.KEX.certificate_version` defaults to `null`, requiring the pool's job field.
A configured `1`, `2`, or `3` is a fallback for a missing field. An explicit
unknown job version is rejected. No activation height is hardcoded.

`pool.agent` defaults to `LRAEP-trainer/0.1`. The configuration's
`protocol_documentation` value is informational and is never run as code.

KEX's default/minimum static difficulty is `2097152`. A password `d=N`
requests a custom value up to `4294967296`. The client always uses the target
in the job and applies the official matrix-work/rank adjustment.

## Evidence and remaining limits

The user authorized reading published protocol references only for pool
networking. Networking was implemented independently from the permitted
message specifications and captures listed in `kex-protocol-review.md`.
Official PEARL proof helpers and their required dependencies are bundled;
no external miner application, GPU implementation, or networking client is used.

Mock sessions and offline official proof checks do not establish live account
credit. Shareable templates leave the wallet empty. The local user's wallet
is configured, and verified TLS authentication and live V3 job reception have
been confirmed. No live accepted share, credited balance, or payout has been
confirmed. A successful TCP check alone does not establish those results.

PearlHash uses encoded JSON frames, `login`, `extranonce` and `notify` messages,
and compact nonce/index submissions rather than KEX proofs. Captured
WR generation 3 is supported; unknown versions stop with a diagnostic
record. Evidence for the framing and three pool-accepted share formats is
bundled in the tests. These captures belong to WR, so they do not by
themselves establish live credit for this custom engine.
