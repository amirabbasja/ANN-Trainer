# LRAEP client for Kryptex and Pearlhash diagnostics

The LRAEP Python client implements Kryptex authentication, incoming jobs,
GPU work, official PEARL proof verification, and share submission. It uses
Python libraries and a renamed subset of PEARL's official helper files bundled
in `src/LRAEP_helpers`. Their original license and provenance notices remain
included. It launches no external
trainer application. Pearlhash training remains unsupported because its
custom job/share wire schema has not been established.

## Configure and run Kryptex

Edit the existing `config.json`, setting `pool.wallet` to your public PEARL
payout address and `pool.worker` to your worker name. The complete TLS template
is `config.kryptex.example.json`; generic and regional templates are also
available. Keep credentials in your personal configuration rather than the
tracked examples.

Use `pool.protocol: "kryptex"`, or keep `"auto"` for a recognized Kryptex host.
The main endpoints are `prl.kryptex.network:7048` for TCP and port `8048` for
TLS. TLS verifies the server certificate and hostname. `pool.ca_file` can
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
- `pool-login-check`: authenticates and receives one Kryptex job, with no GPU
  work or share submission.
- `pool`: receives jobs and submits verified results using the custom GPU loop.

Pool mode does not require a full node or node RPC credentials. The wallet
field is the public payout identity. The client formats the login using the
captured Kryptex `wallet.worker` convention.

## Protocol options

`pool.protocol` accepts `auto`, `kryptex`, or `pearlhash`. Selecting Pearlhash
does not enable its missing training adapter; `pool-check` is still usable.

`pool.compression` defaults to `none`. Set it to `gzip` to request Kryptex's
published gzip session. Only a successful authentication reply explicitly
confirming `type: "v2"` enables gzip. This changes transport compression,
not the native PEARL PlainProof structure.

`pool.certificate_version` defaults to `null`, requiring the pool's job field.
A configured `1`, `2`, or `3` is a fallback for a missing field. An explicit
unknown job version is rejected. No activation height is hardcoded.

`pool.agent` defaults to `LRAEP-trainer/0.1`. The configuration's
`protocol_documentation` value is informational and is never run as code.

Kryptex's default/minimum static difficulty is `2097152`. A password `d=N`
requests a custom value up to `4294967296`. The client always uses the target
in the job and applies the official matrix-work/rank adjustment.

## Evidence and remaining limits

The user authorized reading published protocol references only for pool
networking. Networking was implemented independently from the permitted
message specifications and captures listed in `kryptex-protocol-review.md`.
Official PEARL proof helpers and their required dependencies are bundled;
no external miner application, GPU implementation, or networking client is used.

Mock sessions and offline official proof checks do not establish live account
credit. Shareable templates leave the wallet empty. The local user's wallet
is configured, and verified TLS authentication and live V3 job reception have
been confirmed. No live accepted share, credited balance, or payout has been
confirmed. A successful TCP check alone does not establish those results.

For Pearlhash, its operator must still provide the login, job, target,
certificate-version, and proof-submission formats. Its dashboard statistics
and account lookup APIs do not define those messages. Kryptex's adapter is
not assumed compatible with Pearlhash.
