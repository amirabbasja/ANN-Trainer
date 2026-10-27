# LRAEP Kryptex wire protocol and implementation evidence

Reviewed October 25, 2026. The user authorized **"Allow protocol references
only"** after the original restriction to PEARL's official helpers. This
permits reading public networking documentation and captured messages; it
does not authorize using an external trainer application. GPU arithmetic and
proof construction continue to use this project's code and official PEARL
helpers. The required upstream subset is now bundled under
`src/LRAEP_helpers` with LRAEP package names, retained attribution, and a source
checksum manifest. The rename leaves network and proof bytes unchanged.

## Pinned references

| Source | Evidence |
|---|---|
| [Kryptex CEO Max Malysh's gzip specification](https://gist.github.com/maxmalysh/eaaf4332dbc5ca99d0a78f24a733fffe/45cad9645bd9cd57655851fa823e1299eb4d1a0a) | Object authorization fields, `type: "v2"` confirmation, gzip wrapper (`wbits=31`). Public note; no license is stated in the note. |
| [Kryptex difficulty note](https://gist.github.com/maxmalysh/f4c29f532e650a114b317e234f1393fb/ad76fb08751477ca70f20434025ffdf8cc2dd955) | `password: "d=N"` or an empty password for the default difficulty. Public note; no license is stated in the note. |
| [Captured Kryptex TLS session](https://github.com/xXGuilasXx/spark-pearl-miner/blob/65365633e3bff61b0f29d72109622dd22ff5a379/tests/fixtures/capture-kryptex-8048-authorize.jsonl) | Redacted login, literal successful auth reply, exact V3 header/target job fields. Repository: Apache-2.0. |
| [Kryptex dialect description](https://github.com/xXGuilasXx/spark-pearl-miner/blob/65365633e3bff61b0f29d72109622dd22ff5a379/docs/protocol/kryptex.md) | Submission object and observed target convention; authors report live accepted shares for their implementation. Their result is not a live validation of this project. Repository: Apache-2.0. |
| [Earlier pool survey](https://github.com/Breakwater-ecosystem/pearl-metal-miner/blob/3d4cc7dadbd1de98ba20e9beb162806a0e70266b/docs/research/2026-08-10-pool-survey.md) | Independent subscribe/authorize/job observations. Its Kryptex share submission was not confirmed live; later evidence above is stronger. Repository: Apache-2.0. |

These are protocol references. Their applications and GPU/proof source were
not used to implement the computation. No third-party client implementation
is vendored. A captured header and target are retained as factual test vectors.

## Implemented exchange

Messages are newline-terminated JSON objects over TCP or verified TLS.

The default uncompressed session sends `mining.subscribe` with `[agent]`,
followed immediately by `mining.authorize` with `["wallet.worker", password]`.
The subscribe request may receive no reply; authorization must not wait for it.

Optional gzip uses one object-form authorization:

```json
{"id":1,"method":"mining.authorize","params":{"wallet":"WALLET.worker","agent":"LRAEP-trainer/0.1","password":"x","type":"v2"}}
```

Only a successful matching reply explicitly confirming `type: "v2"` enables
gzip. Without confirmation, the proof is uncompressed. Session `v1`/`v2`
compression is separate from PEARL's certificate versions.

`mining.notify.params` contains `job_id`, `header`, `height`, `target`, and
`cert_version`. The header is exactly 152 hexadecimal characters representing
the original 76-byte incomplete PEARL header. Its network `nbits` remains
unchanged when processing the easier share target.

The target is a big-endian hexadecimal **base target**. At difficulty 2097152,
the capture contains `00000000000007ff...ffff`, equivalent to
`2^224 / difficulty - 1`. This differs from the Bitcoin pdiff formula by the
factor `65535/65536`. The client preserves the received target and uses the
official PEARL target-adjustment helper; it never recomputes from a job ID or
password. The final jackpot digest is interpreted little endian as required
by PEARL.

Shares use `mining.submit` with object parameters
`{"worker":"wallet.worker","job_id":"...","plain_proof":"BASE64"}`.
`PlainProof.to_base64()` supplies the official native bincode serialization.
Gzip, when negotiated, compresses the decoded native bytes and base64-encodes
the compressed result. No legacy binary rewriting is inferred from the
compression session's name.

The client requires literal `result: true`, no error, and a matching pending
request ID before counting an accepted share. New jobs supersede active work;
connection replacement invalidates work from the preceding session. Unknown
certificate versions fail explicitly; missing versions require a configured
fallback. No fork height is hardcoded.

## Validation limits

This project checks protocol parsing, authentication/compression negotiation,
share encoding, stale-work handling, and acceptance semantics with local tests.
The GPU path is compared with official PEARL reference math and native proof
verification. Transport checks reached Kryptex TCP and verified TLS endpoints.
The user's payout identity is configured locally, and TLS authentication and
live V3 job reception have been confirmed. No live accepted share or credited
user balance has been confirmed. Native CUDA acceleration preserves the same
Kryptex wire adapter; complete proof exchanges are checked with a local pool
whose server verifies proofs using PEARL's official native helper.

Kryptex's [OpenAPI specification](https://pool.kryptex.com/openapi.yaml) is
for monitoring. It is not used as a substitute for the Stratum exchange.
Its [FP8 notice](https://pool.kryptex.com/articles/pearl-fp8-fork-en), updated
October 7, describes a draft with no activation date or block set. Runtime
certificate-version validation remains the compatibility check.
