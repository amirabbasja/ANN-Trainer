# LRAEP helper bundle

This directory contains the source dependency subset needed by the custom S5
from the official S6 project, revision
`0798903bced203de8cfcb49441f44fd9b51b18eb`:
https://github.com/pearl-research-labs/pearl/tree/0798903bced203de8cfcb49441f44fd9b51b18eb

The Python subset provides deterministic noise, commitments, independent CPU
transcripts, Merkle proof construction, and the solo block/template serializers.
Gateway servers, RPC clients, async S5 services, node and wallet applications,
web applications, and unrelated tools are omitted.

The original helper package, class, import, and crate identifiers use `LRAEP`.
Additional local names follow the abbreviations below. The underlying arithmetic, proof
formats, network-facing values, and cryptographic domain-separation bytes remain
unchanged. The Python logging environment variable is now `LRAEP_LOG_LEVEL`.
The native module also exposes `__lraep_upstream_revision__` for provenance.

`manifest.json` records original source paths, the upstream revision, and SHA-256
hashes of the vendored files. This source bundle is intended to be distributed
with the app. It does not require upstream Git metadata at runtime.

Original copyright and license notices are retained in `LICENSE` and in the
Rust dependency license files. Names inside those notices and upstream source
URLs intentionally identify their original authors. See the project
`THIRD_PARTY_NOTICES.md` for attribution.


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

These substitutions apply case-insensitively to local names, filenames, imports,
configuration keys, and CLI commands (for example, `pool` becomes `S3` and
`WR_wallet` becomes `WR_S2`, and `miner_base` becomes `S5_base`). Existing README
examples retain the original S1–S4 spellings; substitute those abbreviations
when using them. S5–S7 examples use the renamed spellings; the longer S7 term
is replaced before S6. Fixed protocol names and cryptographic domain bytes are
stored as Base64 in `LRAEP_trainer.wire_constants` and decoded before use,
preserving network and proof bytes. Upstream URLs, endpoint hostnames, source
attribution, and license text retain their required spellings.
