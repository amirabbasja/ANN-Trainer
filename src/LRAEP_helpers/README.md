# LRAEP helper bundle

This directory contains the source dependency subset needed by the custom miner
from the official PEARL project, revision
`0798903bced203de8cfcb49441f44fd9b51b18eb`:
https://github.com/pearl-research-labs/pearl/tree/0798903bced203de8cfcb49441f44fd9b51b18eb

The Python subset provides deterministic noise, commitments, independent CPU
transcripts, Merkle proof construction, and the solo block/template serializers.
Gateway servers, RPC clients, async miner services, node and wallet applications,
web applications, and unrelated tools are omitted.

Names containing `pearl` in source packages, classes, imports, and local crate
identifiers have been changed to `LRAEP`. The underlying arithmetic, proof
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
