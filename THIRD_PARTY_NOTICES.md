# LRAEP third-party notices

LRAEP bundles a required subset of official PEARL Python helpers, its native
proof binding, and the Rust sources needed to rebuild that binding. The custom
CUDA arithmetic also follows official PEARL reference helper files at commit
`0798903bced203de8cfcb49441f44fd9b51b18eb`. Renamed modules and build paths do
not change the original authorship of those helpers.

The vendored source is under `src/LRAEP_helpers`. Its README, source provenance
records, checksum manifest, and license files describe the included subset.
The PEARL ISC license notice is retained below and in
`src/LRAEP_helpers/LICENSE`. The bundled Plonky2 dependency retains its
MIT/Apache-2.0 license files under `src/LRAEP_helpers/rust/plonky2/`. Rust
registry dependencies retain their own applicable licenses.

ISC License

Copyright (c) 2025-2026 Pearl Research Labs
Copyright (c) 2015-2016 The Decred developers

Permission to use, copy, modify, and distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

## Optional Xray VPN core

The optional VPN feature starts an unmodified official Xray-core executable.
`setup_vpn.py` installs release `v26.10.21` after checking its pinned archive
SHA-256 checksum. The binary is installed outside this project's source tree;
it is not renamed or presented as LRAEP-authored code.

Xray-core is distributed under the Mozilla Public License Version 2.0. The
installer retains the release archive's original license text as
`Xray-LICENSE` beside the executable, by default under
`~/.local/share/LRAEP/bin/`. This project does not modify that license text.

- Source: https://github.com/XTLS/Xray-core/tree/v26.10.21
- Release: https://github.com/XTLS/Xray-core/releases/tag/v26.10.21
- Original license: https://github.com/XTLS/Xray-core/blob/v26.10.21/LICENSE

## Permitted networking references

The custom pool networking code was written independently using published
wire specifications and factual captures. Official PEARL proof helpers and
their proof-system dependencies are bundled as described above. No external
miner's executable, GPU implementation, or networking client implementation
is vendored or required at runtime.

- Kryptex's public gzip and difficulty notes by Max Malysh:
  https://gist.github.com/maxmalysh/eaaf4332dbc5ca99d0a78f24a733fffe/45cad9645bd9cd57655851fa823e1299eb4d1a0a
  and https://gist.github.com/maxmalysh/f4c29f532e650a114b317e234f1393fb/ad76fb08751477ca70f20434025ffdf8cc2dd955.
  No license is stated in the notes; only protocol facts are referenced.
- spark-pearl-miner, Apache-2.0, revision
  `65365633e3bff61b0f29d72109622dd22ff5a379`: protocol documentation and a
  redacted-wallet TLS capture. Header/target bytes are used as factual test
  vectors. https://github.com/xXGuilasXx/spark-pearl-miner/tree/65365633e3bff61b0f29d72109622dd22ff5a379
- pearl-metal-miner, Apache-2.0, revision
  `3d4cc7dadbd1de98ba20e9beb162806a0e70266b`: earlier pool wire survey.
  https://github.com/Breakwater-ecosystem/pearl-metal-miner/tree/3d4cc7dadbd1de98ba20e9beb162806a0e70266b

See `docs/kryptex-protocol-review.md` for the scope and evidentiary limits of
these references.
