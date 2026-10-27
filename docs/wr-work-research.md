# PearlHash work derivation and validation

The custom trainer implements the observed WR 0.51.8 generation-3 work
format in Python/CUDA. It does not execute or load WR. Its normal PEARL
helpers still supply commitment derivation, sparse noise and transcript rules.
The local checker supplied the wire captures and offline research evidence.

The studied executable has SHA256
`d5124a8698fdf02114a729c7b50011dd637634821d9872947ca600d641cfa38a`.
Its saved, unpacked CPU image is in
`research/codec/snapshot/` within the WR research workspace (see the main README). The native offsets below
are **file offsets in `62700f001000.bin`**; add `0x1000` for offsets from the
original executable mapping. No binary or memory snapshot is a runtime or test
dependency of this trainer.

## Signals and nonces

`sb: 1` is a boolean indicating a nonzero constant signal base, whose observed
value is **64**. Native setup initializes `[state+0x31b4]` to `0x40` at file
`0x13b75f`; `0x1b0adb` checks this value. At `0x17f75d` the submit builder
converts it to a boolean using `test`/`setne`.

Both unmodified signal matrices contain signed byte value 64. A's final
sixteen bytes replace those constant values with this deterministic tail:

```python
raw = blake3(worker_nonce + miner_nonce).digest()[:16]
tail = bytes((value % 129 - 64) & 255 for value in raw)
```

The worker nonce comes from the connection's `extranonce` notification. Each
nonce is exactly eight bytes in the original wire byte order. The mapping's
signed range is `[-64, 64]`.

The native CPU checker begins at `0x13ebf0`. At `0x13ee2b` it constructs
`worker_nonce || miner_nonce`; `0x13ee82` calls the unkeyed BLAKE3 helper
`0x5b936`. The loop at `0x13eea0` through `0x13eec1` maps the first sixteen
digest bytes. The checker appends that tail to a saved matrix prefix at
`0x13eef5` and finalizes the root at `0x13ef0a`. The prefix generator at
`0x1afb40` excludes exactly sixteen bytes at `0x1afb7e`.

An independently emulated native CPU hash and Python BLAKE3 agree on this
captured vector:

```text
worker_nonce = f8d10b2c6c027fee
miner_nonce  = 0200000000000000
BLAKE3       = 15a00bb270d285ad2a35c5faddeb5baeda2f71e67940962648d8ac76622b4236
tail bytes   = d5dfcbf13011c4eceaf504391c2a1bed
```

The custom trainer searches `training.m` rows and commits one additional A
row. It submits `m = training.m + 1`, keeps the nonce tail in the last row,
and excludes that row from every submitted tile. The submitted `n` and `k`
retain their configured values. The extra row is part of the full commitment,
including its version-3 dimension salt.

## Commitments, noise and target

The commitment key is the standard unkeyed BLAKE3 hash of the 76-byte header
followed by the 52-byte mining configuration. Both matrices use row-major
signal bytes, padded with zeros to a 1024-byte chunk boundary before keyed
BLAKE3 hashing. A includes the nonce tail before padding; B is constant 64.

Generation 3 uses the normal PEARL dimension-bound root derivation. Native
salt constants match the bundled helpers exactly: file `0x577080` contains
`blake3(b"pearl/cert-v3/noise-seed/A")`, while `0x57cde0` contains
`blake3(b"pearl/cert-v3/noise-seed/B")`. The B generation-3 branch begins at
`0x1b1232`. The resulting seeds feed the usual rank-128 sparse noise and keyed
transcript hashing.

The notification's target is a 32-byte **little-endian** integer. The trainer
applies the official mining-configuration target adjustment and compares the
final digest as little endian. Reading the wire string as a display-order
hexadecimal integer produces a different, unusable target.

Before submission, `reconstruct_wr_digest()` rebuilds the compact witness
independently on the CPU. It derives its own signals, full roots, noise seeds
and selected noisy rows/columns, then computes the standard transcript. It
checks the GPU's candidate digest and exact adjusted target. It does not trust
GPU buffers or substitute a KEX base64 proof for PearlHash's nonces and
indices.

## Three accepted-share fixtures

The source capture is
`captures/shares-20261026T203339Z-27408/` within that WR workspace.
Its three `submit` requests have matching pool replies with `status: 1` and
literal `result: true`. Portable headers, targets, compact witnesses and
reconstructed digests are retained in
[`tests/fixtures/wr_accepted_work.json`](../tests/fixtures/wr_accepted_work.json).

All three independently reconstructed digests meet the exact target:

| Captured request ID | Reconstructed digest bytes | Little-endian bit length |
|---|---|---:|
| 322 | `98a0952cd90ed99b6f91ba25b2092a45e81122e49a8052416c579a0f00000000` | 220 |
| 445 | `937f560b6b38df5d08b42f6cbcdc52c554834b6aa5614c54535f400900000000` | 220 |
| 464 | `742d57d0e854f1171844218c969f5df6fe7c690c0848b97f81dcbb0100000000` | 217 |

The exact adjusted target has bit length 222. Using signal value 1, raw nonce
bytes, or the opposite nonce order fails to reproduce this accepted work.
These fixtures also exercise WR's observed noncontiguous 8-by-16 tiles;
the trainer searches its configured contiguous tiles.

`tests/test_wr_work.py` checks native-tail agreement, all three accepted
witnesses, full dimension binding, independent candidate verification, invalid
tiles, and CUDA/CPU agreement with reserved rows and chunk padding. Local TCP
integration tests exercise actual custom GPU search and submission. These are
offline and loopback checks; this change did not submit new shares to the real
pool. The evidence supports the captured generation-3 format and does not
establish other generations or unknown future pool messages.
