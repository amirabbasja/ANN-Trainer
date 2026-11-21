#!/usr/bin/env python3
"""Build native word/wire codec artifacts and synchronize their documentation."""

import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'native' / 'word_codec.cpp'
WIRE_SOURCE = ROOT / 'native' / 'wire_codec.cpp'
WRAPPER = ROOT / 'src' / 'LRAEP_trainer' / 'word_codec.py'
README = ROOT / 'README.md'
START = '<!-- native-word-codec:start -->'
END = '<!-- native-word-codec:end -->'
ENTRY = re.compile(r'(make_base64_entry|make_base64_binary_entry)\(\s*("(?:[^"\\]|\\.)*")\s*,\s*("(?:[^"\\]|\\.)*")\s*\)')


def encoded_entry(alias, value, binary=False):
    raw = bytes.fromhex(value) if binary else value.encode('utf-8')
    if not alias or len(alias.encode()) > 256 or len(raw) > 256:
        raise ValueError('mapping aliases must be nonempty and all fields must fit in 256 bytes')
    kind = 'make_base64_binary_entry' if binary else 'make_base64_entry'
    encoded = base64.b64encode(raw).decode('ascii')
    return f'{kind}({json.dumps(alias)}, "{encoded}"),'


def entries():
    rows = []
    for kind, left, right in ENTRY.findall(SOURCE.read_text()):
        alias, encoded = json.loads(left), json.loads(right).encode('ascii')
        raw = base64.b64decode(encoded, validate=True)
        if base64.b64encode(raw) != encoded:
            raise ValueError('noncanonical Base64 value: ' + alias)
        value = raw if kind == 'make_base64_binary_entry' else raw.decode('utf-8')
        rows.append((alias, value))
    if not rows:
        raise ValueError('native source contains no Base64 word codec entries')
    originals = [value.encode() if isinstance(value, str) else value for _, value in rows]
    if len({alias for alias, _ in rows}) != len(rows) or len(set(originals)) != len(rows):
        raise ValueError('placeholders and decoded values must each be unique')
    if any(not alias or max(len(alias.encode()), len(value)) > 256 for (alias, _), value in zip(rows, originals)):
        raise ValueError('mapping aliases must be nonempty and all fields must fit in 256 bytes')
    return rows


def platform_directory():
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine)
    if platform.system() != 'Linux' or machine not in ('x86_64', 'aarch64'):
        raise ValueError('build on Linux x86_64 or Linux aarch64 for that target platform')
    return 'linux-' + machine


def table_code(value):
    fence = '`' * (1 + max((len(run) for run in re.findall(r'`+', value)), default=0))
    escaped = value.replace('|', '\\|')
    return f'{fence} {escaped} {fence}' if value.startswith('`') or value.endswith('`') else f'{fence}{escaped}{fence}'


def decoded_cell(value):
    if isinstance(value, bytes):
        return f'{table_code(value.hex())} ({len(value)} bytes, hex)'
    return table_code(json.dumps(value) if not value or value != value.strip() else value)


def documentation(rows):
    def base64_cell(value):
        raw = value.encode('utf-8') if isinstance(value, str) else value
        encoded = base64.b64encode(raw).decode('ascii')
        return table_code(encoded[:12] + '…' if len(encoded) > 12 else encoded or '\"\"')
    table = '\n'.join(f'| {table_code(alias)} | {decoded_cell(value)} | {base64_cell(value)} |' for alias, value in rows)
    return f'''{START}
## Native word codec

ANN-Trainer's `native/word_codec.cpp` defines the exact aliases used
by both trainers and production. Training-related names identify internal constants. The
codec resolves them to the original text or bytes wherever the peer protocol,
proof format, diagnostics, or regression fixtures require those values.
Each C++ table row contains exactly two fields: the training alias and its
standard padded Base64 value. On an alias lookup, C++ decodes that Base64 inside
native memory and returns the original text or binary bytes.
Encoding compares the original bytes against a temporary decoded value and
returns its alias. The Python API and the peer's wire format stay unchanged.
Base64 is reversible encoding; the original values are documented below.

The native wire encoder/decoder also translates only the selected JSON fields
and methods and handles the peer's frame format. It uses the same encoded table;
ordinary message strings are preserved.

```python
from LRAEP_trainer.word_codec import FixedAlias, encode_fixed, fixed_buffer, transform_word

transform_word("legacy_protocol", "decode")
transform_word("pearlhash", "encode")
assert encode_fixed(b"target") == FixedAlias("training.field.acceptance_bound")
with fixed_buffer(FixedAlias("training.field.acceptance_bound")) as temporary:
    consume(temporary)
```

The corresponding standalone import is `workrunner.native_text_codec`.
`decode_word` and `encode_word` match structural protocol aliases exactly and
pass unlisted text through unchanged. `encode_fixed` and `fixed_buffer` with
a `FixedAlias` perform exact lookup across the full table. An unknown fixed
alias or value raises `ValueError`. Fixed values can include arbitrary binary
bytes. The codec performs no substring replacement.

### Placeholder table (decode direction)

| Training alias | Decoded value | Base64 encoded value (preview) |
| --- | --- | --- |
{table}

Text values with leading or trailing spaces are quoted to show the exact
spacing. Raw binary values are shown in hex. Base64 previews show at most
12 characters followed by `…` when shortened; they preview the Base64
stored in the native table. Shortened previews cannot be used as decoder inputs. RPC fields and methods use the
structural entries selected by the native wire profile. Explicit constant
aliases are resolved by their callers and do not rewrite arbitrary message
content. The aliases describe internal names and preserve the original peer
protocol and proof behavior.

### Updating and distributing mappings

1. In ANN-Trainer, run `python3 tools/build_word_codec.py --encode-value ALIAS VALUE`
   to print an alias/Base64 row; add `--binary` when VALUE is hexadecimal
   binary data. Add the resulting `make_base64_entry` or
   `make_base64_binary_entry` row to `native/word_codec.cpp`.
   Aliases and original values must each remain unique. For an explicit constant,
   add the row to `kConstants` and use `FixedAlias` in Python. For a new
   network field or command, also select it in the appropriate `key_entry` or
   `Parser::translation` profile in `native/wire_codec.cpp`. Profiles distinguish
   RPC fields, pool fields, command names and ordinary string data; adding a
   word-table entry alone does not rewrite arbitrary message values.
2. Run `python3 tools/build_word_codec.py`. It validates the table, compiles the
   word and wire libraries, regenerates this table in all three READMEs, and atomically
   installs identical libraries and Python bridges. This includes
   `native_text_codec.py`, `wire_io.py` and `rpc_fields.py`. The default standalone
   checkout is the sibling `WR-trainer` (or `LRAEP-WR-Trainer` when present);
   the production checkout is the sibling `Production-LREAP` when present.
   Use `--standalone PATH` and `--production PATH` to choose other checkouts,
   or `--local-only` for an author-only build. Only compiled artifacts and
   Python bridges are copied; native C++ source stays in the trainer checkouts.
3. Run `python3 tools/build_word_codec.py --check` and the protocol/native tests.
   The check detects stale README content, source fingerprints, differing
   binaries, or differing Python bridges. Restart running trainer/notebook
   processes to load a newly installed library; loaded libraries are not
   hot-swapped in place.

The current build targets the host's Linux architecture (`x86_64` or
`aarch64`) and requires a C++17 compiler. The output is
`src/LRAEP_trainer/bin/<platform>/libword_codec.so` and, when distributing,
`src/workrunner/bin/<platform>/libword_codec.so`. The native message layer is
installed alongside it as `libwire_codec.so`. Rebuild on each target
architecture. No compiler or C++ decoder source is needed in the standalone
at runtime. A missing or incompatible library produces an explicit error.

### Buffer lifetime

`word_buffer` and `fixed_buffer` copy values into mutable buffers and clear
the buffers on normal and exception exits. `word_matches` compares ASCII
protocol values with a scoped buffer. Native Base64 and decoded comparison
buffers are also cleared immediately on success and exception exits with volatile
writes, including wire/frame temporaries. The native table retains
only the Base64 bytes; plaintext is recovered separately for each call.

`transform_word`, `decode_word`, and `encode_word` return ordinary immutable
Python strings. Callers must release those references after use; their physical
bytes cannot be guaranteed erased, and serializers, sockets, exception text,
or caller-retained objects may create copies. The framing-only
`customEncoder`/`customDecoder` API implements the peer's frame format.
{END}'''


def expected_readme(rows, path):
    document = path.read_text() if path.exists() else ''
    section = documentation(rows)
    if START in document or END in document:
        if document.count(START) != 1 or document.count(END) != 1:
            raise ValueError('README codec section markers are invalid')
        left, rest = document.split(START, 1)
        _, right = rest.split(END, 1)
        return left + section + right
    return '\n\n'.join(part for part in (document.rstrip(), section) if part) + '\n'


def atomic_write(path, content, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def artifact_source_id(path, prefix='wr_codec'):
    library = ctypes.CDLL(str(path))
    version = getattr(library, prefix + '_abi_version')
    fingerprint = getattr(library, prefix + '_source_id')
    version.restype = ctypes.c_uint
    fingerprint.restype = ctypes.c_char_p
    expected_abi = 3 if prefix == 'wr_codec' else 1
    if version() != expected_abi:
        raise ValueError('unsupported library ABI: ' + str(path))
    return fingerprint().decode('ascii')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_standalone = ROOT.parent / 'WR-trainer'
    if not default_standalone.is_dir():
        default_standalone = ROOT.parent / 'LRAEP-WR-Trainer'
    parser.add_argument('--standalone', type=Path, default=default_standalone)
    default_production = ROOT.parent / 'Production-LREAP'
    parser.add_argument('--production', type=Path, default=default_production if default_production.is_dir() else None)
    parser.add_argument('--encode-value', nargs=2, metavar=('ALIAS', 'VALUE'))
    parser.add_argument('--binary', action='store_true', help='interpret --encode-value VALUE as hexadecimal bytes')
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compiler', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args(argv)
    if args.encode_value:
        print(encoded_entry(*args.encode_value, binary=args.binary))
        return 0
    if args.binary:
        parser.error('--binary requires --encode-value')
    rows = entries()
    label = platform_directory()
    author_library = WRAPPER.parent / 'bin' / label / 'libword_codec.so'
    libraries = [author_library]
    destinations = [] if args.local_only else [args.standalone]
    if not args.local_only and args.production is not None:
        destinations.append(args.production)
    bridges = {}
    for destination in destinations:
        standalone = destination / 'src' / 'workrunner'
        if not (standalone / '__init__.py').is_file():
            raise ValueError('runtime checkout not found: ' + str(destination))
        libraries.append(standalone / 'bin' / label / 'libword_codec.so')
        bridges[standalone / 'native_text_codec.py'] = WRAPPER.read_bytes()
        for name in ('rpc_fields.py', 'wire_io.py'):
            bridges[standalone / name] = WRAPPER.with_name(name).read_bytes().replace(
                b'from .word_codec import ', b'from .native_text_codec import ')
    source_id = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    wire_source_id = hashlib.sha256(SOURCE.read_bytes() + b'\0' + WIRE_SOURCE.read_bytes()).hexdigest()
    wire_libraries = [path.with_name('libwire_codec.so') for path in libraries]
    artifact_groups = [(libraries, source_id, 'wr_codec'), (wire_libraries, wire_source_id, 'wr_wire')]
    readme_paths = [README] + [destination / 'README.md' for destination in destinations]
    readmes = {path: expected_readme(rows, path) for path in readme_paths}
    if args.check:
        problems = []
        for path, expected in readmes.items():
            if not path.is_file() or path.read_text() != expected:
                problems.append('README mapping section is stale: ' + str(path))
        for paths, fingerprint, prefix in artifact_groups:
            for path in paths:
                try:
                    if artifact_source_id(path, prefix) != fingerprint:
                        problems.append('compiled source fingerprint is stale: ' + str(path))
                except (OSError, ValueError, AttributeError) as error:
                    problems.append(str(error))
            if len(paths) > 1 and all(path.is_file() for path in paths):
                if any(path.read_bytes() != paths[0].read_bytes() for path in paths[1:]):
                    problems.append('compiled libraries differ: ' + paths[0].name)
        if not args.local_only:
            for deployed, content in bridges.items():
                if not deployed.is_file() or deployed.read_bytes() != content:
                    problems.append('standalone Python bridge differs: ' + deployed.name)
        if problems:
            raise ValueError('\n'.join(problems))
        print('Native word/wire codec sources, README and deployed artifacts agree (' + str(len(rows)) + ' mappings).')
        return 0
    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise ValueError('C++ compiler not found: ' + args.compiler)
    with tempfile.TemporaryDirectory(prefix='word-codec-build-') as temporary:
        staged = []
        for source, (paths, fingerprint, prefix), macro in zip(
            (SOURCE, WIRE_SOURCE), artifact_groups, ('WR_CODEC_SOURCE_ID', 'WR_WIRE_SOURCE_ID'),
        ):
            compiled = Path(temporary) / paths[0].name
            command = [compiler, '-std=c++17', '-O2', '-fPIC', '-shared', '-fvisibility=hidden', '-Wl,--build-id=none', '-D' + macro + '="' + fingerprint + '"', str(source), '-o', str(compiled)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            if artifact_source_id(compiled, prefix) != fingerprint:
                raise ValueError('compiled source fingerprint does not match: ' + str(source))
            staged.append((paths, compiled.read_bytes()))
        for paths, content in staged:
            for path in paths:
                atomic_write(path, content)
    if not args.local_only:
        for deployed, content in bridges.items():
            atomic_write(deployed, content)
    for path, content in readmes.items():
        atomic_write(path, content.encode())
    print('Built and deployed ' + label + ' native word/wire codecs (' + str(len(rows)) + ' mappings).')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        raise SystemExit(1)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
