#!/usr/bin/env python3
"""Build native word/wire codec artifacts and synchronize their documentation."""

import argparse
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
ENTRY = re.compile(r'make_entry\(\s*("(?:[^"\\]|\\.)*")\s*,\s*("(?:[^"\\]|\\.)*")\s*\)')


def entries():
    rows = [(json.loads(left), json.loads(right)) for left, right in ENTRY.findall(SOURCE.read_text())]
    if not rows:
        raise ValueError('native source contains no word codec entries')
    if len({left for left, _ in rows}) != len(rows) or len({right for _, right in rows}) != len(rows):
        raise ValueError('placeholders and decoded values must each be unique')
    if any(not left or not right or max(len(left.encode()), len(right.encode())) >= 256 for left, right in rows):
        raise ValueError('mapping entries must contain between 1 and 255 UTF-8 bytes')
    return rows


def platform_directory():
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine)
    if platform.system() != 'Linux' or machine not in ('x86_64', 'aarch64'):
        raise ValueError('build on Linux x86_64 or Linux aarch64 for that target platform')
    return 'linux-' + machine


def documentation(rows):
    table = '\n'.join('| `' + left.replace('|', '\\|') + '` | `' + right.replace('|', '\\|') + '` |' for left, right in rows)
    return f'''{START}
## Native word codec

`native/word_codec.cpp` is the authoritative exact-word table. Python calls
the compiled shared library through `src/LRAEP_trainer/word_codec.py`; the
standalone receives the same Python bridge and compiled library, with no
decoder C++ source or mapping documentation. The bridge performs no Python
table lookup and has no Python decoding fallback.

```python
from LRAEP_trainer.word_codec import transform_word, word_buffer

transform_word("legacyMethod", "decode")
transform_word("pearlhash", "encode")
transform_word("unlisted-word", "decode")
with word_buffer("account") as temporary:
    consume(temporary)
```

The corresponding standalone import is `workrunner.word_codec`.
`decode_word` and `encode_word` are convenience functions. Matching is exact
and case-sensitive in each direction: unknown words, empty strings, Unicode,
and embedded NUL characters pass through unchanged. There is no substring
replacement. An unsupported method raises `ValueError`, and a non-string word
raises `TypeError`. The `walletnonce` row deliberately maps to `w_counter`, as
requested; the distinct `device_counter` row preserves the existing wire field.

### Placeholder table (decode direction)

| Placeholder | Decoded value |
| --- | --- |
{table}

The first thirteen rows are the requested mappings. The remaining rows cover
the existing device-counter field, full qualified wire methods, certificate
domain, address prefix, default device name, RPC field names and the three
node request methods. RPC fields are translated at the native network boundary;
`rpc_fields.py` also supports explicit compatibility calls. Model metadata
uses neutral field names with no decoded aliases, and Pydantic string caching
is disabled. Existing S1–S7 source
abbreviations remain identifiers; this table controls dynamic protocol text.
The fixed field schema and protocol/proof bytes remain compatible.

### Updating and distributing mappings

1. Add or change a `make_entry("placeholder", "decoded value")` row in
   `native/word_codec.cpp`. Both columns must remain unique. Use the placeholder
   at the Python boundary where that exact protocol word is needed. For a new
   network field or command, also select it in the appropriate `key_entry` or
   `Parser::translation` profile in `native/wire_codec.cpp`. Profiles distinguish
   RPC fields, pool fields, command names and ordinary string data; adding a
   word-table entry alone does not rewrite arbitrary message values.
2. Run `python3 tools/build_word_codec.py`. It validates the table, compiles the
   word and wire libraries, regenerates this README table, and atomically installs identical
   libraries and Python bridges in both repositories. This includes
   `word_codec.py`, `wire_io.py` and `rpc_fields.py`. The default standalone
   checkout is the sibling `LRAEP-WR-Trainer`; use `--standalone PATH` to choose
   another checkout or `--local-only` for an author-only build.
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

### Buffer lifetime and protection limits

Native static table values are stored in a compile-time encoded representation;
there is no decoded global table or plaintext result cache. `word_buffer`
decodes into a mutable buffer and explicitly overwrites it on normal and
exception exits. `word_matches` compares ASCII protocol values with a scoped
buffer. Legacy fixed binary tables and message constants use `fixed_buffer`,
which performs their existing Base85/XOR transformation in the native library
and clears its output after use.

`transform_word`, `decode_word`, and `encode_word` return ordinary immutable
Python strings. Callers must release those references after use; their physical
bytes cannot be guaranteed erased, and serializers, sockets, exception text,
or caller-retained objects may create copies. The framing-only
`customEncoder`/`customDecoder` API remains unchanged for wire compatibility.
Compilation and encoding provide obfuscation, not encryption or a security
boundary against a host administrator. Embedded values can be recovered from
the binary or observed while decoded. No speedup is claimed without a
workload-specific benchmark.
{END}'''


def expected_readme(rows):
    document = README.read_text()
    section = documentation(rows)
    if START in document or END in document:
        if document.count(START) != 1 or document.count(END) != 1:
            raise ValueError('README codec section markers are invalid')
        left, rest = document.split(START, 1)
        _, right = rest.split(END, 1)
        return left + section + right
    return document.rstrip() + '\n\n' + section + '\n'


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
    if version() != 1:
        raise ValueError('unsupported library ABI: ' + str(path))
    return fingerprint().decode('ascii')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--standalone', type=Path, default=ROOT.parent / 'LRAEP-WR-Trainer')
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compiler', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args(argv)
    rows = entries()
    label = platform_directory()
    author_library = WRAPPER.parent / 'bin' / label / 'libword_codec.so'
    libraries = [author_library]
    standalone_wrapper = args.standalone / 'src' / 'workrunner' / 'word_codec.py'
    bridges = [WRAPPER]
    bridges.extend(WRAPPER.with_name(name) for name in ('rpc_fields.py', 'wire_io.py'))
    if not args.local_only:
        if not (standalone_wrapper.parent / '__init__.py').is_file():
            raise ValueError('standalone checkout not found: ' + str(args.standalone))
        libraries.append(standalone_wrapper.parent / 'bin' / label / 'libword_codec.so')
    source_id = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    wire_source_id = hashlib.sha256(SOURCE.read_bytes() + b'\0' + WIRE_SOURCE.read_bytes()).hexdigest()
    wire_libraries = [path.with_name('libwire_codec.so') for path in libraries]
    artifact_groups = [(libraries, source_id, 'wr_codec'), (wire_libraries, wire_source_id, 'wr_wire')]
    readme = expected_readme(rows)
    if args.check:
        problems = []
        if README.read_text() != readme:
            problems.append('README mapping section is stale')
        for paths, fingerprint, prefix in artifact_groups:
            for path in paths:
                try:
                    if artifact_source_id(path, prefix) != fingerprint:
                        problems.append('compiled source fingerprint is stale: ' + str(path))
                except (OSError, ValueError, AttributeError) as error:
                    problems.append(str(error))
            if len(paths) == 2 and all(path.is_file() for path in paths):
                if paths[0].read_bytes() != paths[1].read_bytes():
                    problems.append('compiled libraries differ: ' + paths[0].name)
        if not args.local_only:
            for bridge in bridges:
                deployed = standalone_wrapper.with_name(bridge.name)
                if not deployed.is_file() or deployed.read_bytes() != bridge.read_bytes():
                    problems.append('standalone Python bridge differs: ' + bridge.name)
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
        for bridge in bridges:
            atomic_write(standalone_wrapper.with_name(bridge.name), bridge.read_bytes())
    atomic_write(README, readme.encode())
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
