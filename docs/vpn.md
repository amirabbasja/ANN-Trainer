# VPN routing and connection tests

LRAEP uses a managed Xray process for application routing. Pool TCP/TLS,
pool transport and login checks, node RPC HTTP/HTTPS, and connection-test
requests use its authenticated local SOCKS5 endpoint. This does not change
the machine's global VPN, routes, or other applications.

## Install the core

From the project directory, run:

```bash
.venv/bin/python setup_vpn.py
```

The installer downloads the official Xray-core release `v26.10.21`, checks its
pinned SHA-256 checksum, and installs `xray` into
`~/.local/share/LRAEP/bin/` by default. Runtime searches the configured
`vpn.core_path`, `xray` on `PATH`, or the application data directory. It does
not download a core when starting a miner or running a link check.

Automatic installation supports Linux x86_64 and arm64. On other platforms,
install a compatible Xray executable yourself and select it with `core_path`.
The installer preserves Xray's original license beside the executable; its
source and license are linked in [third-party notices](../THIRD_PARTY_NOTICES.md).

The core installer and the separate Python/Rust dependency setup are bootstrap
commands. Their downloads are not routed through the application VPN, which
requires an installed core and a working share link first.

## Select a configuration

The configurations include this section, initially disabled:

```json
"vpn": {
  "enabled": false,
  "link": "",
  "core_path": null,
  "startup_timeout": 10,
  "links_file": "links.txt",
  "check_url": "https://api.ipify.org"
}
```

Paste a complete configuration link into `link`, set `enabled` to `true`, and
start the miner with the usual command:

```bash
.venv/bin/python ann_trainer.py --config config.json pool
```

The same setting applies to `pool-check`, `pool-login-check`, and solo `train`
commands. The offline `self-test` needs no proxy or network connection.

For solo mining, select an RPC endpoint reachable from your VPN server. A
loopback RPC address such as `127.0.0.1` refers to the VPN server's own loopback
when tunneled, and the server may block private addresses. There is no automatic
local-node bypass.

`core_path` can select a different installed Xray executable. Relative paths,
including `links_file`, resolve beside the selected configuration file.
`startup_timeout` bounds local core startup in seconds. The app starts a private
loopback SOCKS listener, retains the selected route in its network clients,
and terminates the core when its command finishes.

Supported share-link schemes are:

- `vless://`, including common TLS and REALITY configurations.
- `vmess://` using base64 JSON and AEAD (`alterId` zero).
- `trojan://`.
- `ss://` using supported Shadowsocks encryption methods.

Common TCP, WebSocket, gRPC, HTTPUpgrade, and XHTTP transports are accepted when
the selected protocol and options support them. Unsupported options are rejected
rather than discarded. Subscription URLs, raw JSON configurations, legacy
VMess `alterId`, and Shadowsocks plugins are not supported. Links requesting
disabled TLS certificate verification are rejected.

## Test a list of links

Paste one complete share link per line into `links.txt`; a commented template
is supplied in `links.example.txt`. Blank lines and lines beginning with `#`
are ignored. `links.txt` and the active `config.json` are ignored by Git because
they can contain credentials.

```bash
.venv/bin/python ann_trainer.py --config config.json vpn-check --links links.txt --attempts 3 --timeout 5
```

To save the results:

```bash
.venv/bin/python ann_trainer.py --config config.json vpn-check --links links.txt --output vpn-results.json
```

If `--links` is omitted, the checker uses `vpn.links_file`. Every entry gets
its own temporary core and local authenticated SOCKS listener. Tests use the
pool endpoint in the same configuration file; no wallet authentication, GPU
work, or mining shares are sent. The checker works even when `vpn.enabled` is
false, since it selects each supplied link explicitly.

The reported “ping” is the average tunneled TCP connection time, including
the verified pool TLS handshake when enabled. Loss is the percentage of failed
connection attempts. This is a connection test, not ICMP packet loss or a
measurement of mining share acceptance. Each pool attempt uses the selected
timeout; HTTPS is measured separately.

The HTTPS probe requests `vpn.check_url` through the same proxy. Its default
service returns the proxy's exit IP. A different HTTPS URL can check
reachability, but an exit IP is displayed only if its response is literally an
IP address. Set `check_url` to `null` to skip that probe.

| Status | Meaning |
|---|---|
| `POOL_OK` | The pool's TLS handshake succeeded with certificate and hostname verification. |
| `CONNECT_OK` | A plain TCP pool connection and the independent HTTPS probe succeeded. TCP alone does not verify the pool destination. |
| `UNVERIFIED` | Plain TCP connected, but HTTPS failed or was disabled; proxy usability remains unverified. |
| `POOL_FAIL` | All pool connection attempts failed, even if the HTTPS probe worked. |
| `CORE_FAIL` | The proxy core could not start or stopped during testing. |
| `INVALID` | The link was malformed or used unsupported options. |

HTTPS status is shown separately, so a blocked test website does not hide a
successful verified pool TLS connection. The command exits with status zero
when at least one link is usable (`POOL_OK` or `CONNECT_OK`), and status one
otherwise. JSON reports include the input line number, connection statistics,
HTTPS result, optional exit IP, `usable`, and `pool_proof_level`.

## Routing and privacy behavior

Pool and node target hostnames are sent to SOCKS for remote DNS resolution;
they are not resolved locally by the Python client. Xray still makes its
underlying connection to the configured VPN server, including any bootstrap
resolution of that server's hostname. The selected server can observe the
destinations of traffic it carries.

When VPN mode is enabled, proxy startup, authentication, or connection failures
stop or fail the affected network operation. There is no direct-connection
fallback. Existing pool and RPC TLS certificate and hostname checks remain
enabled, and environment proxy variables cannot override the selected route.

Console output and JSON reports identify configurations by line number and
omit the original share links, passwords, and raw core diagnostics. Temporary
core configuration files use private permissions and are removed on shutdown.

Core source and release information are available from the
[official Xray-core repository](https://github.com/XTLS/Xray-core).
