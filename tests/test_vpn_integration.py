"""Real Xray tunnels against controlled loopback services; no Internet needed."""

from contextlib import ExitStack, contextmanager
import base64
import http.server
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlencode
import uuid

from LRAEP_trainer import network
from LRAEP_trainer.node import NodeClient, NodeError
from LRAEP_trainer.paths import artifacts_dir, bundled_xray
from LRAEP_trainer.vpn import ManagedVPN


def xray_path():
    bundled = bundled_xray()
    if bundled is not None:
        return bundled
    executable = shutil.which('xray')
    if executable:
        return Path(executable)
    path = artifacts_dir().parent / 'bin' / ('xray.exe' if os.name == 'nt' else 'xray')
    return path if path.is_file() else None


class XrayServerFixture:
    """Reusable private loopback proxy server for tests and bounded smoke checks."""

    def __init__(self, protocol='vless', transport='tcp'):
        self.protocol, self.transport = protocol, transport
        self.core = xray_path()
        self.process = None
        self.temporary = None
        self.link = None

    def __enter__(self):
        if self.core is None:
            raise unittest.SkipTest('install the optional Xray core to run real tunnel tests')
        try:
            self.temporary = tempfile.TemporaryDirectory(prefix='LRAEP-test-xray-')
            directory = Path(self.temporary.name)
            directory.chmod(0o700)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                self.port = reservation.getsockname()[1]
            self.user_id, self.password = str(uuid.uuid4()), secrets.token_hex(16)
            stream = {'network': self.transport, 'security': 'none'}
            if self.transport == 'ws':
                stream['wsSettings'] = {'path': '/LRAEP-test'}
            if self.protocol == 'vless':
                settings = {'clients': [{'id': self.user_id}], 'decryption': 'none'}
                options = {'type': self.transport, 'security': 'none', 'encryption': 'none'}
                if self.transport == 'ws':
                    options['path'] = '/LRAEP-test'
                self.link = f'vless://{self.user_id}@127.0.0.1:{self.port}?' + urlencode(options)
            elif self.protocol == 'vmess':
                settings = {'clients': [{'id': self.user_id}]}
                data = {'v': '2', 'add': '127.0.0.1', 'port': str(self.port), 'id': self.user_id,
                        'aid': '0', 'scy': 'auto', 'net': 'tcp', 'type': 'none', 'tls': ''}
                self.link = 'vmess://' + base64.b64encode(json.dumps(data).encode()).decode()
            elif self.protocol == 'shadowsocks':
                settings = {'method': 'aes-128-gcm', 'password': self.password, 'network': 'tcp'}
                encoded = base64.urlsafe_b64encode(f'aes-128-gcm:{self.password}'.encode()).decode().rstrip('=')
                self.link = f'ss://{encoded}@127.0.0.1:{self.port}'
            else:
                raise ValueError('unsupported test fixture protocol')
            configuration = {'log': {'loglevel': 'none', 'access': 'none', 'error': 'none'},
                             'inbounds': [{'listen': '127.0.0.1', 'port': self.port,
                                           'protocol': self.protocol, 'settings': settings,
                                           'streamSettings': stream}],
                             # Recent Xray versions block private targets on
                             # proxy-server outbounds by default. Only this
                             # loopback fixture explicitly permits loopback.
                             'outbounds': [{'protocol': 'freedom', 'settings': {
                                 'finalRules': [{'action': 'allow', 'ip': ['127.0.0.1/32', '::1/128']}],
                             }}]}
            config = directory / 'server.json'
            with os.fdopen(os.open(config, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600), 'w') as handle:
                json.dump(configuration, handle)
            self.process = subprocess.Popen([str(self.core), 'run', '-config', str(config)],
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('local Xray fixture rejected its configuration')
                try:
                    with socket.create_connection(('127.0.0.1', self.port), timeout=0.1):
                        return self
                except OSError:
                    time.sleep(0.02)
            raise RuntimeError('local Xray fixture startup timed out')
        except BaseException:
            self.close()
            raise

    def close(self):
        process, self.process = self.process, None
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            if self.temporary is not None:
                self.temporary.cleanup()
                self.temporary = None

    def __exit__(self, *_):
        self.close()


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.server.requests.append(('GET', self.path))
        self._reply({'ok': True, 'through': 'Xray'})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(('POST', request['method']))
        self._reply({'jsonrpc': '2.0', 'id': request['id'], 'result': 'through-vpn'})

    def _reply(self, value):
        raw = json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@contextmanager
def local_http_server():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, name='LRAEP-test-http', daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError('local HTTP fixture did not stop')


class RealXrayTunnelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = xray_path()
        if cls.core is None:
            raise unittest.SkipTest('install the optional Xray core to run real tunnel tests')

    def _exercise_tunnel(self, protocol, transport='tcp'):
        with local_http_server() as http, XrayServerFixture(protocol, transport) as server:
            settings = {'core_path': str(self.core), 'startup_timeout': 5, 'link': server.link}
            with ManagedVPN(settings) as proxy:
                target = f'http://127.0.0.1:{http.server_port}'
                original_connect = socket.create_connection
                with patch.object(network.socket, 'create_connection', wraps=original_connect) as connections:
                    with network.routed(proxy):
                        opener = network.http_opener(network.current_proxy())
                        node = NodeClient(target, username='fixture-user', password='fixture-password', timeout=3)
                    with opener.open(target + '/probe', timeout=3) as response:
                        self.assertEqual(json.loads(response.read()), {'ok': True, 'through': 'Xray'})
                    # The main route context has ended. The captured opener must
                    # keep the proxy even when its client runs in a new thread.
                    result, errors = [], []
                    def call():
                        try:
                            result.append(node.rpc('getbestblockhash'))
                        except BaseException as error:
                            errors.append(error)
                    thread = threading.Thread(target=call, name='LRAEP-test-rpc')
                    thread.start()
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
                    self.assertFalse(errors, errors)
                    self.assertEqual(result, ['through-vpn'])
                self.assertEqual(connections.call_count, 2)
                self.assertTrue(all(item.args[0] == (proxy.host, proxy.port) for item in connections.call_args_list))
                self.assertEqual(http.requests, [('GET', '/probe'), ('POST', 'getbestblockhash')])

    def test_vless_tcp_tunnels_http_and_threaded_node_rpc(self):
        self._exercise_tunnel('vless')

    def test_vless_websocket_tunnels_http_and_threaded_node_rpc(self):
        self._exercise_tunnel('vless', 'ws')

    def test_vmess_aead_tunnels_http_and_threaded_node_rpc(self):
        self._exercise_tunnel('vmess')

    def test_shadowsocks_tunnels_http_and_threaded_node_rpc(self):
        self._exercise_tunnel('shadowsocks')

    def test_bundled_core_tunnels_without_path_lookup(self):
        core = bundled_xray()
        if core is None:
            self.skipTest('bundled Xray is unavailable on this platform')
        with local_http_server() as http, XrayServerFixture() as server:
            manager = ManagedVPN({'core_path': None, 'startup_timeout': 5, 'link': server.link})
            with patch.dict(os.environ, {'PATH': ''}), \
                    patch('LRAEP_trainer.vpn.shutil.which',
                          side_effect=AssertionError('bundled core must work without PATH lookup')):
                with manager as proxy:
                    self.assertEqual(Path(manager.process.args[0]), core)
                    opener = network.http_opener(proxy)
                    with opener.open(f'http://127.0.0.1:{http.server_port}/bundled', timeout=3) as response:
                        self.assertEqual(json.loads(response.read()), {'ok': True, 'through': 'Xray'})
            self.assertEqual(http.requests, [('GET', '/bundled')])

    def test_dead_client_core_never_falls_back_to_direct_rpc(self):
        with local_http_server() as http, XrayServerFixture() as server:
            manager = ManagedVPN({'core_path': str(self.core), 'startup_timeout': 5, 'link': server.link})
            with manager as proxy:
                with network.routed(proxy):
                    node = NodeClient(f'http://127.0.0.1:{http.server_port}', timeout=0.5)
                manager.process.terminate()
                manager.process.wait(timeout=2)
                original_connect = socket.create_connection
                with patch.object(network.socket, 'create_connection', wraps=original_connect) as connect:
                    with self.assertRaises(NodeError):
                        node.rpc('getbestblockhash')
                self.assertEqual(connect.call_count, 1)
                self.assertEqual(connect.call_args.args[0], (proxy.host, proxy.port))
                self.assertEqual(http.requests, [])

    def test_advanced_link_configurations_are_accepted_by_the_real_core(self):
        # Startup validation only: these deliberately do not contact a remote
        # server, so this test does not claim successful TLS/REALITY handshakes.
        user_id = str(uuid.uuid4())
        public_key = base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip('=')
        links = [
            f'vless://{user_id}@server.example:443?security=tls&type=ws&path=%2Fws&sni=front.example&fp=chrome',
            f'vless://{user_id}@server.example:443?security=tls&type=grpc&serviceName=test&mode=multi&sni=front.example',
            f'vless://{user_id}@server.example:443?security=tls&type=httpupgrade&path=%2Ftest&sni=front.example',
            f'vless://{user_id}@server.example:443?security=tls&type=xhttp&path=%2Ftest&mode=auto&sni=front.example',
            f'vless://{user_id}@server.example:443?security=reality&type=tcp&pbk={public_key}&sid=aabb&sni=front.example&flow=xtls-rprx-vision',
            'trojan://fixture-password@server.example:443?sni=front.example',
        ]
        for number, link in enumerate(links):
            with self.subTest(case=number):
                with ManagedVPN({'core_path': str(self.core), 'startup_timeout': 5}, link=link) as proxy:
                    self.assertIsInstance(proxy, network.SocksProxy)


if __name__ == '__main__':
    unittest.main()
