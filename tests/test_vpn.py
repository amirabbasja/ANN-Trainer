"""Share-link compatibility, private core startup, and fail-closed cleanup."""

import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from LRAEP_trainer import vpn
from LRAEP_trainer.network import SocksProxy


ID = '5783a3e7-e373-51cd-8642-c83782b807c5'
KEY = base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip('=')


def vmess(**updates):
    value = {'v': '2', 'ps': 'private label', 'add': 'server.example', 'port': '443', 'id': ID,
             'aid': '0', 'scy': 'auto', 'net': 'tcp', 'type': 'none', 'tls': 'tls', 'sni': 'front.example'}
    value.update(updates)
    return 'vmess://' + base64.b64encode(json.dumps(value).encode()).decode().rstrip('=')


class ShareLinkTests(unittest.TestCase):
    def test_vless_reality_preserves_authentication_and_transport_settings(self):
        result = vpn.parse_link(f'vless://{ID}@server.example:443?security=reality&type=tcp&pbk={KEY}&sid=aabb&spx=%2Fcrawl&sni=front.example&flow=xtls-rprx-vision#secret-label')
        self.assertEqual(result['settings']['vnext'][0]['users'][0], {'id': ID, 'encryption': 'none', 'flow': 'xtls-rprx-vision'})
        self.assertEqual(result['streamSettings']['realitySettings'], {
            'serverName': 'front.example', 'fingerprint': 'chrome', 'publicKey': KEY, 'shortId': 'aabb', 'spiderX': '/crawl'})
        self.assertNotIn('secret-label', json.dumps(result))

    def test_vless_websocket_tls_decodes_host_and_path(self):
        result = vpn.parse_link(f'vless://{ID}@server.example:443?security=tls&type=ws&host=front.example&path=%2Fsocket%3Fed%3D2048&sni=front.example&alpn=http%2F1.1&fp=chrome')
        self.assertEqual(result['streamSettings']['wsSettings'], {'host': 'front.example', 'path': '/socket?ed=2048'})
        self.assertEqual(result['streamSettings']['tlsSettings']['alpn'], ['http/1.1'])
        self.assertIs(result['streamSettings']['tlsSettings']['allowInsecure'], False)

    def test_grpc_multi_mode_and_service_name(self):
        result = vpn.parse_link(f'vless://{ID}@server.example:443?security=tls&type=grpc&serviceName=service&authority=front.example&mode=multi')
        self.assertEqual(result['streamSettings']['grpcSettings'], {'serviceName': 'service', 'multiMode': True, 'authority': 'front.example'})

    def test_httpupgrade_and_simple_xhttp(self):
        for transport, key in [('httpupgrade', 'httpupgradeSettings'), ('xhttp', 'xhttpSettings')]:
            with self.subTest(transport=transport):
                result = vpn.parse_link(f'vless://{ID}@server.example:443?type={transport}&security=tls&path=%2Fproxy&host=front.example')
                self.assertEqual(result['streamSettings'][key]['path'], '/proxy')
                self.assertEqual(result['streamSettings'][key]['host'], 'front.example')

    def test_trojan_password_is_percent_decoded_and_tls_defaults_on(self):
        result = vpn.parse_link('trojan://secret%3Apassword%40yes@server.example:443?sni=front.example')
        self.assertEqual(result['settings']['servers'][0]['password'], 'secret:password@yes')
        self.assertEqual(result['streamSettings']['security'], 'tls')
        self.assertFalse(result['streamSettings']['tlsSettings']['allowInsecure'])

    def test_ipv6_server_literal(self):
        result = vpn.parse_link(f'vless://{ID}@[::1]:443?type=tcp')
        self.assertEqual(result['settings']['vnext'][0]['address'], '::1')

    def test_vmess_base64_json_aead(self):
        result = vpn.parse_link(vmess())
        self.assertEqual(result['protocol'], 'vmess')
        self.assertEqual(result['settings']['vnext'][0]['port'], 443)
        self.assertEqual(result['settings']['vnext'][0]['users'][0], {'id': ID, 'security': 'auto', 'alterId': 0})
        self.assertEqual(result['streamSettings']['tlsSettings']['serverName'], 'front.example')

    def test_vmess_grpc_path_maps_to_service(self):
        result = vpn.parse_link(vmess(net='grpc', path='service', type='multi'))
        self.assertEqual(result['streamSettings']['grpcSettings'], {'serviceName': 'service', 'multiMode': True})

    def test_vmess_legacy_alter_id_and_unknown_options_rejected(self):
        for link in (vmess(aid='64'), vmess(evil='secret'), vmess(scy='none')):
            with self.assertRaises(vpn.VPNError):
                vpn.parse_link(link)

    def test_shadowsocks_sip002_password_can_contain_colons(self):
        userinfo = base64.urlsafe_b64encode(b'aes-128-gcm:secret:password').decode().rstrip('=')
        result = vpn.parse_link(f'ss://{userinfo}@server.example:8388#private')
        self.assertEqual(result['settings']['servers'][0], {'address': 'server.example', 'port': 8388, 'method': 'aes-128-gcm', 'password': 'secret:password'})

    def test_shadowsocks_legacy_whole_link_and_plain_userinfo(self):
        legacy = base64.b64encode(b'aes-256-gcm:secret@server.example:8388').decode().rstrip('=')
        one = vpn.parse_link('ss://' + legacy)
        two = vpn.parse_link('ss://aes-256-gcm:secret@server.example:8388')
        self.assertEqual(one, two)
        self.assertEqual(one, vpn.parse_link('ss://aes-256-gcm%3Asecret@server.example:8388'))

    def test_duplicate_vmess_json_security_fields_are_rejected(self):
        value = '{"id":"private", "tls":"tls", "tls":"none"}'
        link = 'vmess://' + base64.b64encode(value.encode()).decode()
        with self.assertRaisesRegex(vpn.VPNError, 'duplicate'):
            vpn.parse_link(link)

    def test_shadowsocks_plugins_are_rejected(self):
        with self.assertRaises(vpn.VPNError):
            vpn.parse_link('ss://aes-256-gcm:secret@server.example:8388?plugin=v2ray-plugin')

    def test_unknown_duplicate_conflicting_and_insecure_options_are_rejected(self):
        queries = ['type=quic', 'security=xtls', 'security=tls&allowInsecure=1',
                   'security=tls&type=tcp&type=ws', 'security=tls&sni=a&serverName=b',
                   'security=tls&foo=private', 'type=tcp&headerType=http',
                   'type=tcp&path=%2Fignored', 'type=ws&mode=multi',
                   'type=grpc&path=%2Fignored', 'security=none&sni=front.example',
                   'type=ws&security=reality&pbk=' + KEY + '&sni=front.example']
        for query in queries:
            with self.subTest(query=query), self.assertRaises(vpn.VPNError):
                vpn.parse_link(f'vless://{ID}@server.example:443?{query}')

    def test_invalid_reality_keys_and_short_ids_are_rejected(self):
        for suffix in ('pbk=short&sni=front.example', f'pbk={KEY}&sid=abc&sni=front.example', f'pbk={KEY}'):
            with self.assertRaises(vpn.VPNError):
                vpn.parse_link(f'vless://{ID}@server.example:443?security=reality&{suffix}')

    def test_malformed_links_fail_without_echoing_secrets(self):
        secret = 'DO_NOT_LOG_THIS'
        values = [f'https://{secret}@server.example:443', f'vmess://{secret}',
                  f'vless://{secret}@server.example:99999', f'vless://{secret}@:443',
                  f'vless://{secret}@server.example:443?secret={secret}',
                  f'trojan://{secret}:password@server.example:443', '', ' vless://x@y:443',
                  'vless://x@y:443\n']
        for link in values:
            with self.subTest(case=values.index(link)):
                with self.assertRaises(vpn.VPNError) as caught:
                    vpn.parse_link(link)
                self.assertNotIn(secret, str(caught.exception))


class ManagedCoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.core = Path(self.temporary.name) / 'xray'
        self.core.write_text('fake core')
        self.core.chmod(0o700)
        self.settings = {'core_path': str(self.core), 'startup_timeout': 0.05,
                         'link': f'vless://{ID}@server.example:443?security=tls'}
        self.process = Mock()
        self.process.poll.return_value = None
        self.process.wait.return_value = 0

    def test_context_sends_private_authenticated_socks_config_only_through_stdin(self):
        captured = {}
        def start(args, **kwargs):
            captured['args'] = args
            self.assertEqual(args[1:], ['run', '-format', 'json', '-config', 'stdin:'])
            self.assertNotIn('shell', kwargs)
            self.assertIs(kwargs['stdin'], subprocess.PIPE)
            self.assertIs(kwargs['stdout'], subprocess.DEVNULL)
            self.assertIs(kwargs['stderr'], subprocess.DEVNULL)
            return self.process
        with patch.object(vpn.subprocess, 'Popen', side_effect=start), \
                patch.object(vpn, '_ready', return_value=True), \
                patch.object(vpn.os, 'open', side_effect=AssertionError('no plaintext config files')):
            with vpn.ManagedVPN(self.settings) as endpoint:
                configuration = json.loads(self.process.stdin.write.call_args.args[0])
                self.process.stdin.close.assert_called_once()
                self.assertIsInstance(endpoint, SocksProxy)
                self.assertEqual(endpoint.host, '127.0.0.1')
                self.assertNotIn(endpoint.password, repr(endpoint))
                self.assertNotIn(endpoint.username, repr(endpoint))
                inbound = configuration['inbounds'][0]
                self.assertEqual(inbound['listen'], '127.0.0.1')
                self.assertEqual(inbound['settings']['accounts'], [{'user': endpoint.username, 'pass': endpoint.password}])
                self.assertFalse(inbound['settings']['udp'])
                self.assertEqual([out['protocol'] for out in configuration['outbounds']], ['vless'])
                self.assertNotIn(self.settings['link'], ' '.join(captured['args']))
            self.process.terminate.assert_called_once()
            self.process.wait.assert_called_once()

    def test_startup_exit_fails_closed_and_reaps_process(self):
        self.process.poll.return_value = 2
        manager = vpn.ManagedVPN(self.settings)
        with patch.object(vpn.subprocess, 'Popen', return_value=self.process):
            with self.assertRaisesRegex(vpn.VPNError, 'exited'):
                manager.__enter__()
        self.process.wait.assert_called_once()
        self.assertIsNone(manager.proxy)
        self.assertIsNone(manager.process)

    def test_startup_timeout_terminates_core_and_closes_stdin(self):
        manager = vpn.ManagedVPN(self.settings)
        with patch.object(vpn.subprocess, 'Popen', return_value=self.process), patch.object(vpn, '_ready', return_value=False):
            with self.assertRaisesRegex(vpn.VPNError, 'timeout'):
                manager.__enter__()
        self.process.terminate.assert_called_once()
        self.process.stdin.close.assert_called_once()
        self.assertIsNone(manager.proxy)
        self.assertIsNone(manager.process)

    def test_unresponsive_core_is_killed_after_termination_timeout(self):
        self.process.wait.side_effect = [subprocess.TimeoutExpired('xray', 2), 0]
        with patch.object(vpn.subprocess, 'Popen', return_value=self.process), patch.object(vpn, '_ready', return_value=True):
            with vpn.ManagedVPN(self.settings):
                pass
        self.process.kill.assert_called_once()

    def test_startup_os_error_does_not_expose_core_output_or_credentials(self):
        manager = vpn.ManagedVPN(self.settings)
        with patch.object(vpn.subprocess, 'Popen', side_effect=OSError(self.settings['link'])):
            with self.assertRaises(vpn.VPNError) as caught:
                manager.__enter__()
        self.assertNotIn(self.settings['link'], str(caught.exception))
        self.assertIsNone(manager.proxy)

    def test_keyboard_interrupt_during_startup_cleans_proxy_state(self):
        manager = vpn.ManagedVPN(self.settings)
        with patch.object(vpn.subprocess, 'Popen', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                manager.__enter__()
        self.assertIsNone(manager.proxy)

    def test_broken_config_pipe_fails_closed_without_exposing_credentials(self):
        manager = vpn.ManagedVPN(self.settings)
        self.process.stdin.write.side_effect = BrokenPipeError(self.settings['link'])
        self.process.stdin.closed = False
        with patch.object(vpn.subprocess, 'Popen', return_value=self.process), \
                patch.object(vpn, '_ready') as ready:
            with self.assertRaises(vpn.VPNError) as caught:
                manager.__enter__()
        self.assertNotIn(self.settings['link'], str(caught.exception))
        ready.assert_not_called()
        self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once()
        self.process.stdin.close.assert_called_once()
        self.assertIsNone(manager.process)
        self.assertIsNone(manager.proxy)

    def test_explicit_core_takes_priority_over_bundle_and_path(self):
        with patch.object(vpn, 'bundled_xray',
                          side_effect=AssertionError('explicit core must not inspect bundle')), \
                patch.object(vpn.shutil, 'which',
                             side_effect=AssertionError('explicit core must not inspect PATH')), \
                patch.object(vpn.subprocess, 'Popen', return_value=self.process) as start, \
                patch.object(vpn, '_ready', return_value=True):
            with vpn.ManagedVPN(self.settings):
                self.assertEqual(start.call_args.args[0][0], str(self.core))

    def test_automatic_core_uses_bundle_before_path(self):
        self.settings['core_path'] = None
        with patch.object(vpn, 'bundled_xray', return_value=self.core) as bundled, \
                patch.object(vpn.shutil, 'which',
                             side_effect=AssertionError('bundle must work without PATH')), \
                patch.object(vpn.subprocess, 'Popen', return_value=self.process) as start, \
                patch.object(vpn, '_ready', return_value=True):
            with vpn.ManagedVPN(self.settings):
                self.assertEqual(start.call_args.args[0][0], str(self.core))
        bundled.assert_called_once_with()

    def test_automatic_core_can_use_path_when_bundle_is_unavailable(self):
        self.settings['core_path'] = None
        with patch.object(vpn, 'bundled_xray', return_value=None), \
                patch.object(vpn.shutil, 'which', return_value=str(self.core)) as lookup, \
                patch.object(vpn.subprocess, 'Popen', return_value=self.process) as start, \
                patch.object(vpn, '_ready', return_value=True):
            with vpn.ManagedVPN(self.settings):
                self.assertEqual(start.call_args.args[0][0], str(self.core))
        lookup.assert_called_once_with('xray')

    def test_missing_core_does_not_start_any_process(self):
        self.settings['core_path'] = str(self.core.parent / 'missing')
        with patch.object(vpn.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(vpn.VPNError, 'unavailable'):
                vpn.ManagedVPN(self.settings).__enter__()
            process.assert_not_called()

    def test_readiness_authenticates_without_connecting_to_remote_target(self):
        connection = Mock()
        connection.recv.side_effect = [b'\x05', b'\x02', b'\x01\x00']
        context = Mock()
        context.__enter__ = Mock(return_value=connection)
        context.__exit__ = Mock(return_value=False)
        endpoint = SocksProxy('127.0.0.1', 12345, 'user', 'secret')
        with patch.object(vpn.socket, 'create_connection', return_value=context) as connect:
            self.assertTrue(vpn._ready(endpoint, 0.1))
        connect.assert_called_once_with(('127.0.0.1', 12345), timeout=0.1)
        self.assertEqual(connection.sendall.call_args_list[0].args[0], b'\x05\x01\x02')
        self.assertEqual(connection.sendall.call_args_list[1].args[0], b'\x01\x04user\x06secret')
        self.assertEqual(connection.sendall.call_count, 2)


if __name__ == '__main__':
    unittest.main()
