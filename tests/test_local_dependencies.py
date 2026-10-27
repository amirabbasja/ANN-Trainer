import ast
from pathlib import Path
import subprocess
import sys
import unittest

from blake3 import blake3

from LRAEP_trainer import primitives as local


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (ROOT, ROOT.parent / 'LRAEP-WR-Trainer')
FORBIDDEN_IMPORTS = {'bitcoinutils', 'LRAEP_mining', 'pearl_mining', 'web3', 'eth_account', 'coincurve', 'hdwallet', 'base58check'}
FORBIDDEN_PACKAGES = {'bitcoin-utils', 'py-lraep-mining', 'py-pearl-mining', 'web3', 'eth-account', 'coincurve', 'hdwallet', 'base58check'}


class LocalDependencyTests(unittest.TestCase):
    def test_runtime_sources_and_requirements_do_not_reference_external_currency_sdks(self):
        for project in PROJECTS:
            with self.subTest(project=project.name):
                self.assertTrue(project.is_dir())
                for path in project.rglob('*.py'):
                    if any(part in {'.venv', '.git', 'build', '__pycache__'} for part in path.parts):
                        continue
                    if 'tests' in path.relative_to(project).parts:
                        continue
                    tree = ast.parse(path.read_text())
                    for node in ast.walk(tree):
                        modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
                        for module in modules:
                            self.assertNotIn(module.split('.')[0], FORBIDDEN_IMPORTS, str(path))
                for line in (project / 'requirements.txt').read_text().splitlines():
                    package = line.split('>=')[0].strip().lower().replace('_', '-')
                    self.assertNotIn(package, FORBIDDEN_PACKAGES)

    def test_both_runtimes_import_when_currency_sdks_and_network_access_are_blocked(self):
        script = '''
import importlib.abc
from pathlib import Path
import socket
import sys
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
blocked = {'bitcoinutils', 'LRAEP_mining', 'pearl_mining', 'web3', 'eth_account', 'coincurve', 'hdwallet', 'base58check'}
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Forbidden package import: ' + fullname)
sys.meta_path.insert(0, Deny())
def deny_connection(*args, **kwargs):
    raise AssertionError('Unexpected network connection')
socket.socket.connect = deny_connection
socket.socket.connect_ex = deny_connection
from LRAEP_trainer.protocol import load_official_helpers
load_official_helpers(root / 'src/LRAEP_helpers')
from LRAEP_trainer import pool_training, wr_work, wr_client, primitives
from LRAEP_gateway.blockchain_utils import blockchain_utils, LRAEP_header, LRAEP_block, zk_certificate
from LRAEP_gateway.comm import dataclasses, mining_configuration
from LRAEP_gateway import proof_generator
assert not blocked.intersection(sys.modules)
assert Path(primitives.__file__).resolve().is_relative_to(root)
'''
        for project in PROJECTS:
            with self.subTest(project=project.name):
                result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(project)], text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class LocalWirePrimitiveTests(unittest.TestCase):
    def test_header_and_dense_configuration_match_fixed_wire_bytes(self):
        header_bytes = bytes.fromhex(
            '04030201'
            '1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020100'
            '3f3e3d3c3b3a393837363534333231302f2e2d2c2b2a29282726252423222120'
            '44332211ffff001d'
        )
        header = local.IncompleteBlockHeader(0x01020304, bytes(range(32)), bytes(range(32, 64)), 0x11223344, 0x1D00FFFF)
        self.assertEqual(header.to_bytes(), header_bytes)
        parsed = local.IncompleteBlockHeader.from_bytes(header_bytes)
        self.assertEqual(parsed, header)
        configuration_bytes = bytes.fromhex('0008000080000000000700000000000700000000' + '00' * 32)
        configuration = local.MiningConfiguration(2048, 128, local.MMAType.Int7xInt7ToInt32, local.PeriodicPattern.from_list(range(8)), local.PeriodicPattern.from_list(range(8)))
        self.assertEqual(configuration.to_bytes(), configuration_bytes)
        self.assertEqual(local.MiningConfiguration.from_bytes(configuration_bytes), configuration)

    def test_local_merkle_paths_match_independent_keyed_hash_and_reject_tampering(self):
        key = bytes(range(32))
        for leaves in (1, 2, 3, 5, 7):
            with self.subTest(leaves=leaves):
                data = bytes(range(256)) * (leaves * 4)
                tree = local.MerkleTree(data, key)
                self.assertEqual(tree.root, blake3(data, key=key).digest())
                selected = sorted({0, leaves - 1})
                proof = tree.get_multileaf_proof(selected)
                self.assertTrue(proof.verify(key))
                self.assertFalse(proof.verify(bytes(32)))
                first = proof.leaf_data[0]
                proof.leaf_data[0] = bytes([first[0] ^ 1]) + first[1:]
                self.assertFalse(proof.verify(key))


if __name__ == '__main__':
    unittest.main()
