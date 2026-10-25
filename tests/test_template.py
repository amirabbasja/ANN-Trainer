"""Coinbase regression checks against the real pinned PEARL helper modules."""

import hashlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pearl_trainer.node import NodeClient


def double_sha(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def height_prefix(height):
    """The minimal script-number prefix checked by PEARL's Go full node."""
    if height == 0:
        return b"\x00"
    if height <= 16:
        return bytes([0x50 + height])
    encoded = height.to_bytes((height.bit_length() + 7) // 8, "little")
    if encoded[-1] & 0x80:
        encoded += b"\x00"
    return bytes([len(encoded)]) + encoded


def serialized_coinbase_script(transaction_bytes):
    """Read the actual input script, bypassing Script's display formatting."""
    offset = 4  # version
    if transaction_bytes[offset:offset + 2] == b"\x00\x01":
        offset += 2  # witness marker and flag
    if transaction_bytes[offset] != 1:
        raise AssertionError("expected one coinbase input")
    offset += 1 + 32 + 4  # input count, null outpoint hash, outpoint index
    length = transaction_bytes[offset]
    if length >= 0xFD:
        raise AssertionError("coinbase script is too long")
    return transaction_bytes[offset + 1:offset + 1 + length]


class TemplateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get(
            "PEARL_OFFICIAL_SOURCE",
            str(Path(__file__).resolve().parents[1] / ".official" / "pearl"),
        ))
        if not source.exists():
            raise unittest.SkipTest("pinned official PEARL helper checkout is unavailable")
        try:
            from pearl_trainer.protocol import load_official_helpers
            load_official_helpers(source)
            from bitcoinutils.bech32 import Encoding, bech32_encode, convertbits
        except ImportError as exc:
            raise unittest.SkipTest(f"official native/helper dependency unavailable: {exc}") from None
        cls.address = bech32_encode(
            "prl", [1] + convertbits(bytes(range(32)), 8, 5, True), Encoding.BECH32M
        )

    def template_response(self, height):
        # A structurally valid ordinary transaction, included to exercise the
        # Merkle root update beyond a single coinbase transaction.
        tx = bytes.fromhex(
            "0100000001" + "11" * 32 + "0000000000ffffffff"
            + "0101000000000000000000000000"
        )
        self.tx_hash = double_sha(tx)
        witness_root = double_sha(bytes(32) + self.tx_hash)
        witness_commitment = double_sha(witness_root + bytes(32))
        return {
            "bits": "1d2fffff", "target": f"{0x2FFFFF << (8 * (0x1D - 3)):064x}",
            "curtime": 2000, "height": height, "previousblockhash": "22" * 32,
            "vsizelimit": 1000000, "version": 2, "longpollid": "fixture",
            "maxtime": 3000, "mintime": 1000, "mutable": [], "noncerange": "00000000ffffffff",
            "capabilities": ["coinbasevalue"], "coinbaseaux": {"flags": "0d2f503253482f706561726c642f"},
            "coinbasevalue": 5000000000, "requiredcertversion": 3,
            "default_witness_commitment": witness_commitment.hex(),
            "transactions": [{
                "data": tx.hex(), "hash": self.tx_hash[::-1].hex(),
                "txid": self.tx_hash[::-1].hex(), "depends": [], "fee": 0,
                "vsize": len(tx),
            }],
        }

    def test_minimal_height_merkle_root_and_witness_preserved(self):
        client = NodeClient("http://127.0.0.1:44107")
        for height in (1, 16, 17, 127, 128, 255, 256, 2591, 32767, 32768, 8388607, 8388608):
            with self.subTest(height=height):
                response = self.template_response(height)
                with patch.object(client, "raw_template", return_value=response):
                    template = client.get_template(self.address)
                coinbase = template.coinbase_tx
                script = serialized_coinbase_script(coinbase.to_bytes(True))
                expected_script = height_prefix(height) + b"\x00" + bytes.fromhex(response["coinbaseaux"]["flags"])
                self.assertEqual(script, expected_script)

                # Independently hash the real non-witness serialization rather
                # than trusting a Script object's representation or helper txid.
                coinbase_hash = double_sha(coinbase.to_bytes(False))
                expected_root = double_sha(coinbase_hash + self.tx_hash)[::-1]
                self.assertEqual(bytes(template.header.incomplete_header.merkle_root), expected_root)

                self.assertTrue(coinbase.has_segwit)
                self.assertEqual(coinbase.witnesses[0].stack, [bytes(32).hex()])
                self.assertEqual(
                    coinbase.outputs[-1].script_pubkey.to_hex(),
                    "6a24aa21a9ed" + response["default_witness_commitment"],
                )
                self.assertEqual(template.raw_transactions, [bytes.fromhex(response["transactions"][0]["data"])])
                self.assertEqual(int(template.required_cert_version), 3)


if __name__ == "__main__":
    unittest.main()
