import unittest
from unittest.mock import patch

from LRAEP_trainer import S3_methods
from LRAEP_trainer.config import _S3_protocol
from LRAEP_trainer.S3_protocol import KEXDialect
from LRAEP_trainer.S3_training import select_S3_protocol, validate_S3_endpoint
from LRAEP_trainer.S6_requests import build_submit_request
from LRAEP_trainer.wire_constants import s5_nonce, s6_noise_seed_prefix, s7_legacy_alias
from LRAEP_trainer.wr_protocol import WRDialect


class ProtocolWordCodecTests(unittest.TestCase):
    def test_existing_field_helpers_and_requests_keep_their_wire_values(self):
        self.assertEqual(s5_nonce(), 'miner_nonce')
        self.assertEqual(s7_legacy_alias(), 'pearlhash')
        self.assertEqual(s6_noise_seed_prefix(), b'pearl/cert-v3/noise-seed/')
        auth, = KEXDialect('account-value', 'unit', request_gzip=True).auth_requests()
        self.assertEqual(auth['params']['wallet'], 'account-value.unit')
        request = build_submit_request(
            1, S5_nonce=bytes(range(8)), worker_nonce=bytes(range(8, 16)),
            m=256, n=256, k=2048, rank=128, tile_h=8, tile_w=8, sb=1,
            a_rows=(1,), b_cols=(2,),
        )
        fields, = request['params']
        self.assertEqual(fields['miner_nonce'], bytes(range(8)).hex())
        self.assertEqual(fields['worker_nonce'], bytes(range(8, 16)).hex())
        self.assertNotIn('w_counter', fields)
        self.assertEqual(WRDialect.report_request('device', 25, 1)['params'], [
            {'hashrate': 25.0, 'name': 'device'},
        ])

    def test_notification_and_legacy_comparisons_do_not_make_expected_strings(self):
        with patch.object(S3_methods, 'constant_text', side_effect=AssertionError('decoded string created')):
            self.assertEqual(S3_methods.stratum_notification_method('mining.notify'), S3_methods.WORK_UPDATE)
        self.assertEqual(_S3_protocol('pearlhash'), 'WR')
        self.assertEqual(select_S3_protocol({'protocol': 'pearlhash'}), 'WR')
        with self.assertRaisesRegex(ValueError, 'WR'):
            validate_S3_endpoint({'protocol': 'pearlhash', 'host': ''})


if __name__ == '__main__':
    unittest.main()
