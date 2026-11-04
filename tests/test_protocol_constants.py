from hashlib import sha256
import unittest

from LRAEP_trainer.S6_codec import coefficients
from LRAEP_trainer.wr_client import WRSession
from LRAEP_trainer.wr_protocol import header_mask, work_prefix_mask


class ProtocolConstantsTests(unittest.TestCase):
    def test_public_byte_helpers_preserve_original_payloads(self):
        for factory, size, digest in (
            (header_mask, 32, '9654a5630b6051d3635a690f4cc9dbfc028b5bf597b221ce1363876a5628b1f7'),
            (work_prefix_mask, 76, '461787c2ba9e41d25c97b976ce0c70a1459ca2e8e06c0a70f0648afc727a79bd'),
            (coefficients, 48, 'c6e8f8581586f378a4de6bad03bf8b2248f8cd2075be51087659198a40a39db2'),
        ):
            with self.subTest(factory=factory.__name__):
                value = factory()
                self.assertIsInstance(value, bytes)
                self.assertEqual(len(value), size)
                self.assertEqual(sha256(value).hexdigest(), digest)

    def test_auth_method_remains_available_through_class_and_instance(self):
        self.assertEqual(WRSession.AUTH_METHOD, "login")
        self.assertEqual(object.__new__(WRSession).AUTH_METHOD, "login")
