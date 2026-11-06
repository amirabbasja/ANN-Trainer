import copy
import json
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from LRAEP_trainer.node import NodeClient, NodeError
from LRAEP_trainer.protocol import load_official_helpers
from LRAEP_trainer import rpc_fields


class RPCFieldLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_official_helpers(Path(__file__).resolve().parents[1] / 'src' / 'LRAEP_helpers')
        from LRAEP_gateway.rpc_types import BlockTemplateTx, CoinbaseAux, GetBlockTemplateResponse
        cls.entry_model = BlockTemplateTx
        cls.metadata_model = CoinbaseAux
        cls.template_model = GetBlockTemplateResponse

    def payload(self):
        return {
            'bits': '1d00ffff', 'curtime': 100, 'height': 14, 'version': 1,
            'target': 'ffff', 'maxtime': 200, 'mintime': 1,
            'mutable': [], 'capabilities': [],
            'previousblockhash': '00' * 32, 'vsizelimit': 100,
            'transactions': [{
                'data': '12', 'hash': '01' * 32, 'txid': '02' * 32,
                'depends': [], 'fee': 0, 'vsize': 1,
            }],
            'longpollid': 'abc', 'noncerange': '00000000ffffffff',
            'coinbaseaux': {'flags': 'cc'}, 'coinbasevalue': 1111,
            'default_witness_commitment': '00' * 32, 'requiredcertversion': 3,
        }

    def test_nested_wire_round_trip_preserves_input_and_neutral_access(self):
        payload = self.payload()
        original = copy.deepcopy(payload)
        model = self.template_model.model_validate_json(json.dumps(payload))
        self.assertEqual(model.records[0].record_id, '02' * 32)
        self.assertEqual(model.reward_amount, 1111)
        self.assertEqual(model.parent_digest, '00' * 32)
        self.assertEqual(model.reward_metadata.flags, 'cc')
        self.assertEqual(model.model_dump(), original)
        self.assertEqual(json.loads(model.model_dump_json()), original)
        self.assertEqual(payload, original)
        neutral = model.model_dump(by_alias=False)
        self.assertIn('parent_digest', neutral)
        self.assertNotIn('previousblockhash', neutral)
        self.assertIn('record_id', neutral['records'][0])
        self.assertEqual(self.template_model.model_validate(neutral).model_dump(), original)

    def test_native_wire_model_stream_and_scoped_output(self):
        class ReadIntoOnly(io.BytesIO):
            def read(self, *args):
                raise AssertionError('wire JSON must use a native readinto buffer')

        payload = self.payload()
        stream = ReadIntoOnly(json.dumps(payload).encode())
        model = self.template_model.model_validate_wire(stream)
        self.assertEqual(model.reward_amount, 1111)
        with model.wire_buffer(include={'records': {'__all__': {'record_id'}}}) as buffer:
            self.assertIsInstance(buffer, memoryview)
            self.assertEqual(json.loads(bytes(buffer)), {'transactions': [{'txid': '02' * 32}]})
            retained = buffer
        with self.assertRaises(ValueError):
            retained.tobytes()
        self.assertEqual(model.records[0].record_id, '02' * 32)

    def test_native_wire_input_errors_are_sanitized(self):
        with self.assertRaises(ValidationError) as caught:
            self.template_model.model_validate_wire(b'{"previousblockhash": "private-response", broken}')
        self.assertNotIn('private-response', str(caught.exception))
        self.assertNotIn('previousblockhash', str(caught.exception))
        self.assertIsNone(caught.exception.errors()[0]['input'])
        with self.assertRaises(TypeError):
            self.template_model.model_validate_wire('{}', by_name=True)

    def test_native_wire_limits_apply_to_raw_utf8_and_buffer_inputs(self):
        payload = self.payload()
        payload['longpollid'] = 'سلام'
        text = json.dumps(payload, ensure_ascii=False)
        raw = text.encode('utf-8')
        for source in (text, raw, bytearray(raw), memoryview(bytearray(raw))):
            with self.subTest(kind=type(source).__name__):
                self.assertEqual(
                    self.template_model.model_validate_wire(source, limit=len(raw)).poll_id,
                    'سلام',
                )
                with self.assertRaises(ValidationError) as caught:
                    self.template_model.model_validate_wire(source, limit=len(raw) - 1)
                self.assertEqual(caught.exception.errors()[0]['type'], 'json_invalid')
                self.assertIsNone(caught.exception.errors()[0]['input'])
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.template_model.model_validate_wire(raw, limit=limit)
        self.assertEqual(raw, text.encode('utf-8'))

    def test_runtime_alias_policy_and_precedence(self):
        payload = self.payload()
        neutral = self.template_model.model_validate(payload).model_dump(by_alias=False)
        for mixed in ({'reward_amount': 7, **payload}, {**payload, 'reward_amount': 7}):
            self.assertEqual(self.template_model.model_validate(mixed).reward_amount, 1111)
        with self.assertRaises(ValidationError):
            self.template_model.model_validate(payload, by_alias=False, by_name=True)
        self.assertEqual(
            self.template_model.model_validate(neutral, by_alias=False, by_name=True).reward_amount,
            1111,
        )
        with self.assertRaises(ValidationError):
            self.template_model.model_validate(neutral, by_alias=True, by_name=False)
        self.assertEqual(
            self.template_model.model_validate(payload, by_alias=True, by_name=False).records[0].record_id,
            '02' * 32,
        )

    def test_filtered_serialization_and_optional_defaults(self):
        payload = self.payload()
        del payload['default_witness_commitment'], payload['requiredcertversion']
        model = self.template_model.model_validate(payload)
        self.assertIsNone(model.default_attestation)
        self.assertEqual(int(model.required_version), 1)
        self.assertEqual(model.model_dump(include={'reward_amount'}), {'coinbasevalue': 1111})
        self.assertEqual(
            model.model_dump(include={'records': {'__all__': {'record_id'}}}),
            {'transactions': [{'txid': '02' * 32}]},
        )
        self.assertNotIn('default_witness_commitment', model.model_dump(exclude_none=True))

    def test_wire_names_are_not_retained_as_model_aliases_or_schema_strings(self):
        wire_names = {
            'txid', 'previousblockhash', 'vsizelimit', 'transactions', 'longpollid',
            'noncerange', 'coinbaseaux', 'coinbasevalue',
            'default_witness_commitment', 'requiredcertversion',
        }

        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for key, item in value.items():
                    yield from strings(key)
                    yield from strings(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from strings(item)

        for model in (self.entry_model, self.metadata_model, self.template_model):
            with self.subTest(model=model.__name__):
                self.assertIs(model.model_config['cache_strings'], False)
                self.assertFalse(wire_names.intersection(model.model_fields))
                for field in model.model_fields.values():
                    self.assertIsNone(field.alias)
                    self.assertIsNone(field.validation_alias)
                    self.assertIsNone(field.serialization_alias)
                self.assertFalse(wire_names.intersection(strings(model.__pydantic_core_schema__)))
                self.assertFalse(wire_names.intersection(strings(model.model_json_schema())))

    def test_template_validation_error_releases_owned_raw_reference(self):
        payload = self.payload()
        payload['height'] = 'invalid-height'
        original = copy.deepcopy(payload)
        client = NodeClient('http://127.0.0.1:44107')
        with patch.object(client, 'raw_template', return_value=payload):
            try:
                client.get_template('unused-address')
            except NodeError as error:
                traceback = error.__traceback__
                inspected = False
                while traceback is not None:
                    if traceback.tb_frame.f_code.co_name == 'get_template':
                        inspected = True
                        self.assertNotIn('raw', traceback.tb_frame.f_locals)
                    traceback = traceback.tb_next
                self.assertTrue(inspected)
            else:
                self.fail('invalid input unexpectedly validated')
        self.assertEqual(payload, original)

    def assert_bridge_released_own_references(self, error):
        pending = [error]
        seen = set()
        inspected = set()
        forbidden = {
            '_neutral_fields': ('value', 'name', 'item', 'candidate'),
            '_read_wire_fields': ('value', 'translated', 'handler'),
            '_validate_with_policy': ('value', 'options', 'method'),
            'model_validate': ('obj', 'options'),
            '_write_wire_fields': ('self', 'values', 'translated', 'name', 'item', 'key', 'handler', 'info'),
        }
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
            traceback = current.__traceback__
            while traceback is not None:
                frame = traceback.tb_frame
                name = frame.f_code.co_name
                if frame.f_code.co_filename == rpc_fields.__file__ and name in forbidden:
                    inspected.add(name)
                    for reference in forbidden[name]:
                        self.assertNotIn(reference, frame.f_locals, (name, reference))
                    for container in ('aliases', 'translated'):
                        if container in frame.f_locals:
                            self.assertFalse(frame.f_locals[container])
                traceback = traceback.tb_next
        self.assertTrue(inspected)
        return inspected

    def test_native_translation_failures_release_temporary_references_and_policy(self):
        payload = self.payload()
        original = copy.deepcopy(payload)
        encode_word = rpc_fields.encode_word

        def reject_name(word):
            if word == 'transactions':
                raise RuntimeError('translation failed')
            return encode_word(word)

        with patch.object(rpc_fields, 'encode_word', reject_name):
            try:
                self.template_model.model_validate(payload, by_alias=True, by_name=False)
            except RuntimeError as error:
                inspected = self.assert_bridge_released_own_references(error)
                self.assertIn('_neutral_fields', inspected)
            else:
                self.fail('injected translation failure did not propagate')
        self.assertEqual(rpc_fields._name_policy.get(), (None, None))
        self.assertEqual(payload, original)
        self.assertEqual(self.template_model.model_validate(payload).reward_amount, 1111)

    def test_serialization_failure_releases_staging_references_without_mutating_model(self):
        model = self.template_model.model_validate(self.payload())
        original = model.model_dump(by_alias=False)
        decode_word = rpc_fields.decode_word

        def reject_name(word):
            if word == 'reward_amount':
                raise RuntimeError('translation failed')
            return decode_word(word)

        with patch.object(rpc_fields, 'decode_word', reject_name):
            try:
                model.model_dump()
            except Exception as error:
                inspected = self.assert_bridge_released_own_references(error)
                self.assertIn('_write_wire_fields', inspected)
            else:
                self.fail('injected serialization failure did not propagate')
        self.assertEqual(model.model_dump(by_alias=False), original)


if __name__ == '__main__':
    unittest.main()
