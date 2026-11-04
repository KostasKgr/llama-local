import unittest
from experiments.agentic_slots import check_events
from lib import core


class SlotTests(unittest.TestCase):
    def test_parallel_context_and_validation(self):
        p = core.load('swift15', {'parallel': 2, 'context': 131072})
        args = core.command(p)
        self.assertEqual(args[args.index('-np') + 1], '2')
        self.assertEqual(args[args.index('-c') + 1], '131072')
        self.assertEqual(core.load('swift15')['parallel'], 2)
        for slots in (0, -1, 3, True):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                core.load('swift15', {'parallel': slots, 'kv_unified': False})

    def test_unified_pool_does_not_split_context(self):
        p = core.load('swift15', {'parallel': 3, 'kv_unified': True})
        self.assertIn('--kv-unified', core.command(p))
        self.assertEqual(p['context'], 131072)
        p = core.load('swift15', {'kv_unified': False, 'parallel': 1})
        self.assertNotIn('--kv-unified', core.command(p))
        from unittest.mock import patch
        with patch.dict('os.environ', {'SWIFT_KV_UNIFIED': '0'}):
            self.assertFalse(core.load('swift15')['kv_unified'])
        with self.assertRaises(ValueError):
            core.load('swift15', {'kv_unified': 'false'})

    def test_context_checkpoints_are_per_profile(self):
        args = core.command(core.load('swift15'))
        self.assertEqual(args[args.index('--ctx-checkpoints') + 1], '2')
        args = core.command(core.load('swift15', {'ctx_checkpoints': 0}))
        self.assertEqual(args[args.index('--ctx-checkpoints') + 1], '0')
        with self.assertRaises(ValueError):
            core.load('swift15', {'ctx_checkpoints': -1})

    def test_agent_requires_successful_tools_and_evidence(self):
        message = {'role': 'assistant', 'provider': 'llama-local', 'model': 'test',
                   'stopReason': 'stop', 'content': [{'type': 'text', 'text': 'EVIDENCE=abc'}]}
        events = [
            {'type': 'tool_execution_start', 'toolName': 'read', 'toolCallId': '1',
             'args': {'path': 'api.py'}},
            {'type': 'tool_execution_end', 'toolCallId': '1', 'isError': False},
            {'type': 'message_end', 'message': message},
        ]
        self.assertEqual(check_events(events, ['api.py'], 'abc', 'test')['tool_calls'], 1)
        with self.assertRaises(ValueError):
            check_events(events, ['store.py'], 'abc', 'test')
        with self.assertRaises(ValueError):
            check_events(events, ['api.py'], 'wrong', 'test')
        events[1]['isError'] = True
        with self.assertRaises(ValueError):
            check_events(events, ['api.py'], 'abc', 'test')
        message['stopReason'] = 'length'
        with self.assertRaises(ValueError):
            check_events(events, [], 'abc', 'test')
