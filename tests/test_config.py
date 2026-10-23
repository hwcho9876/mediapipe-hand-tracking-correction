import unittest
from handlab.config import Config


class ConfigTests(unittest.TestCase):
    def test_roundtrip(self):
        self.assertEqual(Config.from_dict(Config().to_dict()), Config())

    def test_invalid(self):
        for change in ({'min_cutoff': 0}, {'beta': float('nan')}, {'ema_enabled': 1},
                       {'suspect_confirm': 1.5}, {'unknown': 1}, {'hold_seconds': 0.001}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Config.from_dict({**Config().to_dict(), **change})
