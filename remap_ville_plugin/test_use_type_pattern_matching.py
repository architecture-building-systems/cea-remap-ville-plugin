import unittest
from remap_use_types import match_pattern


class TestUseTypePatternMatching(unittest.TestCase):
    def test_any(self):
        self.assertTrue(match_pattern(("ANY", "ANY", "ANY", "ANY", "ANY", "ANY"), ("A", 0.5, "B", 0.3, "C", 0.2)))
        self.assertTrue(match_pattern(("A", 0.5, "ANY", 0.3, "ANY", "ANY"), ("A", 0.5, "B", 0.3, "C", 0.2)))
        self.assertFalse(match_pattern(("A", 0.5, "ANY", 0.3, "ANY", "ANY"), ("A", 0.4, "B", 0.3, "C", 0.2)))

    def test_no_match(self):
        self.assertFalse(match_pattern(
            ("A", 0.5, "B", 0.3, "C", 0.2),
            ("X", 1.0, "Y", 0.0, "Z", 0.0)
        ))

    def test_exact_match(self):
        self.assertTrue(match_pattern(
            ("A", 0.5, "B", 0.3, "C", 0.2),
            ("A", 0.5, "B", 0.3, "C", 0.2)
        ))



if __name__ == '__main__':
    unittest.main()
