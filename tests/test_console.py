import unittest
import os
from unittest.mock import patch

from a2rl_drone_training.console import format_table


class ConsoleTableTests(unittest.TestCase):
    def test_table_groups_and_aligns_metrics(self):
        output = format_table(
            [
                ("time", [("fps", "1,437"), ("iterations", "3 / 10")]),
                ("rollout", [("ep_rew_mean", "2.500")]),
            ]
        )

        self.assertIn("| time/", output)
        self.assertIn("|     fps", output)
        self.assertIn("1,437", output)
        self.assertIn("| rollout/", output)
        line_lengths = {len(line) for line in output.splitlines()}
        self.assertEqual(len(line_lengths), 1)

    def test_long_values_are_truncated_without_breaking_alignment(self):
        output = format_table(
            [("checkpoint", [("path", "a" * 100)])],
            max_value_width=12,
        )

        self.assertIn("aaaaaaaaa...", output)
        self.assertNotIn("a" * 13, output)

    def test_table_fits_narrow_and_wide_terminals(self):
        sections = [
            ("curriculum", [("qualification_samples", "32 min"),
                            ("local_start_probability", "0.083 " * 12)]),
            ("checkpoint", [("path", "checkpoints/" + "a" * 120)]),
        ]
        for width in (20, 40, 80, 120, 160):
            with self.subTest(width=width):
                lines = format_table(sections, terminal_width=width).splitlines()
                self.assertTrue(all(len(line) < width for line in lines))
                self.assertEqual(len({len(line) for line in lines}), 1)
                self.assertIn("32 min", "\n".join(lines))
                separators = [[i for i, char in enumerate(line) if char == "|"]
                              for line in lines[1:-1]]
                self.assertTrue(all(item == separators[0] for item in separators))

    def test_terminal_resize_is_detected_for_each_update(self):
        with patch("a2rl_drone_training.console.get_terminal_size",
                   side_effect=[os.terminal_size((40, 24)), os.terminal_size((100, 24))]):
            sections = [("gates", [("active_pass", "0.800 " * 12)])]
            narrow = format_table(sections)
            wide = format_table(sections)
        self.assertLess(len(narrow.splitlines()[0]), len(wide.splitlines()[0]))
        self.assertLess(len(narrow.splitlines()[0]), 40)

    def test_multiline_values_do_not_break_rows(self):
        output = format_table([("system", [("path", "first\nsecond\tthird")])])
        self.assertIn("first second third", output)
        self.assertEqual(len(output.splitlines()), 4)

    def test_tiny_terminal_and_empty_metrics(self):
        for width in (2, 8, 12):
            output = format_table([("time", [("fps", "1234")])], terminal_width=width)
            self.assertTrue(all(len(line) < width for line in output.splitlines()))
        self.assertEqual(format_table([], terminal_width=80), "")


if __name__ == "__main__":
    unittest.main()
