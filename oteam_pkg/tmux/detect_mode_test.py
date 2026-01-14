# Tests for tmux.detect_mode module
import unittest

from oteam_pkg.tmux import detect_mode


class DetectModeTests(unittest.TestCase):
    def test_detect_plan_mode_symbol(self) -> None:
        pane = "some output\n▣  plan · MiniMax-M2.1\nmore output"
        self.assertEqual(detect_mode.detect_mode(pane), "plan")

    def test_detect_plan_mode_pipe(self) -> None:
        pane = "output\n┃  plan  MiniMax-M2.1 MiniMax\nrest"
        self.assertEqual(detect_mode.detect_mode(pane), "plan")

    def test_detect_plan_mode_word(self) -> None:
        pane = "text\nplan mode active\nmore"
        self.assertEqual(detect_mode.detect_mode(pane), "plan")

    def test_detect_build_mode_symbol(self) -> None:
        pane = "output\n▣  build · MiniMax-M2.1\nrest"
        self.assertEqual(detect_mode.detect_mode(pane), "build")

    def test_detect_build_mode_pipe(self) -> None:
        pane = "text\n┃  build  MiniMax-M2.1 MiniMax\nmore"
        self.assertEqual(detect_mode.detect_mode(pane), "build")

    def test_detect_build_mode_word(self) -> None:
        pane = "stuff\nbuild mode here\nthings"
        self.assertEqual(detect_mode.detect_mode(pane), "build")

    def test_detect_unknown_when_empty(self) -> None:
        self.assertEqual(detect_mode.detect_mode(""), "unknown")
        self.assertEqual(detect_mode.detect_mode("   "), "unknown")

    def test_detect_unknown_when_no_indicator(self) -> None:
        pane = "some random output\nno mode indicator here\njust text"
        self.assertEqual(detect_mode.detect_mode(pane), "unknown")

    def test_plan_takes_precedence_over_build(self) -> None:
        pane = "both\n▣  build · MiniMax\n┃  plan  MiniMax\nindicators"
        self.assertEqual(detect_mode.detect_mode(pane), "plan")

    def test_is_plan_mode(self) -> None:
        self.assertTrue(detect_mode.is_plan_mode("▣  plan · MiniMax"))
        self.assertFalse(detect_mode.is_plan_mode("▣  build · MiniMax"))

    def test_is_build_mode(self) -> None:
        self.assertTrue(detect_mode.is_build_mode("▣  build · MiniMax"))
        self.assertFalse(detect_mode.is_build_mode("▣  plan · MiniMax"))

    def test_is_unknown_mode(self) -> None:
        self.assertTrue(detect_mode.is_unknown_mode("no indicators"))
        self.assertFalse(detect_mode.is_unknown_mode("▣  plan"))


class ParseModeIndicatorTests(unittest.TestCase):
    def test_parse_model_selector(self) -> None:
        pane = "output\n▣  plan · MiniMax-M2.1\n┃  Plan  MiniMax-M2.1 MiniMax"
        selector, status = detect_mode.parse_mode_indicator(pane)
        self.assertIn("▣", selector)
        self.assertIn("plan", selector.lower())

    def test_parse_status_bar(self) -> None:
        pane = "text\n▣  build · MiniMax\n┃  Build  MiniMax-M2.1 MiniMax"
        selector, status = detect_mode.parse_mode_indicator(pane)
        self.assertIn("┃", status)
        self.assertIn("build", status.lower())

    def test_parse_empty_output(self) -> None:
        selector, status = detect_mode.parse_mode_indicator("")
        self.assertEqual(selector, "")
        self.assertEqual(status, "")


if __name__ == "__main__":
    unittest.main()
