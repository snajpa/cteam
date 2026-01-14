# Tests for tmux.send_keys module
import unittest
from unittest.mock import patch, MagicMock
import oteam
from oteam_pkg.tmux import send_keys


class SendKeysTests(unittest.TestCase):
    @patch.object(oteam, "tmux_send_keys")
    def test_switch_to_plan_mode(self, mock_send: MagicMock) -> None:
        mock_send.return_value = None
        result = send_keys.switch_to_plan_mode("sess", "window")
        self.assertTrue(result)
        mock_send.assert_called_once_with("sess", "window", ["Tab"])

    @patch.object(oteam, "tmux_send_keys")
    def test_switch_to_build_mode(self, mock_send: MagicMock) -> None:
        mock_send.return_value = None
        result = send_keys.switch_to_build_mode("sess", "window")
        self.assertTrue(result)
        mock_send.assert_called_once_with("sess", "window", ["Tab"])

    @patch.object(oteam, "tmux_send_keys")
    def test_toggle_mode(self, mock_send: MagicMock) -> None:
        mock_send.return_value = None
        result = send_keys.toggle_mode("sess", "window")
        self.assertTrue(result)
        mock_send.assert_called_once_with("sess", "window", ["Tab"])

    @patch.object(oteam, "tmux_send_keys")
    def test_send_enter(self, mock_send: MagicMock) -> None:
        mock_send.return_value = None
        result = send_keys.send_enter("sess", "window")
        self.assertTrue(result)
        mock_send.assert_called_once_with("sess", "window", ["Enter"])

    @patch.object(oteam, "tmux_send_line")
    def test_send_text(self, mock_send: MagicMock) -> None:
        mock_send.return_value = True
        result = send_keys.send_text("sess", "window", "hello")
        self.assertTrue(result)
        mock_send.assert_called_once_with("sess", "window", "hello")

    @patch.object(oteam, "tmux")
    def test_capture_pane(self, mock_tmux: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "some output"
        mock_tmux.return_value = mock_result
        result = send_keys.capture_pane("sess", "window")
        self.assertEqual(result, "some output")

    @patch.object(oteam, "tmux")
    def test_capture_pane_empty_on_error(self, mock_tmux: MagicMock) -> None:
        mock_tmux.side_effect = Exception("fail")
        result = send_keys.capture_pane("sess", "window")
        self.assertEqual(result, "")

    def test_is_pane_ready(self) -> None:
        with patch.object(send_keys, "capture_pane") as mock_capture:
            mock_capture.return_value = "▣  plan · MiniMax"
            self.assertTrue(send_keys.is_pane_ready("sess", "window"))

    def test_is_pane_not_ready_thinking(self) -> None:
        with patch.object(send_keys, "capture_pane") as mock_capture:
            mock_capture.return_value = "▣  plan · MiniMax\nThinking:"
            self.assertFalse(send_keys.is_pane_ready("sess", "window"))

    def test_is_pane_not_ready_working(self) -> None:
        with patch.object(send_keys, "capture_pane") as mock_capture:
            mock_capture.return_value = "▣  build · MiniMax\nWorking:"
            self.assertFalse(send_keys.is_pane_ready("sess", "window"))


class SendKeysHelpersTests(unittest.TestCase):
    @patch.object(oteam, "tmux_send_keys")
    def test_send_keys_success(self, mock_send: MagicMock) -> None:
        mock_send.return_value = None
        result = send_keys._send_keys("sess", "win", ["a", "b"])
        self.assertTrue(result)

    @patch.object(oteam, "tmux_send_keys")
    def test_send_keys_failure(self, mock_send: MagicMock) -> None:
        mock_send.side_effect = Exception("fail")
        result = send_keys._send_keys("sess", "win", ["a"])
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
