# Tab switching - send Tab key to tmux pane for Plan/Build mode toggle
from typing import List, Optional
from pathlib import Path

import oteam


def switch_to_plan_mode(session: str, window: str) -> bool:
    """Switch OpenCode pane to Plan mode by pressing Tab."""
    return _send_tab(session, window)


def switch_to_build_mode(session: str, window: str) -> bool:
    """Switch OpenCode pane to Build mode by pressing Tab."""
    return _send_tab(session, window)


def toggle_mode(session: str, window: str) -> bool:
    """Toggle between Plan and Build mode."""
    return _send_tab(session, window)


def send_enter(session: str, window: str) -> bool:
    """Send Enter key to tmux pane."""
    return _send_keys(session, window, ["Enter"])


def send_text(session: str, window: str, text: str) -> bool:
    """Send text to tmux pane."""
    oteam.tmux_send_line(session, window, text)
    return True


def _send_tab(session: str, window: str) -> bool:
    """Send Tab key to tmux pane."""
    return _send_keys(session, window, ["Tab"])


def _send_keys(session: str, window: str, keys: List[str]) -> bool:
    """Send keys to tmux pane."""
    try:
        oteam.tmux_send_keys(session, window, keys)
        return True
    except Exception:
        return False


def capture_pane(session: str, window: str) -> str:
    """Capture tmux pane output."""
    try:
        cp = oteam.tmux(
            ["capture-pane", "-p", "-t", f"{session}:{window}"], capture=True
        )
        return cp.stdout or ""
    except Exception:
        return ""


def is_pane_ready(session: str, window: str) -> bool:
    """Check if OpenCode pane is ready for input."""
    output = capture_pane(session, window)
    return "▣" in output and "Thinking:" not in output and "Working:" not in output


def wait_for_ready(session: str, window: str, timeout: float = 30.0) -> bool:
    """Wait for OpenCode pane to be ready for input."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_pane_ready(session, window):
            return True
        time.sleep(0.5)
    return False
