# Mode detection - detect OpenCode Plan vs Build mode from tmux pane
from typing import Tuple


PLAN_MODE_INDICATORS = ["▣  plan", "┃  plan", "plan", "Plan"]
BUILD_MODE_INDICATORS = ["▣  build", "┃  build", "build", "Build"]


def detect_mode(pane_output: str) -> str:
    """Detect Plan, Build, or Unknown from pane output."""
    if not pane_output:
        return "unknown"

    last_lines = pane_output[-500:] if len(pane_output) > 500 else pane_output
    last_lines_lower = last_lines.lower()

    for indicator in PLAN_MODE_INDICATORS:
        if indicator.lower() in last_lines_lower:
            return "plan"

    for indicator in BUILD_MODE_INDICATORS:
        if indicator.lower() in last_lines_lower:
            return "build"

    return "unknown"


def is_plan_mode(pane_output: str) -> bool:
    """Check if pane is in Plan mode."""
    return detect_mode(pane_output) == "plan"


def is_build_mode(pane_output: str) -> bool:
    """Check if pane is in Build mode."""
    return detect_mode(pane_output) == "build"


def is_unknown_mode(pane_output: str) -> bool:
    """Check if pane mode is unknown."""
    return detect_mode(pane_output) == "unknown"


def parse_mode_indicator(pane_output: str) -> Tuple[str, str]:
    """Parse the model selector line from pane output.

    Returns (model_selector, status_bar) or ("", "") if not found.
    """
    if not pane_output:
        return "", ""

    lines = pane_output.split("\n")
    for i, line in enumerate(lines):
        if "▣" in line or "▢" in line:
            selector = line.strip()
            if i + 1 < len(lines):
                status = lines[i + 1].strip()
                return selector, status
            return selector, ""

    return "", ""
