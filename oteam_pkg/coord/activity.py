# Activity logging - Track agent actions for real-time feed
from pathlib import Path
from typing import Optional
import datetime as dt

import oteam


DIR_RUNTIME = ".oteam/runtime"
DIR_ACTIVITY = f"{DIR_RUNTIME}/activity"


def get_activity_dir(root: Path) -> Path:
    """Get the activity directory."""
    return root / DIR_ACTIVITY


def get_activity_log(root: Path) -> Path:
    """Get the main activity log file."""
    return get_activity_dir(root) / "activity.log"


def log_activity(
    root: Path,
    agent: str,
    action: str,
    details: Optional[str] = None,
) -> None:
    """Log an agent activity.

    Args:
        root: Workspace root path
        agent: Agent name
        action: Action type (e.g., "grabbed", "pushed", "merged")
        details: Optional details about the action
    """
    activity_dir = get_activity_dir(root)
    activity_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {agent} {action}"
    if details:
        entry += f": {details}"
    entry += "\n"

    activity_log = get_activity_log(root)
    oteam.append_text(activity_log, entry)


def get_activity(root: Path, limit: int = 50) -> list:
    """Get recent activity entries.

    Args:
        root: Workspace root path
        limit: Maximum number of entries to return

    Returns:
        List of activity entries
    """
    activity_log = get_activity_log(root)
    if not activity_log.exists():
        return []

    content = activity_log.read_text(encoding="utf-8")
    lines = [l for l in content.split("\n") if l.strip()]

    return lines[-limit:]


def get_activity_since(root: Path, timestamp: str) -> list:
    """Get activity entries after a given timestamp.

    Args:
        root: Workspace root path
        timestamp: Time to filter by (HH:MM:SS format)

    Returns:
        List of activity entries after the timestamp
    """
    activity_log = get_activity_log(root)
    if not activity_log.exists():
        return []

    content = activity_log.read_text(encoding="utf-8")
    lines = [l for l in content.split("\n") if l.strip()]

    for i, line in enumerate(lines):
        if line.startswith(f"[{timestamp}]"):
            return lines[i + 1 :]

    return []


def clear_activity(root: Path) -> None:
    """Clear the activity log."""
    activity_log = get_activity_log(root)
    if activity_log.exists():
        activity_log.unlink()


def activity_count(root: Path) -> int:
    """Get the count of activity entries."""
    activity = get_activity(root)
    return len(activity)


def format_activity_entry(
    agent: str, action: str, details: Optional[str] = None
) -> str:
    """Format an activity entry for logging."""
    ts = dt.datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {agent} {action}"
    if details:
        entry += f": {details}"
    return entry


def log_ticket_grabbed(root: Path, agent: str, ticket_id: str) -> None:
    """Log that an agent grabbed a ticket."""
    log_activity(root, agent, f"grabbed ticket T{ticket_id}")


def log_ticket_pushed(root: Path, agent: str, ticket_id: str) -> None:
    """Log that an agent pushed a ticket."""
    log_activity(root, agent, f"pushed agent/{agent}/T{ticket_id}")


def log_ticket_merged(root: Path, agent: str, ticket_id: str, passed: bool) -> None:
    """Log that a ticket was merged."""
    result = "CI passed" if passed else "CI failed"
    log_activity(root, agent, f"merged T{ticket_id} ({result})")
