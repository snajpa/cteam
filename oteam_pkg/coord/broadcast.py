# Broadcast - File-based agent messaging (replaces mail with 0 latency)
from pathlib import Path
from typing import Optional, Dict, Any
import json
import datetime as dt

import oteam


DIR_RUNTIME = ".oteam/runtime"
DIR_BROADCAST = f"{DIR_RUNTIME}/broadcasts"


def get_broadcast_dir(root: Path) -> Path:
    """Get the directory where broadcasts are stored."""
    return root / DIR_BROADCAST


def get_broadcast_file(root: Path) -> Path:
    """Get the main broadcast log file."""
    return get_broadcast_dir(root) / "broadcasts.log.md"


def broadcast(
    root: Path,
    sender: str,
    message: str,
    recipient: Optional[str] = None,
    priority: str = "normal",
) -> str:
    """Send a broadcast message to agents.

    Args:
        root: Workspace root path
        sender: Name of the sending agent
        message: Message content
        recipient: Optional specific recipient (None = all agents)
        priority: "normal" or "urgent"

    Returns:
        The broadcast ID
    """
    broadcast_dir = get_broadcast_dir(root)
    broadcast_dir.mkdir(parents=True, exist_ok=True)

    ts = oteam.now_iso()
    broadcast_id = oteam.ts_for_filename(ts)

    entry = _format_broadcast_entry(
        broadcast_id, ts, sender, message, recipient, priority
    )

    broadcast_file = get_broadcast_file(root)
    oteam.append_text(broadcast_file, entry)

    return broadcast_id


def _format_broadcast_entry(
    broadcast_id: str,
    ts: str,
    sender: str,
    message: str,
    recipient: Optional[str],
    priority: str,
) -> str:
    """Format a broadcast entry for the log."""
    lines = [
        f"--- Broadcast {broadcast_id} ---",
        f"Timestamp: {ts}",
        f"From: {sender}",
        f"Priority: {priority}",
    ]
    if recipient:
        lines.append(f"To: @{recipient}")
    else:
        lines.append("To: @all")
    lines.extend(["", message, "", ""])
    return "\n".join(lines)


def get_latest_broadcasts(root: Path, after: Optional[str] = None) -> list:
    """Get broadcasts after a given timestamp.

    Args:
        root: Workspace root path
        after: Optional timestamp to filter by

    Returns:
        List of broadcast entries
    """
    broadcast_file = get_broadcast_file(root)
    if not broadcast_file.exists():
        return []

    content = broadcast_file.read_text(encoding="utf-8")
    entries = _parse_broadcasts(content)

    if after:
        after_ts = oteam.iso_to_unix(after)
        entries = [e for e in entries if oteam.iso_to_unix(e["timestamp"]) > after_ts]

    return entries


def _parse_broadcasts(content: str) -> list:
    """Parse broadcast entries from log content."""
    entries = []
    current_entry = {}

    for line in content.split("\n"):
        if line.startswith("--- Broadcast "):
            if current_entry:
                entries.append(current_entry)
            current_entry = {
                "id": line.replace("--- Broadcast ", "").replace(" ---", "")
            }
        elif line.startswith("Timestamp: ") and current_entry:
            current_entry["timestamp"] = line.replace("Timestamp: ", "")
        elif line.startswith("From: ") and current_entry:
            current_entry["sender"] = line.replace("From: ", "")
        elif line.startswith("Priority: ") and current_entry:
            current_entry["priority"] = line.replace("Priority: ", "")
        elif line.startswith("To: @") and current_entry:
            current_entry["recipient"] = line.replace("To: @", "")
        elif line and not line.startswith("---") and "timestamp" in current_entry:
            if "message" not in current_entry:
                current_entry["message"] = line
            else:
                current_entry["message"] += "\n" + line

    if current_entry:
        entries.append(current_entry)

    return entries


def notify_agent(
    root: Path,
    sender: str,
    recipient: str,
    message: str,
    priority: str = "normal",
) -> str:
    """Send a direct notification to a specific agent.

    Args:
        root: Workspace root path
        sender: Name of the sending agent
        recipient: Agent to notify
        message: Message content
        priority: "normal" or "urgent"

    Returns:
        The broadcast ID
    """
    return broadcast(
        root, sender, f"@{recipient} {message}", recipient=None, priority=priority
    )


def clear_broadcasts(root: Path) -> None:
    """Clear all broadcasts (for testing)."""
    broadcast_file = get_broadcast_file(root)
    if broadcast_file.exists():
        broadcast_file.unlink()


def get_broadcast_count(root: Path) -> int:
    """Get the count of broadcasts."""
    broadcasts = get_latest_broadcasts(root)
    return len(broadcasts)
