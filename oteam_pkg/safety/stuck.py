# Stuck flag - Notify PM when agent is stuck
from pathlib import Path
from typing import Optional
import datetime as dt

import oteam
from oteam_pkg.coord import activity


DIR_RUNTIME = ".oteam/runtime"
DIR_STUCK = f"{DIR_RUNTIME}/stuck"


def mark_stuck(root: Path, agent: str, reason: str) -> None:
    """Mark agent as stuck on a ticket.

    Args:
        root: Workspace root path
        agent: Agent name
        reason: Why they're stuck
    """
    stuck_dir = get_stuck_dir(root)
    stuck_dir.mkdir(parents=True, exist_ok=True)

    stuck_file = get_stuck_file(root, agent)
    ts = dt.datetime.now().isoformat()
    content = f"Stuck at: {ts}\nReason: {reason}\n"

    stuck_file.write_text(content, encoding="utf-8")

    activity.log_activity(root, agent, "marked as stuck", reason)

    _notify_pm_stuck(root, agent, reason)


def clear_stuck(root: Path, agent: str) -> None:
    """Clear stuck status for an agent."""
    stuck_file = get_stuck_file(root, agent)
    if stuck_file.exists():
        stuck_file.unlink()

    activity.log_activity(root, agent, "resolved stuck status")


def is_stuck(root: Path, agent: str) -> bool:
    """Check if an agent is marked as stuck."""
    stuck_file = get_stuck_file(root, agent)
    return stuck_file.exists()


def get_stuck_reason(root: Path, agent: str) -> Optional[str]:
    """Get the reason an agent is stuck."""
    stuck_file = get_stuck_file(root, agent)
    if stuck_file.exists():
        return stuck_file.read_text(encoding="utf-8")
    return None


def get_stuck_agents(root: Path) -> list:
    """Get list of stuck agents."""
    stuck_dir = get_stuck_dir(root)
    if not stuck_dir.exists():
        return []

    stuck_agents = []
    for f in stuck_dir.glob("*.txt"):
        agent = f.stem
        if is_stuck(root, agent):
            stuck_agents.append(agent)

    return stuck_agents


def get_stuck_dir(root: Path) -> Path:
    """Get the stuck directory."""
    return root / DIR_STUCK


def get_stuck_file(root: Path, agent: str) -> Path:
    """Get the stuck file for an agent."""
    return get_stuck_dir(root) / f"{agent}.txt"


def _notify_pm_stuck(root: Path, agent: str, reason: str) -> None:
    """Notify PM that an agent is stuck."""
    try:
        oteam.write_message(
            root,
            oteam.load_state(root),
            sender=agent,
            recipient="pm",
            subject=f"{agent} is stuck",
            body=f"Agent {agent} is stuck:\n\n{reason}\n\nPlease help!",
            nudge=True,
            start_if_needed=False,
        )
    except Exception:
        pass
