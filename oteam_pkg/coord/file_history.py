# File history - Get recent changes for tickets
from pathlib import Path
from typing import List, Optional
import subprocess

import oteam
from oteam_pkg.coord import activity


def get_file_history(root: Path, files: List[str], limit: int = 5) -> str:
    """Get recent commits touching these files.

    Args:
        root: Workspace root path
        files: List of file paths
        limit: Maximum number of commits per file

    Returns:
        Formatted history string
    """
    if not files:
        return "(No files specified)"

    history_lines = []

    for f in files:
        commits = _get_file_commits(root, f, limit)
        if commits:
            for commit in commits:
                history_lines.append(f"- {commit} ({f})")

    if not history_lines:
        return "(No recent changes)"

    return "\n".join(history_lines)


def _get_file_commits(root: Path, file: str, limit: int) -> List[str]:
    """Get recent commits for a file."""
    project = root / oteam.DIR_PROJECT_CHECKOUT

    if not project.exists():
        return []

    try:
        cp = oteam.run_cmd(
            ["git", "log", "--oneline", f"-{limit}", "--", file],
            cwd=project,
            capture=True,
        )
        lines = cp.stdout.strip().split("\n")
        return [line for line in lines if line.strip()]
    except Exception:
        return []


def get_ticket_file_history(root: Path, ticket: dict, limit: int = 3) -> str:
    """Get history for files related to a ticket.

    Args:
        root: Workspace root path
        ticket: Ticket dict
        limit: Max commits per file

    Returns:
        Formatted history string
    """
    files = _extract_files_from_ticket(ticket)
    if not files:
        return "(No file history available)"

    return get_file_history(root, files, limit)


def _extract_files_from_ticket(ticket: dict) -> List[str]:
    """Extract file paths from ticket description."""
    desc = ticket.get("description", "")
    lines = desc.split("\n")
    files = []

    for line in lines:
        if line.strip().startswith("File:") or line.strip().startswith("Files:"):
            file_part = line.split(":", 1)[1].strip()
            for f in file_part.split(","):
                f = f.strip()
                if f and (f.endswith(".py") or f.endswith(".md") or f.endswith(".txt")):
                    files.append(f)

    return files


def get_related_changes(root: Path, agent_name: str, ticket_id: str) -> str:
    """Get changes relevant to an agent's current ticket.

    Args:
        root: Workspace root path
        agent_name: Agent name
        ticket_id: Ticket ID

    Returns:
        Formatted changes summary
    """
    ticket = oteam._find_ticket_by_id(root, ticket_id)
    if not ticket:
        return f"Ticket {ticket_id} not found"

    project = root / oteam.DIR_PROJECT_CHECKOUT
    if not project.exists():
        return "(Project directory not found)"

    changes = []

    try:
        cp = oteam.run_cmd(
            ["git", "log", "--oneline", "-10"],
            cwd=project,
            capture=True,
        )
        lines = [l for l in cp.stdout.strip().split("\n") if l.strip()]
        changes.extend(lines[:5])
    except Exception:
        pass

    if changes:
        return "Recent changes:\n" + "\n".join(f"- {c}" for c in changes)

    return "(No recent changes)"


def get_ticket_related_activity(root: Path, ticket_id: str) -> str:
    """Get activity related to a ticket."""
    activities = activity.get_activity(root)
    ticket_activities = [
        a for a in activities if f"T{ticket_id}" in a or ticket_id in a
    ]

    if not ticket_activities:
        return "(No activity for this ticket)"

    return "Activity:\n" + "\n".join(f"- {a}" for a in ticket_activities)


def suggest_test_command(root: Path, ticket: dict) -> str:
    """Suggest a test command based on ticket description."""
    desc = ticket.get("description", "").lower()

    if "test" in desc:
        if "test_auth" in desc:
            return "pytest tests/test_auth.py -v"
        if "unit" in desc:
            return "make test"
        return "pytest"

    return "make test"


def get_verify_command(root: Path, ticket: dict) -> str:
    """Get the verification command for a ticket."""
    desc = ticket.get("description", "")

    lines = desc.split("\n")
    for line in lines:
        if line.lower().startswith("verify:") or line.lower().startswith("test:"):
            return line.split(":", 1)[1].strip()

    return suggest_test_command(root, ticket)
