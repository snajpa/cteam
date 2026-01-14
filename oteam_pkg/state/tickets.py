# Ticket state management - self-assignment extensions
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import sqlite3

import oteam


READY_TAGS = {"ready", "auto"}


def is_ready_ticket(ticket: Dict[str, Any]) -> bool:
    """Check if a ticket is ready for self-assignment."""
    if ticket.get("status") != oteam.TICKET_STATUS_OPEN:
        return False
    if ticket.get("assignee"):
        return False
    tags_str = ticket.get("tags") or "[]"
    try:
        tags = json.loads(tags_str)
        return bool(tags and any(t in READY_TAGS for t in tags))
    except (json.JSONDecodeError, TypeError):
        return False


def load_ready_tickets(root: Path) -> List[Dict[str, Any]]:
    """Load all tickets that are ready for self-assignment."""
    tickets = oteam.load_tickets(root, status=oteam.TICKET_STATUS_OPEN)
    return [t for t in tickets if is_ready_ticket(t)]


def can_grab_ticket(
    root: Path, ticket_id: str, agent_name: str
) -> tuple[bool, Optional[str]]:
    """Check if an agent can grab a ticket. Returns (can_grab, reason_if_not)."""
    ticket = oteam._find_ticket_by_id(root, ticket_id)
    if not ticket:
        return False, f"Ticket {ticket_id} not found"
    if ticket["status"] != oteam.TICKET_STATUS_OPEN:
        return False, f"Ticket {ticket_id} is not open"
    if ticket.get("assignee"):
        return False, f"Ticket {ticket_id} is already assigned"
    if not is_ready_ticket(ticket):
        return False, "Only ready tickets can be self-assigned"
    return True, None


def grab_ticket(
    root: Path, ticket_id: str, agent_name: str, user: str
) -> Dict[str, Any]:
    """Assign a ticket to an agent via self-assignment."""
    can_grab, reason = can_grab_ticket(root, ticket_id, agent_name)
    if not can_grab:
        raise oteam.OTeamError(reason)
    ticket = oteam.ticket_assign(
        root,
        ticket_id=ticket_id,
        assignee=agent_name,
        user=user,
        note="Self-assigned via ticket-grab",
    )
    if not ticket:
        raise oteam.OTeamError(f"Failed to assign ticket {ticket_id}")
    return ticket


def create_agent_branch(root: Path, agent_name: str, ticket_id: str) -> None:
    """Create a git branch for the agent's ticket work."""
    agent = next(
        (a for a in oteam.load_state(root)["agents"] if a["name"] == agent_name), None
    )
    if not agent:
        raise oteam.OTeamError(f"Agent {agent_name} not found")
    repo_dir = root / agent["repo_dir_rel"]
    branch_name = f"agent/{agent_name}/{ticket_id}"
    oteam.run_cmd(["git", "checkout", "-b", branch_name], cwd=repo_dir)


def get_ticket_branch_name(agent_name: str, ticket_id: str) -> str:
    """Get the expected branch name for a ticket."""
    return f"agent/{agent_name}/{ticket_id}"
