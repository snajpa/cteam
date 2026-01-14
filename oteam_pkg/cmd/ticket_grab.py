# oteam ticket-grab command - self-assign a ready ticket
import argparse
from pathlib import Path

import oteam
from oteam_pkg.state import tickets as state_tickets


def cmd_ticket_grab(args: argparse.Namespace) -> None:
    root = oteam.find_project_root(Path(args.workdir))
    if not root:
        raise oteam.OTeamError(f"not an oteam workspace: {args.workdir}")

    agent_name = args.agent
    ticket_id = args.ticket_id

    can_grab, reason = state_tickets.can_grab_ticket(root, ticket_id, agent_name)
    if not can_grab:
        raise oteam.OTeamError(reason)

    ticket = state_tickets.grab_ticket(root, ticket_id, agent_name, agent_name)
    print(f"Ticket T{ticket['id']} assigned to {agent_name}")

    branch_name = state_tickets.get_ticket_branch_name(agent_name, str(ticket["id"]))
    print(f"Branch: {branch_name}")
    print()
    print("Next steps:")
    print(f"  1. cd agents/{agent_name}/proj")
    print(f"  2. git checkout {branch_name}")
    print("  3. Read ticket context (if available)")
    print("  4. Press Tab to switch to Plan mode")
