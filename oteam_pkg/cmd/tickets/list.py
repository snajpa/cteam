# oteam tickets list command - list ready tickets for self-assignment
import argparse
from pathlib import Path

import oteam
from oteam_pkg.state import tickets as state_tickets


def cmd_tickets_list(args: argparse.Namespace) -> None:
    root = oteam.find_project_root(Path(args.workdir))
    if not root:
        raise oteam.OTeamError(f"not an oteam workspace: {args.workdir}")

    ready_tickets = state_tickets.load_ready_tickets(root)

    if not ready_tickets:
        print("No ready tickets available for self-assignment.")
        print("Use 'oteam tickets create --tag ready' to add ready tickets.")
        return

    print("Ready tickets for self-assignment:")
    print("-" * 60)
    for ticket in ready_tickets:
        ticket_id = ticket["id"]
        title = (
            ticket["title"][:50] + "..."
            if len(ticket["title"]) > 50
            else ticket["title"]
        )
        tags = ticket.get("tags") or "[]"
        print(f"  T{ticket_id}: {title}")
        print(f"         Tags: {tags}")
        print()
