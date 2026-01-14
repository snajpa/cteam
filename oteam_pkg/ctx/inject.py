# Context injection - generate ticket-specific context for self-assignment
from pathlib import Path
from typing import Any, Dict, Optional
import json
import re

import oteam


DIR_RUNTIME = ".oteam/runtime"
DIR_CONTEXTS = f"{DIR_RUNTIME}/contexts"


def get_contexts_dir(root: Path) -> Path:
    """Get the directory where ticket contexts are stored."""
    return root / DIR_CONTEXTS


def get_context_file_path(root: Path, ticket_id: str) -> Path:
    """Get the path to a ticket's context file."""
    return get_contexts_dir(root) / f"T{ticket_id}.md"


def generate_context(
    root: Path,
    ticket: Dict[str, Any],
    agent_name: str,
) -> str:
    """Generate context content from template for a ticket."""
    ticket_id = str(ticket["id"])
    context_file = get_context_file_path(root, ticket_id)
    context_dir = get_contexts_dir(root)
    context_dir.mkdir(parents=True, exist_ok=True)

    template_path = Path(__file__).parent / "templates" / "ticket_context.md"
    if not template_path.exists():
        return _default_context(root, ticket, agent_name)

    template = template_path.read_text()

    previous_ticket = _find_previous_ticket(root, ticket)
    previous_file = _find_previous_file(root, ticket)

    related = _find_related_agent_work(root, ticket)

    test_command = _guess_test_command(root, ticket)
    verify_command = _guess_verify_command(root, ticket)

    scope_in = _parse_scope(ticket.get("description", ""), "IN") or [
        "See ticket description"
    ]
    scope_out = _parse_scope(ticket.get("description", ""), "OUT") or [
        "See ticket description"
    ]

    context = template.replace("{{ticket_id}}", ticket_id)
    context = context.replace("{{title}}", ticket.get("title", "Untitled"))
    context = context.replace("{{role}}", "developer")
    context = context.replace("{{focus}}", _guess_focus(root, ticket))
    context = context.replace("{{context_file_path}}", str(context_file))
    context = context.replace("{{agent_name}}", agent_name)

    if previous_ticket:
        context = context.replace(
            "{% if previous_ticket %}", f"- Previous: T{previous_ticket} merged"
        )
        context = context.replace("{% else %}", "")
    else:
        context = context.replace("{% if previous_ticket %}", "")

    context = context.replace("{{test_command}}", test_command)
    context = context.replace("{{verify_command}}", verify_command)
    context = context.replace("{{related_agent}}", related.get("agent", ""))
    context = context.replace("{{related_ticket}}", related.get("ticket", ""))
    context = context.replace("{{previous_ticket}}", previous_ticket or "")
    context = context.replace("{{previous_file}}", previous_file or "")

    scope_in_block = "\n".join(f"- {item}" for item in scope_in)
    scope_out_block = "\n".join(f"- {item}" for item in scope_out)

    context = re.sub(
        r"\{% for item in scope_in %\}\n.*?\{% endfor %\}",
        scope_in_block,
        context,
        flags=re.DOTALL,
    )
    context = re.sub(
        r"\{% for item in scope_out %\}\n.*?\{% endfor %\}",
        scope_out_block,
        context,
        flags=re.DOTALL,
    )

    context = re.sub(
        r"\{% if related_agent %\}.*?\{% endif %\}", "", context, flags=re.DOTALL
    )

    return context


def _default_context(root: Path, ticket: Dict[str, Any], agent_name: str) -> str:
    """Generate a simple default context if template is missing."""
    ticket_id = str(ticket["id"])
    return f"""# T{ticket_id}: {ticket.get("title", "Untitled")}

**Role:** developer
**Focus:** See ticket description

---

## Context
- Test to run: make test

---

## Scope
- See ticket description

---

## Verify
```
make test
```

---

## Start
```
cd agents/{agent_name}/proj
git checkout agent/{agent_name}/T{ticket_id}
```

**Just do it. Only ask if CI fails.**
"""


def _find_previous_ticket(root: Path, ticket: Dict[str, Any]) -> Optional[str]:
    """Find the most recent closed ticket that modified files in this ticket."""
    tickets = oteam.load_tickets(root, status=oteam.TICKET_STATUS_CLOSED)
    ticket_files = _extract_files_from_ticket(ticket)
    if not ticket_files:
        for t in reversed(tickets):
            return str(t["id"])
    return None


def _find_previous_file(root: Path, ticket: Dict[str, Any]) -> Optional[str]:
    """Find which file was modified by the previous ticket."""
    return None


def _find_related_agent_work(root: Path, ticket: Dict[str, Any]) -> Dict[str, str]:
    """Find other agents working on related files."""
    return {}


def _extract_files_from_ticket(ticket: Dict[str, Any]) -> list:
    """Extract file paths from ticket description."""
    return []


def _guess_test_command(root: Path, ticket: Dict[str, Any]) -> str:
    """Guess the relevant test command for this ticket."""
    return "make test"


def _guess_verify_command(root: Path, ticket: Dict[str, Any]) -> str:
    """Guess the verification command for this ticket."""
    return "make test"


def _guess_focus(root: Path, ticket: Dict[str, Any]) -> str:
    """Guess the focus area for this ticket."""
    return "See ticket description"


def _parse_scope(description: str, marker: str) -> Optional[list]:
    """Parse scope IN/OUT from ticket description."""
    lines = description.split("\n")
    in_scope = []
    in_block = False
    for line in lines:
        if f"Scope ({marker})" in line or f"Scope ({marker.lower()})" in line:
            in_block = True
            continue
        if in_block:
            if line.strip().startswith("- "):
                in_scope.append(line.strip()[2:])
            elif (
                line.strip() and not line.startswith(" ") and not line.startswith("\t")
            ):
                break
    return in_scope if in_scope else None


def inject_context(root: Path, ticket: Dict[str, Any], agent_name: str) -> Path:
    """Generate and write context file for a ticket."""
    context_file = get_context_file_path(root, str(ticket["id"]))
    context_content = generate_context(root, ticket, agent_name)
    context_file.write_text(context_content, encoding="utf-8")
    return context_file


def get_context(root: Path, ticket_id: str) -> Optional[str]:
    """Get the context for a ticket if it exists."""
    context_file = get_context_file_path(root, ticket_id)
    if context_file.exists():
        return context_file.read_text(encoding="utf-8")
    return None
