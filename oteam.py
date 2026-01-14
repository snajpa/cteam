# Minimal oteam module for backward compatibility with tests
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import datetime as dt
import subprocess
import os


STATE_FILENAME = "oteam.json"
STATE_VERSION = 11

DIR_PROJECT_BARE = "project.git"
DIR_PROJECT_CHECKOUT = "project"
DIR_SEED = "seed"
DIR_SEED_EXTRAS = "seed-extras"
DIR_SHARED = "shared"
DIR_AGENTS = "agents"
DIR_MAIL = "shared/mail"
DIR_RUNTIME = ".oteam/runtime"

TICKET_STATUS_OPEN = "open"
TICKET_STATUS_BLOCKED = "blocked"
TICKET_STATUS_CLOSED = "closed"

DEFAULT_OPENCODE_SANDBOX = "danger-unrestricted"
DEFAULT_OPENCODE_APPROVAL = "never"
DEFAULT_AUTOSTART = "pm"


class OTeamError(Exception):
    pass


def build_state(
    root: Path,
    name: str,
    devs: int,
    mode: str,
    imported_from: Optional[str],
    opencode_cmd: str,
    opencode_model: Optional[str],
    sandbox: str,
    approval: str,
    search: bool,
    full_auto: bool,
    yolo: bool,
    autostart: str,
    router: bool,
) -> Dict[str, Any]:
    agents = []
    for i in range(devs):
        agents.append(
            {
                "name": f"dev{i + 1}",
                "role": "developer",
                "title": f"Developer {i + 1}",
                "repo_dir_rel": f"agents/dev{i + 1}/proj",
                "repo_dir_abs": str(root / f"agents/dev{i + 1}/proj"),
                "persona": "You are a software developer. You are thorough and leave code always better than you found it.",
                "opencode": {
                    "cmd": opencode_cmd,
                    "model": opencode_model or "sonnet",
                    "sandbox": sandbox,
                    "ask_for_approval": approval,
                    "search": search,
                    "full_auto": full_auto,
                    "yolo": yolo,
                    "autostart": autostart,
                },
            }
        )
    return {
        "version": STATE_VERSION,
        "name": name,
        "mode": mode,
        "imported_from": imported_from,
        "agents": agents,
        "coordination": {
            "assignment_from": ["pm"],
            "assignment_type": "Type: ASSIGNMENT",
            "router": router,
        },
    }


def save_state(root: Path, state: Dict[str, Any]) -> None:
    (root / STATE_FILENAME).write_text(json.dumps(state, indent=2))


def load_state(root: Path) -> Dict[str, Any]:
    return json.loads((root / STATE_FILENAME).read_text())


def ensure_shared_scaffold(root: Path, state: Dict[str, Any]) -> None:
    for d in [DIR_SHARED, DIR_MAIL]:
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / "message.md").write_text("")


def git_init_bare(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    run_cmd(["git", "init", "--bare"], cwd=path)


def git_clone(src: str, dst: Path) -> None:
    run_cmd(["git", "clone", src, str(dst)], cwd=dst.parent)


def run_cmd(
    cmd: List[str], cwd: Optional[Path] = None, capture: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture,
        text=True,
        check=True,
    )


def now_iso() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def ts_for_filename(ts: str) -> str:
    return (
        ts.replace("-", "")[:8]
        + "_"
        + ts[9:15].replace(":", "")
        + "p"
        + ts[16:22].replace(":", "")
    )


def find_project_root(p: Path) -> Optional[Path]:
    while p != p.parent:
        if (p / STATE_FILENAME).exists():
            return p
        p = p.parent
    return None


def slugify(s: str) -> str:
    return s.lower().replace(" ", "-")


def load_tickets(root: Path, status: Optional[str] = None) -> List[Dict[str, Any]]:
    tickets = []
    for f in (root / DIR_SHARED).glob("T*.json"):
        try:
            t = json.loads(f.read_text())
            if status is None or t.get("status") == status:
                tickets.append(t)
        except Exception:
            pass
    return tickets


def _find_ticket_by_id(root: Path, ticket_id: str) -> Optional[Dict[str, Any]]:
    for f in (root / DIR_SHARED).glob("T*.json"):
        try:
            t = json.loads(f.read_text())
            if str(t.get("id")) == ticket_id.lstrip("T"):
                return t
        except Exception:
            pass
    return None


def ticket_create(
    root: Path,
    title: str,
    description: str,
    creator: str,
    assignee: Optional[str],
    tags: Optional[List[str]],
) -> Dict[str, Any]:
    tickets = load_tickets(root)
    max_id = 0
    for t in tickets:
        try:
            max_id = max(max_id, int(str(t.get("id", 0))))
        except Exception:
            pass

    new_id = max_id + 1
    ticket = {
        "id": new_id,
        "title": title,
        "description": description,
        "status": TICKET_STATUS_OPEN,
        "assignee": assignee,
        "tags": json.dumps(tags) if tags else "[]",
        "creator": creator,
        "created": now_iso(),
    }

    (root / DIR_SHARED / f"T{new_id}.json").write_text(json.dumps(ticket, indent=2))
    return ticket


def ticket_assign(
    root: Path,
    ticket_id: str,
    assignee: str,
    user: str,
    note: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    ticket = _find_ticket_by_id(root, ticket_id)
    if not ticket:
        return None

    ticket["assignee"] = assignee
    ticket["assigned_by"] = user
    ticket["assigned_at"] = now_iso()
    if note:
        ticket["assignment_note"] = note

    (root / DIR_SHARED / f"T{ticket['id']}.json").write_text(
        json.dumps(ticket, indent=2)
    )
    return ticket


def ticket_block(root: Path, ticket_id: str, blocked_on: str, user: str) -> None:
    ticket = _find_ticket_by_id(root, ticket_id)
    if not ticket:
        raise OTeamError(f"Ticket {ticket_id} not found")

    ticket["status"] = TICKET_STATUS_BLOCKED
    ticket["blocked_on"] = blocked_on
    ticket["blocked_by"] = user
    ticket["blocked_at"] = now_iso()

    (root / DIR_SHARED / f"T{ticket['id']}.json").write_text(
        json.dumps(ticket, indent=2)
    )


def write_message(
    root: Path,
    state: Dict[str, Any],
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    nudge: bool = False,
    start_if_needed: bool = False,
    msg_type: str = "MESSAGE",
    ticket_id: Optional[str] = None,
) -> None:
    ts = now_iso()
    entry = format_message(ts, sender, recipient, subject, body, msg_type, ticket_id)
    msg_path = root / DIR_MAIL / recipient / "message.md"
    msg_path.write_text(entry)

    if nudge:
        pass


def format_message(
    ts: str,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    msg_type: str,
    ticket_id: Optional[str],
) -> str:
    lines = [
        f"From: {sender}",
        f"To: {recipient}",
        f"ts: {ts}",
        f"type: {msg_type}",
    ]
    if ticket_id:
        lines.append(f"ticket: {ticket_id}")
    lines.extend(["", subject, "", body, ""])
    return "\n".join(lines)


def _parse_entry(entry: str) -> tuple:
    lines = entry.split("\n")
    ts = sender = recipient = subject = body = ticket_id = msg_type = ""
    for i, line in enumerate(lines):
        if line.startswith("ts: "):
            ts = line[4:]
        elif line.startswith("From: "):
            sender = line[6:]
        elif line.startswith("To: "):
            recipient = line[4:]
        elif line.startswith("type: "):
            msg_type = line[6:]
        elif line.startswith("ticket: "):
            ticket_id = line[8:]
        elif i >= 5:
            body += line + "\n"
    return ts, sender, recipient, subject, body.strip(), ticket_id, msg_type


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.write_text(path.read_text() + text)
    else:
        path.write_text(text)


def tmux(args: List[str], capture: bool = False) -> subprocess.CompletedProcess:
    return run_cmd(["tmux"] + args, capture=capture)


def tmux_send_keys(session: str, window: str, keys: List[str]) -> None:
    for key in keys:
        tmux(["send-keys", "-t", f"{session}:{window}", key])


def tmux_send_line(session: str, window: str, line: str) -> None:
    tmux_send_keys(session, window, [line, "Enter"])


def iso_to_unix(ts: str) -> float:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
