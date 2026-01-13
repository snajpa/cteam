#!/usr/bin/env python3
"""
OpenCode Team (oteam) — single-file multi-agent manager for tmux + OpenCode CLI + git repos.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import fcntl
import json
import os
import re
import secrets
import string
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import readline
import textwrap
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

STATE_FILENAME = "oteam.json"
STATE_VERSION = 11

DIR_PROJECT_BARE = "project.git"
DIR_PROJECT_CHECKOUT = "project"
DIR_SEED = "seed"
DIR_SEED_EXTRAS = "seed-extras"
DIR_SHARED = "shared"
DIR_SHARED_DRIVE = f"{DIR_SHARED}/drive"
TICKETS_JSON_REL = f"{DIR_SHARED}/TICKETS.json"
DIR_AGENTS = "agents"
DIR_LOGS = "logs"
DIR_MAIL = f"{DIR_SHARED}/mail"
DIR_RUNTIME = f"{DIR_SHARED}/runtime"
CUSTOMER_WINDOW = "customer"
ROUTER_WINDOW = "router"
DEFAULT_GIT_BRANCH = "main"
DEFAULT_OPENCODE_CMD = "opencode"
DEFAULT_OPENCODE_SANDBOX = "danger-full-access"
DEFAULT_OPENCODE_APPROVAL = "never"
DEFAULT_OPENCODE_SEARCH = True
DEFAULT_AUTOSTART = "all"
TELEGRAM_CONFIG_REL = f"{DIR_RUNTIME}/telegram.json"
TELEGRAM_MAX_MESSAGE = 3900
DASHBOARD_JSON_REL = f"{DIR_RUNTIME}/dashboard.json"
STATUS_BOARD_MD_REL = f"{DIR_SHARED}/STATUS_BOARD.md"
BRIEF_MD_REL = f"{DIR_SHARED}/BRIEF.md"
CUSTOMER_BURST_BUFFER_REL = f"{DIR_RUNTIME}/customer_burst_buffer.md"
SPAWN_REQUESTS_REL = f"{DIR_RUNTIME}/spawn_requests.json"

ASSIGNMENT_TYPE = "ASSIGNMENT"
AGENT_ROLES = ["project_manager", "developer", "tester", "researcher", "architect"]
TICKET_STATUS_OPEN = "open"
TICKET_STATUS_BLOCKED = "blocked"
TICKET_STATUS_CLOSED = "closed"
TICKET_MIGRATION_FLAG = f"{DIR_RUNTIME}/tickets_migration_notified"

_nudge_history: Dict[str, Tuple[float, Set[str]]] = {}
_router_noise_until: Dict[str, float] = {}
_nudge_backoff: Dict[str, Tuple[int, float]] = {}


class OTeamError(RuntimeError):
    pass


def now_utc() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat(timespec="seconds")


def ts_for_filename(ts: str) -> str:
    safe = ts.replace(":", "").replace("-", "").replace("+", "p")
    safe = safe.replace("Z", "z").replace("T", "_")
    return safe


def iso_to_unix(ts: str) -> float:
    try:
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def customer_state_path(root: Path) -> Path:
    return root / DIR_RUNTIME / "customer_state.json"


def load_customer_state(root: Path) -> Dict[str, Any]:
    p = customer_state_path(root)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_customer_state(root: Path, state: Dict[str, Any]) -> None:
    try:
        atomic_write_text(
            customer_state_path(root), json.dumps(state, indent=2, sort_keys=True)
        )
    except Exception:
        pass


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "project"


def _looks_like_git_url(src: str) -> bool:
    s = src.strip()
    return bool(re.match(r"^(https?://|ssh://|git@|file://|.+\.git$)", s))


TARBALL_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz")


def _looks_like_tarball(src: str) -> bool:
    s = src.strip().lower()
    return any(s.endswith(ext) for ext in TARBALL_SUFFIXES)


def _safe_extract_tarball(tf: tarfile.TarFile, dest: Path) -> None:
    dest_abs = dest.resolve()
    for member in tf.getmembers():
        member_path = dest_abs / member.name
        try:
            member_abs = member_path.resolve()
        except Exception:
            member_abs = member_path
        if not str(member_abs).startswith(str(dest_abs)):
            raise OTeamError(
                f"tarball contains unsafe path outside extract dir: {member.name}"
            )
        if member.islnk() or member.issym():
            target = Path(member.linkname or "")
            if target.is_absolute():
                raise OTeamError(
                    f"tarball contains symlink with absolute target: {member.name}"
                )
            target_path = (member_path.parent / target).resolve()
            if not str(target_path).startswith(str(dest_abs)):
                raise OTeamError(
                    f"tarball contains symlink outside extract dir: {member.name}"
                )
    tf.extractall(dest_abs)


def _collapse_single_dir(base: Path) -> Path:
    entries = [p for p in base.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return base


def _read_seed_from_dir(src_root: Path) -> Optional[str]:
    for candidate in [src_root / "SEED.md", src_root / "seed" / "SEED.md"]:
        try:
            if candidate.exists() and candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return text
        except Exception:
            continue
    return None


def _maybe_extract_tarball_src(
    src: str,
) -> Tuple[Optional[tempfile.TemporaryDirectory], Optional[Path], Optional[str]]:
    src_path = Path(src).expanduser()
    tar_like = _looks_like_tarball(src)
    tar_path: Optional[Path] = None
    tmpdir: Optional[tempfile.TemporaryDirectory] = None

    if tar_like and src_path.exists() and src_path.is_file():
        if not tarfile.is_tarfile(src_path):
            raise OTeamError(f"--src tarball is not readable: {src}")
        tmpdir = tempfile.TemporaryDirectory(prefix="oteam_import_")
        tar_path = src_path
    elif tar_like and src.strip().lower().startswith(("http://", "https://")):
        tmpdir = tempfile.TemporaryDirectory(prefix="oteam_import_")
        tar_path = Path(tmpdir.name) / "src.tar"
        try:
            with urllib.request.urlopen(src, timeout=60.0) as resp:
                tar_path.write_bytes(resp.read())
        except Exception as e:
            tmpdir.cleanup()
            raise OTeamError(f"failed to download tarball from {src}: {e}") from e
    elif tar_like:
        raise OTeamError(f"--src tarball not found or unreadable: {src}")
    else:
        return None, None, None

    if not tarfile.is_tarfile(tar_path):
        tmpdir.cleanup()
        raise OTeamError(f"--src tarball is not readable: {src}")

    extract_root = Path(tmpdir.name) / "extract"
    mkdirp(extract_root)
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            _safe_extract_tarball(tf, extract_root)
    except Exception:
        tmpdir.cleanup()
        raise

    src_root = _collapse_single_dir(extract_root)
    try:
        has_any = any(src_root.iterdir())
    except Exception:
        has_any = False
    if not has_any:
        tmpdir.cleanup()
        raise OTeamError(f"--src tarball appears to be empty: {src}")

    seed_text = _read_seed_from_dir(src_root)
    return tmpdir, src_root, seed_text


def _maybe_trigger_import_kickoff_on_customer_message(
    root: Path, state: Dict[str, Any]
) -> None:
    try:
        if state.get("mode") != "import":
            return
    except Exception:
        return

    try:
        cust_state = load_customer_state(root)
    except Exception:
        cust_state = {}

    pending = bool(cust_state.get("import_pending"))
    kickoff_done = bool(cust_state.get("import_kickoff_done"))
    if not pending or kickoff_done:
        return

    cust_state["import_kickoff_done"] = True
    cust_state["import_pending"] = False
    save_customer_state(root, cust_state)

    try:
        state = load_state(root)
    except Exception:
        pass

    try:
        tmux_cfg = state.setdefault("tmux", {})
        tmux_cfg["paused"] = False
        save_state(root, state)
    except Exception:
        pass

    try:
        write_message(
            root,
            state,
            sender="oteam",
            recipient="pm",
            subject="Import kickoff: infer goals + plan",
            body=(
                "This is an imported codebase.\n\n"
                "1) Inspect README/docs/code and fill shared/GOALS.md with inferred goals + open questions.\n"
                "2) Write shared/PLAN.md (high-level only).\n"
                "3) Create tickets for discovery tasks and assign them with `oteam tickets` + `oteam assign --ticket ...`.\n"
                "4) Keep agents coordinated; they will wait for ticketed assignments.\n"
            ),
            msg_type=ASSIGNMENT_TYPE,
            task="PM-IMPORT-KICKOFF",
            nudge=True,
            start_if_needed=True,
        )
    except Exception:
        pass

    try:
        if cust_state.get("import_recon"):
            for who, subj, task, body in [
                (
                    "architect",
                    "Recon: architecture map",
                    "RECON-ARCH",
                    "Create a short architecture map and send to PM. Avoid code changes.",
                ),
                (
                    "tester",
                    "Recon: how to run/tests",
                    "RECON-TEST",
                    "Figure out how to run the project and tests; write a short run/test note to PM.",
                ),
                (
                    "researcher",
                    "Recon: dependencies",
                    "RECON-DEPS",
                    "Inventory dependencies/versions; send a short risk/opportunity note to PM.",
                ),
            ]:
                if any(a.get("name") == who for a in state.get("agents", [])):
                    write_message(
                        root,
                        state,
                        sender="oteam",
                        recipient=who,
                        subject=subj,
                        body=body,
                        msg_type=ASSIGNMENT_TYPE,
                        task=task,
                        nudge=True,
                        start_if_needed=True,
                    )
    except Exception:
        pass

    try:
        ensure_tmux(root, state, launch_opencode=True)
    except Exception:
        pass


def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def log_line(root: Path, message: str) -> None:
    ts = now_iso()
    line = f"[{ts}] {message}"
    print(line)
    try:
        log_dir = root / DIR_LOGS
        mkdirp(log_dir)
        with (log_dir / "router.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def tickets_json_path(root: Path, state: Optional[Dict[str, Any]] = None) -> Path:
    rel = None
    if state:
        rel = (state.get("tickets") or {}).get("json_rel")
    rel = rel or TICKETS_JSON_REL
    return (root / rel).resolve()


@contextmanager
def ticket_lock(root: Path, state: Dict[str, Any]):
    lock_path = tickets_json_path(root, state).with_suffix(".lock")
    mkdirp(lock_path.parent)
    f = lock_path.open("a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        f.close()


def _ticket_store_default() -> Dict[str, Any]:
    return {"meta": {"next_id": 1}, "tickets": []}


def load_tickets(root: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    p = tickets_json_path(root, state)
    if not p.exists():
        store = _ticket_store_default()
        save_tickets(root, state, store)
        return store
    try:
        store = json.loads(p.read_text(encoding="utf-8"))
        if "meta" not in store or "tickets" not in store:
            raise ValueError("missing keys")
    except Exception:
        store = _ticket_store_default()
        save_tickets(root, state, store)
    meta = store.setdefault("meta", {})
    meta.setdefault("next_id", 1)
    if not isinstance(store.get("tickets"), list):
        store["tickets"] = []
    return store


def save_tickets(root: Path, state: Dict[str, Any], store: Dict[str, Any]) -> None:
    atomic_write_text(
        tickets_json_path(root, state),
        json.dumps(store, indent=2, sort_keys=True) + "\n",
    )


def _next_ticket_id(meta: Dict[str, Any], store: Dict[str, Any]) -> str:
    existing_ids: Set[str] = set()
    for t in store.get("tickets", []):
        if t.get("id"):
            existing_ids.add(str(t.get("id")).upper())
    try:
        n = int(meta.get("next_id", 1))
    except Exception:
        n = 1
    max_attempts = 10000
    for _ in range(max_attempts):
        tid = f"T{n:03d}"
        if tid.upper() not in existing_ids:
            meta["next_id"] = n + 1
            return tid
        n += 1
    raise OTeamError("could not generate unique ticket ID after many attempts")


def _find_ticket(store: Dict[str, Any], ticket_id: str) -> Optional[Dict[str, Any]]:
    tid = (ticket_id or "").upper()
    for t in store.get("tickets", []):
        if str(t.get("id", "")).upper() == tid:
            return t
    return None


def _add_history(ticket: Dict[str, Any], event: Dict[str, Any]) -> None:
    hist = ticket.setdefault("history", [])
    hist.append(event)


def _format_ticket_summary(store: Dict[str, Any]) -> str:
    tickets = store.get("tickets") or []
    active = [
        t
        for t in tickets
        if t.get("status") in {TICKET_STATUS_OPEN, TICKET_STATUS_BLOCKED}
    ]
    if not active:
        return "No open or blocked tickets."
    lines = []
    for t in sorted(active, key=lambda x: x.get("id", "")):
        tid = t.get("id", "")
        status = t.get("status", TICKET_STATUS_OPEN)
        assignee = t.get("assignee") or "unassigned"
        title = t.get("title", "")
        block_note = ""
        if status == TICKET_STATUS_BLOCKED and t.get("blocked_on"):
            block_note = f" (blocked on {t.get('blocked_on')})"
        lines.append(f"- {tid} [{status}] @{assignee} — {title}{block_note}")
    return "\n".join(lines)


def _ticket_required(store: Dict[str, Any], ticket_id: str) -> Dict[str, Any]:
    t = _find_ticket(store, ticket_id)
    if not t:
        raise OTeamError(f"ticket not found: {ticket_id}")
    return t


def _ticket_summary_block(ticket: Dict[str, Any]) -> str:
    assignee = ticket.get("assignee") or "unassigned"
    status = ticket.get("status", TICKET_STATUS_OPEN)
    blocked_on = ticket.get("blocked_on") or ""
    lines = [
        f"Ticket: {ticket.get('id', '')} — {ticket.get('title', '')}",
        f"Status: {status}"
        + (
            f" (blocked on {blocked_on})"
            if status == TICKET_STATUS_BLOCKED and blocked_on
            else ""
        ),
        f"Assignee: {assignee}",
    ]
    desc = (ticket.get("description") or "").strip()
    if desc:
        lines.append("")
        lines.append(desc)
    return "\n".join(lines).strip()


def _notify_ticket_change(
    root: Path,
    state: Dict[str, Any],
    ticket: Dict[str, Any],
    *,
    recipients: List[str],
    subject: str,
    body: str,
    nudge: bool = True,
    start_if_needed: bool = False,
) -> None:
    summary = _ticket_summary_block(ticket)
    combined = f"{summary}\n\n{body.strip()}" if body.strip() else summary
    for r in recipients:
        if not r:
            continue
        write_message(
            root,
            state,
            sender="pm",
            recipient=r,
            subject=subject,
            body=combined,
            msg_type="MESSAGE",
            task=None,
            nudge=nudge,
            start_if_needed=start_if_needed,
            ticket_id=ticket.get("id"),
        )


def _parse_tags(tag_str: Optional[str]) -> List[str]:
    if not tag_str:
        return []
    tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    return list(dict.fromkeys(tags))


def ticket_create(
    root: Path,
    state: Dict[str, Any],
    *,
    title: str,
    description: str,
    creator: str,
    assignee: Optional[str],
    tags: Optional[List[str]] = None,
    assign_note: Optional[str] = None,
) -> Dict[str, Any]:
    with ticket_lock(root, state):
        store = load_tickets(root, state)
        ts = now_iso()
        tid = _next_ticket_id(store["meta"], store)
        ticket = {
            "id": tid,
            "title": title.strip(),
            "description": description.strip(),
            "status": TICKET_STATUS_OPEN,
            "assignee": assignee or None,
            "blocked_on": "",
            "created_at": ts,
            "updated_at": ts,
            "created_by": creator,
            "tags": tags or [],
            "history": [],
        }
        _add_history(
            ticket,
            {"ts": ts, "type": "created", "user": creator, "note": title.strip()},
        )
        if assignee:
            _add_history(
                ticket,
                {
                    "ts": ts,
                    "type": "assigned",
                    "user": creator,
                    "from": None,
                    "to": assignee,
                    "note": (assign_note or "").strip(),
                },
            )
        store["tickets"].append(ticket)
        save_tickets(root, state, store)
        return ticket


def ticket_assign(
    root: Path,
    state: Dict[str, Any],
    *,
    ticket_id: str,
    assignee: str,
    user: str,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    with ticket_lock(root, state):
        store = load_tickets(root, state)
        t = _ticket_required(store, ticket_id)
        if t.get("status") == TICKET_STATUS_CLOSED:
            raise OTeamError(f"ticket {ticket_id} is closed; reopen before assigning")
        prev = t.get("assignee")
        t["assignee"] = assignee
        t["updated_at"] = now_iso()
        _add_history(
            t,
            {
                "ts": t["updated_at"],
                "type": "assigned",
                "user": user,
                "from": prev,
                "to": assignee,
                "note": (note or "").strip(),
            },
        )
        save_tickets(root, state, store)
        return t


def ticket_block(
    root: Path,
    state: Dict[str, Any],
    *,
    ticket_id: str,
    user: str,
    blocked_on: str,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    with ticket_lock(root, state):
        store = load_tickets(root, state)
        t = _ticket_required(store, ticket_id)
        t["status"] = TICKET_STATUS_BLOCKED
        t["blocked_on"] = blocked_on.strip()
        t["updated_at"] = now_iso()
        _add_history(
            t,
            {
                "ts": t["updated_at"],
                "type": "blocked",
                "user": user,
                "blocked_on": blocked_on,
                "note": (note or "").strip(),
            },
        )
        save_tickets(root, state, store)
        return t


def ticket_reopen(
    root: Path,
    state: Dict[str, Any],
    *,
    ticket_id: str,
    user: str,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    with ticket_lock(root, state):
        store = load_tickets(root, state)
        t = _ticket_required(store, ticket_id)
        t["status"] = TICKET_STATUS_OPEN
        t["blocked_on"] = ""
        t["updated_at"] = now_iso()
        _add_history(
            t,
            {
                "ts": t["updated_at"],
                "type": "reopened",
                "user": user,
                "note": (note or "").strip(),
            },
        )
        save_tickets(root, state, store)
        return t


def ticket_close(
    root: Path,
    state: Dict[str, Any],
    *,
    ticket_id: str,
    user: str,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    with ticket_lock(root, state):
        store = load_tickets(root, state)
        t = _ticket_required(store, ticket_id)
        t["status"] = TICKET_STATUS_CLOSED
        t["blocked_on"] = ""
        t["updated_at"] = now_iso()
        _add_history(
            t,
            {
                "ts": t["updated_at"],
                "type": "closed",
                "user": user,
                "note": (note or "").strip(),
            },
        )
        save_tickets(root, state, store)
        return t


def configure_readline(history_path: Optional[Path] = None) -> None:
    try:
        readline.parse_and_bind("set editing-mode emacs")
        readline.parse_and_bind('"\\e[1;5D": backward-word')
        readline.parse_and_bind('"\\e[1;5C": forward-word')
        readline.parse_and_bind("set show-all-if-ambiguous on")
        readline.parse_and_bind("TAB: complete")
    except Exception:
        pass
    if not history_path:
        return
    try:
        mkdirp(history_path.parent)
        readline.read_history_file(history_path)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _readline_save_history(history_path: Optional[Path]) -> None:
    if not history_path:
        return
    try:
        mkdirp(history_path.parent)
        readline.write_history_file(history_path)
    except Exception:
        pass


def prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            raw = input(prompt + suffix).strip().lower()
        except EOFError:
            return default
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please type 'y' or 'n'.")


def prompt_line(
    prompt: str, *, default: Optional[str] = None, secret: bool = False
) -> str:
    show = f" [{default}]" if default else ""
    while True:
        try:
            if secret:
                import getpass

                raw = getpass.getpass(prompt + show + ": ")
            else:
                raw = input(prompt + show + ": ")
        except EOFError:
            return default or ""
        raw = raw.strip()
        if raw:
            return raw
        if default is not None:
            return default


def prompt_multiline(
    header: str,
    *,
    terminator: str = ".",
    history_path: Optional[Path] = None,
) -> str:
    print(header.rstrip())
    print(
        f"(paste multiple lines; finish with a line containing only '{terminator}', or press Ctrl-D)"
    )
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            print("")
            break
        if line.strip() == terminator:
            break
        lines.append(line.rstrip("\n"))
    _readline_save_history(history_path)
    return "\n".join(lines).rstrip() + "\n" if lines else ""


def ensure_executable(name: str, hint: str = "") -> None:
    if shutil.which(name) is None:
        msg = f"required executable not found in PATH: {name}"
        if hint:
            msg += f"\n{hint}"
        raise OTeamError(msg)


def run_cmd(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    capture: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=check,
            text=True,
            capture_output=capture,
            env=env,
        )
    except FileNotFoundError as e:
        raise OTeamError(f"missing executable: {cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "").strip()
        err = (e.stderr or "").strip()
        details = "\n".join(x for x in [out, err] if x)
        raise OTeamError(
            f"command failed: {shlex.join(cmd)}" + (f"\n{details}" if details else "")
        ) from e


def pick_shell() -> List[str]:
    shell = os.environ.get("SHELL", "")
    candidates: List[str] = []
    if shell:
        candidates.append(shell)
    candidates.extend(["bash", "zsh", "sh"])
    for c in candidates:
        exe = c if "/" in c else shutil.which(c)
        if exe and Path(exe).exists():
            return [exe, "-lc"]
    return ["sh", "-lc"]


def install_self_into_root(root: Path) -> None:
    try:
        src = Path(__file__).resolve()
    except Exception:
        return
    dst = root / "oteam.py"
    try:
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            return
        shutil.copy2(src, dst)
    except Exception:
        pass


def samefile(a: Path, b: Path) -> bool:
    try:
        if not a.exists() or not b.exists():
            return False
        return os.path.samefile(a, b)
    except Exception:
        return False


def safe_link_file(target: Path, link: Path) -> None:
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    try:
        os.symlink(str(target), str(link))
        return
    except OSError:
        pass
    try:
        abs_target = (
            target if target.is_absolute() else (link.parent / target).resolve()
        )
        if abs_target.exists() and abs_target.is_file():
            os.link(str(abs_target), str(link))
            return
    except OSError:
        pass
    abs_target = target if target.is_absolute() else (link.parent / target).resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(abs_target, link)


def safe_link_dir(target: Path, link: Path) -> None:
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    try:
        os.symlink(str(target), str(link))
        return
    except OSError:
        link.mkdir(parents=True, exist_ok=True)
        abs_target = (
            target if target.is_absolute() else (link.parent / target).resolve()
        )
        atomic_write_text(
            link / "README_oteam_pointer.md",
            textwrap.dedent(
                f"""\
                # oteam pointer

                This directory is a placeholder because the filesystem disallowed a symlink.

                Canonical directory:
                `{abs_target}`

                Please use the canonical path above.
                """
            ),
        )


def copytree_merge(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != ".git"]
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            d = dst / rel / f
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)


def find_project_root(start: Path) -> Optional[Path]:
    start = start.expanduser().resolve()
    if start.is_file():
        start = start.parent
    for p in [start] + list(start.parents):
        if (p / STATE_FILENAME).exists():
            return p
    return None


def sync_oteam_into_agents(root: Path, state: Dict[str, Any]) -> None:
    oteam_path = root / "oteam.py"
    if not oteam_path.exists():
        return
    for agent in state.get("agents", []):
        a_dir = root / agent["dir_rel"]
        repo_dir = root / agent["repo_dir_rel"]
        try:
            mkdirp(a_dir)
            mkdirp(repo_dir)
            safe_link_file(oteam_path, a_dir / "oteam.py")
            safe_link_file(oteam_path, repo_dir / "oteam.py")
        except Exception:
            pass


def git_init_bare(path: Path, branch: str = DEFAULT_GIT_BRANCH) -> None:
    mkdirp(path)
    try:
        run_cmd(["git", "init", "--bare", f"--initial-branch={branch}"], cwd=path)
    except OTeamError:
        run_cmd(["git", "init", "--bare"], cwd=path)
        atomic_write_text(path / "HEAD", f"ref: refs/heads/{branch}\n")


def git_clone(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    mkdirp(dst.parent)
    run_cmd(["git", "clone", str(src), str(dst)], cwd=dst.parent)


def git_clone_bare(src: str, dst: Path) -> None:
    if dst.exists():
        return
    mkdirp(dst.parent)
    run_cmd(["git", "clone", "--bare", src, str(dst)], cwd=dst.parent)


def git_config_local(repo: Path, key: str, value: str) -> None:
    run_cmd(["git", "-C", str(repo), "config", key, value], capture=True)


def git_fetch(repo: Path) -> None:
    run_cmd(["git", "-C", str(repo), "fetch", "--all", "--prune"], capture=True)


def git_status_short(repo: Path) -> str:
    try:
        cp = run_cmd(["git", "-C", str(repo), "status", "-sb"], capture=True)
        return (cp.stdout or "").strip()
    except OTeamError as e:
        return f"(git status failed: {e})"


def git_try_pull_ff(repo: Path, branch: str = DEFAULT_GIT_BRANCH) -> str:
    try:
        git_fetch(repo)
        cp = run_cmd(
            ["git", "-C", str(repo), "pull", "--ff-only", "origin", branch],
            capture=True,
        )
        out = (cp.stdout or "").strip()
        return out or f"pulled {branch}"
    except OTeamError as e:
        return f"(pull failed: {e})"


def git_list_heads(bare_repo: Path) -> List[str]:
    try:
        cp = run_cmd(
            [
                "git",
                "-C",
                str(bare_repo),
                "for-each-ref",
                "--format=%(refname:short)\t%(objectname:short)\t%(committerdate:iso8601)\t%(subject)",
                "refs/heads",
            ],
            capture=True,
        )
        return [l.strip() for l in (cp.stdout or "").splitlines() if l.strip()]
    except OTeamError:
        return []


def git_append_info_exclude(repo: Path, lines: List[str]) -> None:
    exclude = repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = read_text(exclude, "")
    add: List[str] = []
    for line in lines:
        if line.strip() and line not in existing:
            add.append(line)
    if add:
        if existing and not existing.endswith("\n"):
            existing += "\n"
        atomic_write_text(exclude, existing + "\n".join(add) + "\n")


def tmux(
    cmd: List[str], *, capture: bool = True, check: bool = True
) -> subprocess.CompletedProcess:
    return run_cmd(["tmux", *cmd], capture=capture, check=check)


def tmux_has_session(session: str) -> bool:
    try:
        tmux(["has-session", "-t", session], capture=True, check=True)
        return True
    except OTeamError:
        return False


def tmux_list_windows(session: str) -> List[str]:
    cp = tmux(["list-windows", "-t", session, "-F", "#{window_name}"], capture=True)
    return [l.strip() for l in (cp.stdout or "").splitlines() if l.strip()]


def tmux_new_session(session: str, root_dir: Path) -> None:
    tmux(
        ["new-session", "-d", "-s", session, "-n", "root", "-c", str(root_dir)],
        capture=True,
    )
    tmux(
        ["set-option", "-t", session, "remain-on-exit", "on"], capture=True, check=False
    )
    tmux(
        ["set-option", "-t", session, "allow-rename", "off"], capture=True, check=False
    )


def tmux_new_window(
    session: str, name: str, start_dir: Path, command_args: Optional[List[str]] = None
) -> None:
    cmd = ["new-window", "-t", session, "-n", name, "-c", str(start_dir)]
    if command_args:
        cmd.extend(command_args)
    tmux(cmd, capture=True)


def tmux_respawn_window(
    session: str, window: str, start_dir: Path, command_args: List[str]
) -> None:
    cmd = ["respawn-window", "-t", f"{session}:{window}", "-c", str(start_dir)]
    cmd.extend(command_args)
    try:
        tmux(cmd, capture=True)
        return
    except OTeamError:
        tmux(["kill-window", "-t", f"{session}:{window}"], capture=True, check=False)
        tmux_new_window(session, window, start_dir, command_args=command_args)


def tmux_send_keys(session: str, window: str, keys: List[str]) -> None:
    tmux(["send-keys", "-t", f"{session}:{window}", *keys], capture=True)


def tmux_select_window(session: str, window: str) -> None:
    tmux(["select-window", "-t", f"{session}:{window}"], capture=True, check=False)


def tmux_send_line(
    session: str,
    window: str,
    text: str,
) -> bool:
    payload = text
    target = f"{session}:{window}"
    tmux(["set-buffer", "--", payload], capture=True)
    tmux_send_keys(session, window, ["C-u"])
    tmux(["paste-buffer", "-d", "-t", target], capture=True)
    tmux_send_keys(session, window, ["Enter"])
    return True


def tmux_send_line_when_quiet(
    session: str,
    window: str,
    text: str,
    *,
    quiet_for: float = 2.0,
    block_timeout: float = 10.0,
) -> bool:
    payload = text
    target = f"{session}:{window}"
    deadline = time.time() + max(block_timeout, 0.5)
    while time.time() < deadline:
        if wait_for_pane_quiet(
            session,
            window,
            quiet_for=quiet_for,
            timeout=min(quiet_for + 2.0, deadline - time.time()),
        ):
            tmux(["set-buffer", "--", payload], capture=True)
            tmux_send_keys(session, window, ["C-u"])
            tmux(["paste-buffer", "-d", "-t", target], capture=True)
            tmux_send_keys(session, window, ["Enter"])
            return True
    tmux_send_line(session, window, text)
    return True


def tmux_pane_current_command(session: str, window: str) -> str:
    try:
        cp = tmux(
            [
                "list-panes",
                "-t",
                f"{session}:{window}",
                "-F",
                "#{pane_current_command}",
            ],
            capture=True,
        )
        out = (cp.stdout or "").strip().splitlines()
        return out[0].strip() if out else ""
    except Exception:
        return ""


def wait_for_pane_command(
    session: str, window: str, target_substring: str, timeout: float = 4.0
) -> bool:
    deadline = time.time() + max(timeout, 0.1)
    target = target_substring.lower()
    while time.time() < deadline:
        cmd = tmux_pane_current_command(session, window).lower()
        if target in cmd:
            return True
        time.sleep(0.1)
    return False


def _pane_sig(session: str, window: str) -> Tuple[int, str]:
    try:
        cp = tmux(
            ["capture-pane", "-p", "-t", f"{session}:{window}", "-J"], capture=True
        )
        txt = cp.stdout or ""
        return (len(txt), txt[-400:])
    except Exception:
        return (0, "")


def wait_for_pane_quiet(
    session: str, window: str, *, quiet_for: float = 1.0, timeout: float = 6.0
) -> bool:
    deadline = time.time() + max(timeout, 0.1)
    quiet_start = time.time()
    last_sig = _pane_sig(session, window)
    while time.time() < deadline:
        time.sleep(0.2)
        sig = _pane_sig(session, window)
        if sig == last_sig:
            if time.time() - quiet_start >= quiet_for:
                return True
        else:
            quiet_start = time.time()
            last_sig = sig
    return False


def tmux_attach(session: str, window: Optional[str] = None) -> None:
    if window:
        try:
            tmux(
                ["select-window", "-t", f"{session}:{window}"],
                capture=True,
                check=False,
            )
        except Exception:
            pass
    os.execvp("tmux", ["tmux", "attach-session", "-t", session])


def tmux_kill_session(session: str) -> None:
    tmux(["kill-session", "-t", session], capture=True, check=False)


@dataclass
class OpenCodeCaps:
    cmd: str
    help_text: str

    def supports(self, flag: str) -> bool:
        return flag in self.help_text

    def has_any(self, flags: List[str]) -> Optional[str]:
        for f in flags:
            if self.supports(f):
                return f
        return None


_opencode_caps_cache: Dict[str, OpenCodeCaps] = {}


def opencode_caps(opencode_cmd: str) -> OpenCodeCaps:
    if opencode_cmd in _opencode_caps_cache:
        return _opencode_caps_cache[opencode_cmd]
    help_text = ""
    try:
        cp = run_cmd([opencode_cmd, "--help"], capture=True, check=False)
        help_text = (cp.stdout or "") + "\n" + (cp.stderr or "")
    except Exception:
        help_text = ""
    caps = OpenCodeCaps(cmd=opencode_cmd, help_text=help_text)
    _opencode_caps_cache[opencode_cmd] = caps
    return caps


def build_opencode_base_args(state: Dict[str, Any]) -> List[str]:
    opencode = state["opencode"]
    cmd = DEFAULT_OPENCODE_CMD
    caps = opencode_caps(cmd)
    args: List[str] = [cmd]
    if opencode.get("yolo", False):
        flag = caps.has_any(["--dangerously-bypass-approvals-and-sandbox", "--yolo"])
        if flag:
            args.append(flag)
    elif opencode.get("full_auto", False):
        if caps.supports("--full-auto"):
            args.append("--full-auto")
        else:
            if caps.supports("--ask-for-approval"):
                args += ["--ask-for-approval", "on-failure"]
            if caps.supports("--sandbox"):
                args += ["--sandbox", "workspace-write"]
    else:
        if caps.supports("--sandbox"):
            args += ["--sandbox", opencode.get("sandbox", DEFAULT_OPENCODE_SANDBOX)]
        if caps.supports("--ask-for-approval"):
            args += [
                "--ask-for-approval",
                opencode.get("ask_for_approval", DEFAULT_OPENCODE_APPROVAL),
            ]
        elif caps.supports("--approval-policy"):
            args += [
                "--approval-policy",
                opencode.get("ask_for_approval", DEFAULT_OPENCODE_APPROVAL),
            ]
        elif caps.supports("--approval-mode"):
            ap = opencode.get("ask_for_approval", DEFAULT_OPENCODE_APPROVAL)
            mode = "full-auto" if ap == "never" else "auto"
            args += ["--approval-mode", mode]
    if opencode.get("search", DEFAULT_OPENCODE_SEARCH) and caps.supports("--search"):
        args.append("--search")
    model = opencode.get("model")
    if model and caps.supports("--model"):
        args += ["--model", model]
    return args


def build_opencode_args_for_agent(
    state: Dict[str, Any], agent: Dict[str, Any], start_dir: Path
) -> List[str]:
    base = build_opencode_base_args(state)
    cmd = DEFAULT_OPENCODE_CMD
    caps = opencode_caps(cmd)
    args = list(base)
    if caps.supports("--cd"):
        args += ["--cd", str(start_dir)]
    if caps.supports("--add-dir"):
        root = Path(state["root_abs"])
        shared = root / DIR_SHARED
        agent_dir = root / agent["dir_rel"]
        for d in [root, shared, agent_dir]:
            args += ["--add-dir", str(d)]
    return args


def default_agents(devs: int = 1) -> List[Tuple[str, str, str]]:
    agents: List[Tuple[str, str, str]] = [
        ("pm", "project_manager", "Project Manager"),
        ("architect", "architect", "Architect"),
    ]
    if devs <= 1:
        agents.append(("dev1", "developer", "Developer"))
    else:
        for i in range(1, devs + 1):
            agents.append((f"dev{i}", "developer", f"Developer {i}"))
    agents.append(("tester", "tester", "Tester / QA"))
    agents.append(("researcher", "researcher", "Researcher"))
    return agents


def role_persona(role: str) -> str:
    return {
        "project_manager": "A decisive, organized PM who coordinates others and drives the project to completion.",
        "architect": "A pragmatic architect who favors simple, testable designs and makes crisp tradeoffs.",
        "developer": "A careful, productive developer who writes maintainable code and small commits.",
        "tester": "A thorough QA engineer focused on reproducibility and automated coverage.",
        "researcher": "A concise researcher who reduces uncertainty with actionable notes.",
    }.get(role, "A helpful software collaborator.")


def build_state(
    root: Path,
    project_name: str,
    devs: int,
    *,
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
    root_abs = str(root.resolve())
    session = f"oteam_{slugify(project_name)}"
    agents_list: List[Dict[str, Any]] = []
    for name, role, title in default_agents(devs):
        dir_rel = f"{DIR_AGENTS}/{name}"
        repo_dir_rel = f"{dir_rel}/proj"
        agents_list.append(
            {
                "name": name,
                "role": role,
                "title": title,
                "persona": role_persona(role),
                "dir_rel": dir_rel,
                "repo_dir_rel": repo_dir_rel,
            }
        )
    state: Dict[str, Any] = {
        "version": STATE_VERSION,
        "created_at": now_iso(),
        "project_name": project_name,
        "mode": mode,
        "imported_from": imported_from,
        "root_abs": root_abs,
        "git": {"default_branch": DEFAULT_GIT_BRANCH},
        "opencode": {
            "cmd": opencode_cmd,
            "model": opencode_model,
            "sandbox": sandbox,
            "ask_for_approval": approval,
            "search": bool(search),
            "full_auto": bool(full_auto),
            "yolo": bool(yolo),
        },
        "coordination": {
            "autostart": autostart,
            "pm_is_boss": True,
            "start_agents_on_assignment": True,
            "assignment_type": ASSIGNMENT_TYPE,
            "assignment_from": ["pm", "oteam"],
            "stage": "discovery",
            "auto_dispatch": False,
        },
        "supervision": {
            "customer_burst_seconds": 20,
            "customer_ack_cooldown_seconds": 25,
            "agent_idle_seconds": 120,
            "agent_stalled_nudge_seconds": 10,
            "agent_unassigned_nudge_seconds": 600,
            "pm_digest_interval_seconds": 15,
            "dashboard_write_seconds": 10,
            "agent_resume_seconds": 180,
            "agent_resume_cooldown_seconds": 180,
            "agent_resume_max_prompts": 2,
            "mailbox_nudge_cooldown_seconds": 90,
            "assigned_ticket_reminder_seconds": 60,
        },
        "tmux": {
            "session": session,
            "router": bool(router),
            "paused": False,
        },
        "tickets": {
            "json_rel": TICKETS_JSON_REL,
        },
        "telegram": {
            "enabled": False,
            "config_rel": TELEGRAM_CONFIG_REL,
        },
        "agents": agents_list,
    }
    compute_agent_abspaths(state)
    return state


def compute_agent_abspaths(state: Dict[str, Any]) -> None:
    root = Path(state["root_abs"])
    state["shared_abs"] = str((root / DIR_SHARED).resolve())
    for a in state["agents"]:
        a["dir_abs"] = str((root / a["dir_rel"]).resolve())
        a["repo_dir_abs"] = str((root / a["repo_dir_rel"]).resolve())


def upgrade_state_if_needed(root: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    v = int(state.get("version") or 1)
    if v == STATE_VERSION:
        return state
    if v < 4:
        state.setdefault("created_at", now_iso())
        state.setdefault("project_name", root.name)
        state.setdefault("mode", state.get("mode", "new"))
        state.setdefault("imported_from", state.get("imported_from"))
        state.setdefault("root_abs", str(root.resolve()))
        state.setdefault("git", {}).setdefault("default_branch", DEFAULT_GIT_BRANCH)
        opencode = state.get("opencode", {})
        opencode.setdefault("cmd", DEFAULT_OPENCODE_CMD)
        opencode.setdefault("model", None)
        opencode.setdefault("sandbox", DEFAULT_OPENCODE_SANDBOX)
        opencode.setdefault("ask_for_approval", DEFAULT_OPENCODE_APPROVAL)
        opencode.setdefault("search", DEFAULT_OPENCODE_SEARCH)
        opencode.setdefault("full_auto", False)
        opencode.setdefault("yolo", False)
        state["opencode"] = opencode
        coord = state.get("coordination", {})
        coord.setdefault("autostart", DEFAULT_AUTOSTART)
        coord.setdefault("pm_is_boss", True)
        coord.setdefault("start_agents_on_assignment", True)
        coord.setdefault("assignment_type", ASSIGNMENT_TYPE)
        coord.setdefault("assignment_from", ["pm", "oteam"])
        state["coordination"] = coord
        tm = state.get("tmux", {})
        tm.setdefault(
            "session", f"oteam_{slugify(state.get('project_name', root.name))}"
        )
        tm.setdefault("router", True)
        tm.setdefault("paused", False)
        state["tmux"] = tm
        agents: List[Dict[str, Any]] = state.get("agents") or []
        for a in agents:
            a.setdefault("title", a.get("name", "agent").title())
            a.setdefault("persona", role_persona(a.get("role", "")))
            if "dir_rel" not in a:
                a["dir_rel"] = f"{DIR_AGENTS}/{a['name']}"
            if "repo_dir_rel" not in a:
                a["repo_dir_rel"] = f"{a['dir_rel']}/proj"
        state["agents"] = agents
    if v < 5:
        tg = state.get("telegram", {})
        if not isinstance(tg, dict):
            tg = {}
        tg.setdefault("enabled", False)
        tg.setdefault("config_rel", TELEGRAM_CONFIG_REL)
        state["telegram"] = tg
    if v < 6:
        coord = state.get("coordination", {})
        if not isinstance(coord, dict):
            coord = {}
        current = coord.get("assignment_from", ["pm", "oteam"])
        if "customer" in current:
            coord["assignment_from"] = [x for x in current if x != "customer"]
        coord.setdefault("assignment_type", ASSIGNMENT_TYPE)
        coord.setdefault("start_agents_on_assignment", True)
        coord.setdefault("pm_is_boss", True)
        state["coordination"] = coord
    if v < 7:
        tickets_cfg = state.get("tickets") or {}
        if not isinstance(tickets_cfg, dict):
            tickets_cfg = {}
        tickets_cfg.setdefault("json_rel", TICKETS_JSON_REL)
        state["tickets"] = tickets_cfg
        try:
            _ = load_tickets(root, state)
        except Exception:
            save_tickets(root, state, _ticket_store_default())
    else:
        tickets_cfg = state.get("tickets") or {}
        if isinstance(tickets_cfg, dict) and "md_rel" in tickets_cfg:
            tickets_cfg.pop("md_rel", None)
            state["tickets"] = tickets_cfg
    if v < 8:
        coord = state.get("coordination") or {}
        if not isinstance(coord, dict):
            coord = {}
        coord.setdefault("stage", "discovery")
        coord.setdefault("auto_dispatch", False)
        state["coordination"] = coord
        sup = state.get("supervision") or {}
        if not isinstance(sup, dict):
            sup = {}
        sup.setdefault("customer_burst_seconds", 20)
        sup.setdefault("customer_ack_cooldown_seconds", 25)
        sup.setdefault("agent_idle_seconds", 120)
        sup.setdefault("agent_stalled_nudge_seconds", 10)
        sup.setdefault("agent_unassigned_nudge_seconds", 600)
        sup.setdefault("pm_digest_interval_seconds", 15)
        sup.setdefault("dashboard_write_seconds", 10)
        sup.setdefault("agent_resume_seconds", 180)
        sup.setdefault("agent_resume_cooldown_seconds", 180)
        sup.setdefault("agent_resume_max_prompts", 2)
        sup.setdefault("mailbox_nudge_cooldown_seconds", 90)
        sup.setdefault("assigned_ticket_reminder_seconds", 60)
        state["supervision"] = sup
    if v < 9:
        sup = state.get("supervision") or {}
        if not isinstance(sup, dict):
            sup = {}
        sup.setdefault("agent_resume_seconds", 180)
        sup.setdefault("agent_resume_cooldown_seconds", 180)
        sup.setdefault("agent_resume_max_prompts", 2)
        sup.setdefault("mailbox_nudge_cooldown_seconds", 90)
        state["supervision"] = sup
    if v < 11:
        sup = state.get("supervision") or {}
        if not isinstance(sup, dict):
            sup = {}
        sup["assigned_ticket_reminder_seconds"] = float(
            sup.get("assigned_ticket_reminder_seconds", 60) or 60
        )
        state["supervision"] = sup
    state["version"] = STATE_VERSION
    compute_agent_abspaths(state)
    save_state(root, state)
    return state


def save_state(root: Path, state: Dict[str, Any]) -> None:
    atomic_write_text(
        root / STATE_FILENAME, json.dumps(state, indent=2, sort_keys=True) + "\n"
    )


def migrate_cteam_state_if_needed(root: Path) -> bool:
    cteam_path = root / "cteam.json"
    oteam_path = root / STATE_FILENAME
    if not cteam_path.exists():
        return False
    try:
        cteam_state = json.loads(cteam_path.read_text(encoding="utf-8"))
        if oteam_path.exists():
            oteam_state = json.loads(oteam_path.read_text(encoding="utf-8"))
            if oteam_state.get("version", 0) >= STATE_VERSION:
                cteam_path.rename(cteam_path.with_suffix(".json.bak"))
                log_line(
                    root, f"archived legacy cteam.json (oteam.json already current)"
                )
                return True
            oteam_state["root_abs"] = str(root.resolve())
            coord = oteam_state.get("coordination", {})
            assignment_from = coord.get("assignment_from", ["pm", "oteam"])
            if "cteam" in assignment_from:
                assignment_from = [
                    "oteam" if x == "cteam" else x for x in assignment_from
                ]
                coord["assignment_from"] = assignment_from
                oteam_state["coordination"] = coord
            save_state(root, oteam_state)
        else:
            cteam_state["root_abs"] = str(root.resolve())
            coord = cteam_state.get("coordination", {})
            assignment_from = coord.get("assignment_from", ["pm", "oteam"])
            if "cteam" in assignment_from:
                assignment_from = [
                    "oteam" if x == "cteam" else x for x in assignment_from
                ]
                coord["assignment_from"] = assignment_from
                cteam_state["coordination"] = coord
            save_state(root, cteam_state)
        cteam_path.rename(cteam_path.with_suffix(".json.bak"))
        log_line(root, f"migrated cteam.json to {STATE_FILENAME}")
        return True
    except Exception as e:
        log_line(root, f"failed to migrate cteam.json: {e}")
        return False
    try:
        state = json.loads(cteam_path.read_text(encoding="utf-8"))
        if oteam_path.exists() and not _needs_cteam_migration(oteam_path, state):
            cteam_path.rename(cteam_path.with_suffix(".json.bak"))
            return True
        if not oteam_path.exists():
            state["root_abs"] = str(root.resolve())
            coord = state.get("coordination", {})
            assignment_from = coord.get("assignment_from", ["pm", "oteam"])
            if "cteam" in assignment_from:
                assignment_from = [
                    "oteam" if x == "cteam" else x for x in assignment_from
                ]
                coord["assignment_from"] = assignment_from
                state["coordination"] = coord
            save_state(root, state)
        cteam_path.rename(cteam_path.with_suffix(".json.bak"))
        log_line(root, f"migrated cteam.json to {STATE_FILENAME}")
        return True
    except Exception as e:
        log_line(root, f"failed to migrate cteam.json: {e}")
        return False


def _needs_cteam_migration(oteam_path: Path, cteam_state: Dict[str, Any]) -> bool:
    try:
        oteam_state = json.loads(oteam_path.read_text(encoding="utf-8"))
        if oteam_state.get("version", 0) >= STATE_VERSION:
            return True
        return False
    except Exception:
        return False


def load_state(root: Path) -> Dict[str, Any]:
    p = root / STATE_FILENAME
    if not p.exists():
        migrate_cteam_state_if_needed(root)
        p = root / STATE_FILENAME
    if not p.exists():
        raise OTeamError(f"state file not found: {p}")
    state = json.loads(p.read_text(encoding="utf-8"))
    state["root_abs"] = str(root.resolve())
    state = upgrade_state_if_needed(root, state)
    compute_agent_abspaths(state)
    for a in state.get("agents", []):
        a["name"] = a.get("name", "").strip()
    return state


def render_seed_readme() -> str:
    return textwrap.dedent(
        """\
        # Seed (customer / higher-up inputs)

        Put the source-of-truth requirements and constraints here.

        Examples:
        - problem statement / scope
        - acceptance criteria
        - UX docs
        - API specs
        - constraints (tech stack, infra, compliance)

        This directory is NOT part of the git repo.
        """
    )


def render_seed_extras_readme() -> str:
    return textwrap.dedent(
        """\
        # Seed extras (research inputs)

        Supplementary materials:
        - internal references
        - competitor notes
        - standards/RFCs
        - datasets
        - links and snippets

        This directory is NOT part of the git repo.
        """
    )


def render_protocol_md() -> str:
    return textwrap.dedent(
        f"""\
        # Coordination protocol (PM-led, router-supervised)

        This workspace runs a **PM-led** process with a lightweight **router/supervisor** (the `oteam watch` loop).
        The router exists because each model is effectively single-threaded: it prevents silent idling, batches bursty
        customer IM, and keeps a live status board.

        ## Shared memory and ground truth
        - `shared/BRIEF.md` is the **living memory**. Keep it current.
        - `shared/GOALS.md` lists goals and success criteria.
        - `shared/PLAN.md` holds the current plan and milestones.
        - `shared/DECISIONS.md` records decisions and rationale.
        - `shared/TICKETS.json` is the work queue.
        - `shared/STATUS_BOARD.md` is generated by the router (do not edit).

        ## Stages
        The team works in explicit stages (stored in `oteam.json` under `coordination.stage`):
        1) **discovery**: clarify the job, constraints, and acceptance criteria; produce an initial plan.
        2) **planning**: break work into tickets with clear scope, dependencies, and test strategy.
        3) **execution**: implement tickets, review/merge, and iterate.

        The router can auto-dispatch work **only** in `execution`.

        ## Ticket protocol
        Tickets live in `shared/TICKETS.json` and have a `status`:
        - `open`: ready to be assigned or in progress
        - `blocked`: cannot proceed (must specify why)
        - `closed`: done

        **READY tickets**:
        - A ticket is considered **ready** if it is `open`, unassigned, and tagged with `ready` (or `auto`).
        - If `coordination.auto_dispatch=true` and stage is `execution`, the router may **auto-assign** ready tickets to idle agents.

        PM responsibilities for tickets:
        - Keep ticket descriptions concrete (what to change, where, how to test).
        - Mark tickets `ready` only when requirements are known enough that an engineer can execute without re-asking.
        - Keep dependencies explicit (`depends_on`), and close tickets promptly when done.
        - Actively manage load: ensure each active ticket has an assignee, ETA, and next step; reassign or split work when agents are idle/stalled.

        ## Messaging protocol
        Agents communicate via mailbox files in `shared/mail/<agent>/...`.

        - PM assigns work by sending messages with:
          - `Type: {ASSIGNMENT_TYPE}`
          - `Ticket: Txxx` (whenever applicable)
        - Non-PM agents **must not** start implementation without an assignment.

        **Read `agents/<name>/AGENTS.md` for detailed instructions on:**
        - How to read your mail
        - How to send messages (`oteam msg`)
        - How to send assignments (`oteam assign`)
        - How to manage tickets (`oteam tickets`)
        - Common workflow patterns with examples

        If you are unassigned:
        - Check `shared/STATUS_BOARD.md`, `shared/BRIEF.md`, and the ticket queue.
        - Propose 1–3 next tickets or improvements to PM (short + actionable).
        - Do not idle silently.

        ## Status protocol (prevents idling)
        Every agent maintains an `agents/<name>/STATUS.md` file:
        - Current ticket / task
        - Last action taken
        - Next step (one concrete thing)
        - Blockers (if any)

        The router nudges:
        - Assigned agents who appear stalled
        - Unassigned agents who are idle
        - PM when the whole team is idle

        **Important (OpenCode is single-shot):** models only act when prompted. The router will automatically re-prompt assigned agents who go quiet so they resume work and send status back to PM. Expect explicit updates; do not assume silent background progress.

        ## Customer IM batching + commands
        Customer messages may arrive bursty (message-per-thought). The router batches them and sends PM a digest after a short quiet window.

        Customer can use:
        - `/tickets`
        - `/ticket T001`
        - `/status`
        - `/tail <agent> [N]`
        - `/approve <TOKEN>` / `/deny <TOKEN>` (for spawning new agents)

        ## Spawning new agents
        The PM may request new agents, but the **customer must explicitly approve**.
        Use: `oteam request-agent --role developer --reason "..."`
        The customer approves in IM with `/approve <TOKEN>`.

        """
    )


def render_brief_template(project_name: str) -> str:
    return textwrap.dedent(
        f"""\
        # Project brief (living memory)

        This file is the **shared short-term memory** for the whole team.
        Keep it current. If you learn something important from the customer, capture it here.

        ## One-sentence goal
        - TODO

        ## User / customer
        - TODO (who are we building for? what do they care about?)

        ## Non-goals
        - TODO

        ## Constraints
        - TODO (languages, platforms, deadlines, performance, security, licensing, etc.)

        ## Environment
        - Repo: {project_name}
        - How to run: TODO
        - How to test: TODO

        ## Definition of done
        - TODO (what must be true for the customer to accept the work?)

        ## Open questions
        - TODO

        ## Prior answers from the customer
        - TODO (add bullets with dates / links to messages when possible)
        """
    )


def render_goals_template(project_name: str) -> str:
    return textwrap.dedent(
        f"""\
        # GOALS — {project_name}

        Owned by: **PM**

        ## What we believe this project is
        - (Infer from README/docs/code and any seed materials.)

        ## Goals
        - ...

        ## Non-goals
        - ...

        ## Users / stakeholders
        - ...

        ## Success criteria
        - ...

        ## Open questions for humans
        - ...
        """
    )


def render_plan_template() -> str:
    return textwrap.dedent(
        """\
        # PLAN

        Owned by: **PM**

        Use this page for high-level goals/milestones only. All actionable work must live in tickets (`oteam tickets ...`).

        Suggested structure:
        - Goals (link to GOALS.md)
        - Milestones (tie to ticket IDs)
        - Risks & mitigations
        - Definition of Done
        """
    )


def render_tasks_template() -> str:
    return textwrap.dedent(
        """\
        # TASKS

        Owned by: **PM**

        All tasks are tracked in tickets (`shared/TICKETS.json`). Do not list work here.

        Useful commands:
        - List tickets: `python3 oteam.py . tickets list`
        - Show ticket:  `python3 oteam.py . tickets show --id T001`
        """
    )


def render_decisions_template() -> str:
    return textwrap.dedent(
        """\
        # DECISIONS

        Owned by: **Architect**

        Record key decisions with rationale:
        - Context
        - Options
        - Decision
        - Consequences
        """
    )


def render_timeline_template() -> str:
    return textwrap.dedent(
        """\
        # TIMELINE

        Owned by: **PM**
        """
    )


def render_shared_readme(state: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""\
        # Shared workspace — OpenCode Team (oteam)

        Project: **{state["project_name"]}**
        Root: `{state["root_abs"]}`

        This directory is outside the git repo; it's for coordination artifacts.

        Key files:
        - GOALS.md, PLAN.md (high-level milestones), TICKETS.json (ticket database)
        - DECISIONS.md, TIMELINE.md
        - PROTOCOL.md

        Mail:
        - shared/mail/<agent>/message.md (append-only mailbox)

        Shared drive:
        - shared/drive/ for non-repo artifacts (design exports, videos, binaries). Do NOT park code here.

        Router:
        - tmux window `{ROUTER_WINDOW}` runs `python3 oteam.py . watch`
        """
    )


def render_team_roster(state: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Team roster (oteam)\n")
    lines.append(f"- Project: **{state['project_name']}**")
    lines.append(f"- Root: `{state['root_abs']}`")
    lines.append(f"- Created: {state['created_at']}")
    lines.append(f"- Mode: `{state.get('mode', 'new')}`")
    if state.get("imported_from"):
        lines.append(f"- Imported from: `{state['imported_from']}`")
    lines.append("")
    lines.append("## Agents\n")
    for a in state["agents"]:
        lines.append(
            f"- **{a['name']}** — {a['title']}  \n"
            f"  Role: `{a['role']}`  \n"
            f"  Workdir: `{a['dir_rel']}`  \n"
            f"  Repo: `{a['repo_dir_rel']}`\n"
        )
    return "\n".join(lines)


def render_agent_status_template(agent: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""\
        # STATUS — {agent["name"]} ({agent["title"]})

        Last updated: (edit me)

        ## Current assignment
        - (task id / description)

        ## Progress since last update
        - ...

        ## Blockers / questions
        - ...

        ## Next actions
        - ...

        ## Notes
        - ...
        """
    )


def render_agent_agents_md(
    agent_or_state: Dict[str, Any], agent: Optional[Dict[str, Any]] = None
) -> str:
    if agent is None:
        agent = agent_or_state
    name = agent["name"]

    has_seed = False
    seed_text = ""
    if isinstance(agent_or_state, dict):
        root_abs = agent_or_state.get("root_abs", "")
        if root_abs:
            seed_path = Path(root_abs) / DIR_SEED / "SEED.md"
            has_seed = seed_path.exists()
    if has_seed:
        seed_text = f"""

## CRITICAL: Do NOT ask questions - SEED.md has answers!

**SEED.md exists** in `seed/SEED.md`. It contains comprehensive instructions, requirements, and constraints for this project.

**DO NOT ask the customer questions until you have:**
1. Thoroughly read and understood `seed/SEED.md`
2. Checked `shared/PLAN.md` and `shared/GOALS.md` for context
3. Reviewed existing tickets and documentation
4. Made a genuine attempt to resolve the question yourself using available information

**If SEED.md doesn't answer your question:**
- Re-read SEED.md more carefully
- Check shared/BRIEF.md for project context
- Review active tickets for scope
- Propose an answer yourself rather than asking

Only ask the customer a question if SEED.md explicitly requires customer input or clarification on something outside the defined scope.

"""

    header = f"""# Agent: {name}
Role: {agent["role"]}
Title: {agent.get("title", "")}

## Your mission
You are part of a multi-agent developer team. The PM coordinates. You execute assigned work efficiently and keep the team unblocked.

This repo is structured for models:
- Shared memory: `shared/`
- Your workspace: `agents/{name}/`
- Messages: `shared/mail/{name}/message.md`

{seed_text}

---

# MESSAGING & TICKETING SYSTEM (READ THIS!)

**ALL communication MUST go through oteam commands. Do NOT edit shared files directly!**

---

## How to read mail

```bash
less shared/mail/{name}/message.md              # All messages
ls shared/mail/{name}/inbox/                     # New items
less shared/mail/{name}/inbox/<file>            # Specific message
```

Look for **Type** (ASSIGNMENT = work now), **Subject**, and **Ticket** (e.g., T001).

## How to send a message

```bash
# Report progress
oteam msg --to pm --subject "T001 progress" --body "Completed X. Next: Y."

# Ask a question (be specific!)
oteam msg --to pm --subject "Question about T002" --body "Issue: X. Options: A or B? Recommend A."

# Report blocker
oteam msg --to pm --subject "T003 blocked" --body "Blocked on: waiting for API."

# Status update
oteam msg --to pm --subject "Status" --body "Working on T004. ETA: next turn."
```

Key: `--to` (recipient), `--subject` (brief), `--body` (detail), `--ticket` (optional link).

## How tickets work

**NEVER edit shared/TICKETS.json directly!** Use:

```bash
oteam tickets list              # Show open tickets
oteam tickets show T001         # Show ticket details
oteam tickets create --title "..." --desc "..." --assignee dev1  # Create
oteam tickets assign T001 dev1 --note "..."  # Assign/reassign
oteam tickets block T001 --on "reason" --note "..."  # Mark blocked
oteam tickets reopen T001       # Reopen
oteam tickets close T001 --note "..."  # Close
```

### Ticket workflow

```
PM creates → Assigns to agent → Agent works → Agent reports → PM closes
```

### When YOU are assigned:
1. Acknowledge: `oteam msg --to pm --subject "T001 received" --body "Starting now..."`
2. Work on it
3. Report progress: `oteam msg --to pm --subject "T001 progress" --body "..."`
4. When done: `oteam msg --to pm --subject "T001 done" --body "Completed X, Y, Z"`

### When YOU are blocked:
1. Block: `oteam tickets block T001 --on "reason"`
2. Message PM: `oteam msg --to pm --subject "T001 blocked" --body "Options: A, B, C. Recommend A."`

### When YOU are unassigned/idle:
1. `oteam tickets list` - see open tickets
2. Propose work: `oteam msg --to pm --subject "Next steps" --body "I see T002 is ready."`
3. **Do NOT sit idle - communicate!**

## How to send assignments (PM only)

```bash
# Assign existing ticket
oteam assign --ticket T001 --to dev1 "Work description"

# Auto-create and assign
oteam assign --to dev1 --title "Fix X" --desc "When Y happens..." "Implement the fix"
```

## CRITICAL: Customer communication

**Customer messages are TOP PRIORITY - reply IMMEDIATELY!**

When you see a customer message:
1. Reply NOW: `oteam msg --to pm --subject "Customer reply" --body "Response..."`
2. PM will forward to customer
3. **Never leave customer messages unanswered**

## Work loop (when assigned)

1. **Acknowledge** the assignment
2. **Create branch**: `cd agents/{name}/proj && git checkout -b agent/{name}/T001`
3. **Analyse** the problem, **devise** a solution, **plan** the steps, **implement** the change, **verify** it works
4. **Run tests**
5. **Update STATUS.md**:
   ```markdown
   - Ticket: T001
   - Last action: Implemented feature X
   - Next step: Write tests
   - Blockers: None
   ```
6. **Report to PM** with what changed, commands ran, results

## Remember

- **Communicate through `oteam msg` and `oteam assign`**
- **Track all work through tickets**
- **Don't edit shared files directly**
- **Reply to customer messages immediately**
- **Keep PM informed of progress and blockers**

---

## Router nudges

Router nudges when:
- You are idle without a ticket
- You are stalled on an assigned ticket
- **Customer is waiting** (URGENT - reply first!)

When nudged: check message.md, do the next task, update STATUS.md.

## Useful commands

```bash
# Messaging
oteam msg --to pm --subject "..." --body "..."              # Send message
oteam assign --ticket T001 --to dev1 "work"                 # Assign ticket
oteam broadcast --subject "..." --body "..."                # Message all agents

# Tickets
oteam tickets list                                          # Show open tickets
oteam tickets show T001                                     # Show ticket
oteam tickets create --title "..." --desc "..." --assignee dev1
oteam tickets assign T001 dev1 --note "..."
oteam tickets block T001 --on "reason" --note "..."
oteam tickets reopen T001 --note "..."
oteam tickets close T001 --note "..."

# Workspace
oteam dashboard                                             # Live status
oteam capture <agent> --lines 200 --to pm                   # Send output to PM
oteam update-workdir                                        # Refresh this file
```

---

## Start here (always)
1) Read `shared/PROTOCOL.md`
2) Read `shared/BRIEF.md` (living memory)
3) Check inbox: `less shared/mail/{name}/message.md`

"""

    return header


def initial_prompt_for_pm(state: Dict[str, Any]) -> str:
    mode = state.get("mode", "new")
    root_abs = state.get("root_abs", "")
    path_hint = f"oteam.py location: {root_abs}/oteam.py. "
    if mode == "import":
        return (
            "You are the Project Manager.\n\n"
            "CRITICAL PRIORITY ORDER:\n"
            "1) REPLY TO CUSTOMER - never leave customer messages unanswered\n"
            "2) Coordinate the team - assign work, unblock agents\n"
            "3) Update documentation - BRIEF, PLAN, tickets\n\n"
            "First: open message.md for any kickoff notes.\n"
            "Then: infer what this codebase is for, analyse, devise, plan, implement, verify solutions, write shared/GOALS.md + shared/PLAN.md.\n"
            "Track all work via `oteam tickets ...`. Assign with `oteam assign --ticket ...`.\n"
            "Start now. "
            f"{path_hint}"
        )
    return (
        "You are the Project Manager.\n\n"
        "CRITICAL PRIORITY ORDER:\n"
        "1) REPLY TO CUSTOMER - never leave customer messages unanswered\n"
        "2) Coordinate the team - assign work, unblock agents\n"
        "3) Update documentation - BRIEF, PLAN, tickets\n\n"
        "First: open message.md for any kickoff notes.\n"
        "Then: read seed/ and analyse, devise, plan, implement, verify solutions, write shared/GOALS.md + shared/PLAN.md.\n"
        "Track work via `oteam tickets ...`. Assign with `oteam assign --ticket ...`.\n"
        "Start now. "
        f"{path_hint}"
    )


def prompt_on_mail(agent_name: str) -> str:
    return (
        f"MAILBOX UPDATED for {agent_name}. Check message.md with: `less shared/mail/{agent_name}/message.md`. "
        'Use `oteam msg --to pm --subject "..." --body "..."` to communicate. '
        "If it's an assignment, analyse, devise, plan, implement, verify the solution, and update STATUS.md. "
        "If unclear, ask PM a precise question via oteam msg."
    )


def standby_banner(agent: Dict[str, Any], state: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""\
        OpenCode Team (oteam) — {state["project_name"]}
        Agent window: {agent["name"]} ({agent["title"]})
        Mode: {state.get("mode", "new")}

        Standby mode (PM-led coordination):
        - Waiting for PM/human assignment.
        - Your inbox: {DIR_MAIL}/{agent["name"]}/message.md
        - Tickets: managed via `oteam tickets ...` (work only on tickets assigned to you)
        - If you are assigned work, this window will be repurposed to run OpenCode automatically.

        Tip: If you want to manually check mail:
          less -R {shlex.quote(str(Path(state["root_abs"]) / DIR_MAIL / agent["name"] / "message.md"))}
        """
    ).strip()


# -----------------------------
# Workspace creation
# -----------------------------


def ensure_shared_scaffold(root: Path, state: Dict[str, Any]) -> bool:
    mkdirp(root / DIR_LOGS)
    mkdirp(root / DIR_SEED)
    mkdirp(root / DIR_SEED_EXTRAS)
    mkdirp(root / DIR_SHARED)
    mkdirp(root / DIR_SHARED_DRIVE)
    mkdirp(root / DIR_AGENTS)
    mkdirp(root / DIR_RUNTIME)

    tickets_existed = tickets_json_path(root, state).exists()
    try:
        _ = load_tickets(root, state)
    except Exception:
        save_tickets(root, state, _ticket_store_default())
    created_tickets = not tickets_existed

    if not (root / DIR_SEED / "README.md").exists():
        atomic_write_text(root / DIR_SEED / "README.md", render_seed_readme())
    if not (root / DIR_SEED_EXTRAS / "README.md").exists():
        atomic_write_text(
            root / DIR_SEED_EXTRAS / "README.md", render_seed_extras_readme()
        )

    shared_files = {
        "README.md": render_shared_readme(state),
        "TEAM_ROSTER.md": render_team_roster(state),
        "PROTOCOL.md": render_protocol_md(),
        "BRIEF.md": render_brief_template(state["project_name"]),
        "STATUS_BOARD.md": "# Status board (generated by cteam)\n\n",
        "GOALS.md": render_goals_template(state["project_name"]),
        "PLAN.md": render_plan_template(),
        "TASKS.md": render_tasks_template(),
        "DECISIONS.md": render_decisions_template(),
        "TIMELINE.md": render_timeline_template(),
        "MESSAGES.log.md": "# Message log (append-only)\n\n",
        "ASSIGNMENTS.log.md": "# Assignments log (append-only)\n\n",
    }
    for name, content in shared_files.items():
        p = root / DIR_SHARED / name
        if not p.exists():
            atomic_write_text(p, content)

    drive_readme = root / DIR_SHARED_DRIVE / "README.md"
    if not drive_readme.exists():
        atomic_write_text(
            drive_readme,
            textwrap.dedent(
                """\
                # Shared drive (cteam)

                Use this for large or non-repo artifacts (design exports, videos, binaries).
                Code/config still belongs in git (agents/*/proj and project/).
                """
            ),
        )

    mail_root = root / DIR_MAIL
    mkdirp(mail_root)
    for agent in state["agents"]:
        base = mail_root / agent["name"]
        mkdirp(base / "inbox")
        mkdirp(base / "outbox")
        mkdirp(base / "sent")
        msg = base / "message.md"
        if not msg.exists():
            atomic_write_text(msg, f"# Inbox — {agent['name']}\n\n(append-only)\n\n")
    # Customer channel (for PM <-> customer updates)
    customer_dir = mail_root / "customer"
    mkdirp(customer_dir / "inbox")
    mkdirp(customer_dir / "outbox")
    mkdirp(customer_dir / "sent")
    if not (customer_dir / "message.md").exists():
        atomic_write_text(
            customer_dir / "message.md",
            "# Customer channel\n\nUse this file to record customer updates, questions, and answers.\n\n",
        )

    return created_tickets


def create_git_scaffold_new(root: Path, state: Dict[str, Any]) -> None:
    bare = root / DIR_PROJECT_BARE
    checkout = root / DIR_PROJECT_CHECKOUT

    if not (bare / "HEAD").exists():
        git_init_bare(bare, branch=state["git"]["default_branch"])

    if not checkout.exists():
        git_clone(bare, checkout)

        readme = checkout / "README.md"
        if not readme.exists():
            atomic_write_text(
                readme, f"# {state['project_name']}\n\n(Initialized by cteam)\n"
            )
        docs = checkout / "docs"
        mkdirp(docs)
        if not (docs / "README.md").exists():
            atomic_write_text(docs / "README.md", "# Docs\n\n")
        git_config_local(checkout, "user.name", "oteam")
        git_config_local(checkout, "user.email", "cteam@local")
        run_cmd(["git", "-C", str(checkout), "add", "-A"])
        run_cmd(
            [
                "git",
                "-C",
                str(checkout),
                "commit",
                "-m",
                "chore: initialize repository",
            ],
            check=False,
        )
        run_cmd(
            [
                "git",
                "-C",
                str(checkout),
                "push",
                "-u",
                "origin",
                state["git"]["default_branch"],
            ],
            check=False,
        )

        try:
            git_append_info_exclude(
                checkout, [f"{DIR_SHARED}/", f"{DIR_SEED}/", f"{DIR_SEED_EXTRAS}/"]
            )
        except Exception:
            pass


def create_git_scaffold_import(
    root: Path, state: Dict[str, Any], src: str
) -> Optional[str]:
    bare = root / DIR_PROJECT_BARE
    checkout = root / DIR_PROJECT_CHECKOUT

    if bare.exists() and any(bare.iterdir()):
        raise OTeamError(
            f"{DIR_PROJECT_BARE} already exists and is not empty in {root}"
        )

    tar_tmp: Optional[tempfile.TemporaryDirectory] = None
    src_seed_text: Optional[str] = None

    try:
        tar_tmp, tar_src_root, src_seed_text = _maybe_extract_tarball_src(src)
        if tar_src_root is not None:
            src_path = tar_src_root
        else:
            imported_as_git = False
            try:
                git_clone_bare(src, bare)
                imported_as_git = True
            except OTeamError:
                imported_as_git = False
                if _looks_like_git_url(src):
                    raise

            if imported_as_git:
                if not checkout.exists():
                    git_clone(bare, checkout)
                try:
                    git_append_info_exclude(
                        checkout,
                        [f"{DIR_SHARED}/", f"{DIR_SEED}/", f"{DIR_SEED_EXTRAS}/"],
                    )
                except Exception:
                    pass
                return src_seed_text

            src_path = Path(src).expanduser().resolve()
            if not src_path.exists() or not src_path.is_dir():
                raise OTeamError(
                    f"--src is neither a git repo, tarball, nor a readable directory: {src}"
                )

        git_init_bare(bare, branch=state["git"]["default_branch"])
        git_clone(bare, checkout)

        for item in src_path.iterdir():
            if item.name == ".git":
                continue
            dest = checkout / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)

        git_config_local(checkout, "user.name", "oteam-import")
        git_config_local(checkout, "user.email", "cteam+import@local")
        run_cmd(["git", "-C", str(checkout), "add", "-A"])
        try:
            run_cmd(
                ["git", "-C", str(checkout), "commit", "-m", "Import existing project"],
                capture=True,
            )
        except OTeamError:
            run_cmd(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "commit",
                    "--allow-empty",
                    "-m",
                    "Import existing project",
                ],
                capture=True,
            )

        run_cmd(["git", "-C", str(checkout), "push", "origin", "HEAD"], capture=True)

        try:
            git_append_info_exclude(
                checkout, [f"{DIR_SHARED}/", f"{DIR_SEED}/", f"{DIR_SEED_EXTRAS}/"]
            )
        except Exception:
            pass
        return src_seed_text
    finally:
        if tar_tmp:
            tar_tmp.cleanup()


def create_agent_dirs(root: Path, state: Dict[str, Any], agent: Dict[str, Any]) -> None:
    install_self_into_root(root)
    agent_dir = root / agent["dir_rel"]
    mkdirp(agent_dir)
    mkdirp(agent_dir / "notes")
    mkdirp(agent_dir / "artifacts")

    copytree_merge(root / DIR_SEED, agent_dir / "seed")
    if agent["role"] in ("researcher", "architect", "project_manager"):
        copytree_merge(root / DIR_SEED_EXTRAS, agent_dir / "seed-extras")

    if not (agent_dir / "AGENTS.md").exists():
        atomic_write_text(agent_dir / "AGENTS.md", render_agent_agents_md(state, agent))
    if not (agent_dir / "STATUS.md").exists():
        atomic_write_text(agent_dir / "STATUS.md", render_agent_status_template(agent))

    canonical_msg_rel = Path("..") / ".." / DIR_MAIL / agent["name"] / "message.md"
    canonical_inbox_rel = Path("..") / ".." / DIR_MAIL / agent["name"] / "inbox"
    canonical_outbox_rel = Path("..") / ".." / DIR_MAIL / agent["name"] / "outbox"
    safe_link_file(canonical_msg_rel, agent_dir / "message.md")
    safe_link_dir(canonical_inbox_rel, agent_dir / "inbox")
    safe_link_dir(canonical_outbox_rel, agent_dir / "outbox")

    repo_dir = root / agent["repo_dir_rel"]
    if not repo_dir.exists():
        git_clone(root / DIR_PROJECT_BARE, repo_dir)

    try:
        git_config_local(repo_dir, "user.name", f"cteam-{agent['name']}")
        git_config_local(repo_dir, "user.email", f"cteam+{agent['name']}@local")
    except Exception:
        pass

    try:
        git_append_info_exclude(
            repo_dir,
            [
                "AGENTS.md",
                "STATUS.md",
                "message.md",
                "seed/",
                "seed-extras/",
                "shared/",
                "oteam.py",
            ],
        )
    except Exception:
        pass

    safe_link_file(Path("..") / "AGENTS.md", repo_dir / "AGENTS.md")
    safe_link_file(Path("..") / "STATUS.md", repo_dir / "STATUS.md")
    safe_link_file(Path("..") / "message.md", repo_dir / "message.md")
    safe_link_dir(Path("..") / ".." / ".." / DIR_SHARED, repo_dir / "shared")
    safe_link_dir(
        Path("..") / ".." / ".." / DIR_SHARED_DRIVE, repo_dir / "shared-drive"
    )
    safe_link_dir(Path("..") / ".." / DIR_SHARED_DRIVE, agent_dir / "shared-drive")
    if (agent_dir / "seed").exists():
        safe_link_dir(Path("..") / "seed", repo_dir / "seed")
    if (agent_dir / "seed-extras").exists():
        safe_link_dir(Path("..") / "seed-extras", repo_dir / "seed-extras")

    if (root / "oteam.py").exists():
        safe_link_file(Path("..") / ".." / "oteam.py", agent_dir / "oteam.py")
        safe_link_file(Path("..") / ".." / ".." / "oteam.py", repo_dir / "oteam.py")


def ensure_agents_created(root: Path, state: Dict[str, Any]) -> None:
    for a in state["agents"]:
        create_agent_dirs(root, state, a)


def update_roster(root: Path, state: Dict[str, Any]) -> None:
    atomic_write_text(root / DIR_SHARED / "TEAM_ROSTER.md", render_team_roster(state))


def _kill_agent_window_if_present(state: Dict[str, Any], agent_name: str) -> None:
    session = (state.get("tmux") or {}).get("session")
    if not session or not tmux_has_session(session):
        return
    try:
        wins = set(tmux_list_windows(session))
        targets = [w for w in wins if w == agent_name or w.startswith(f"{agent_name}-")]
        for w in targets:
            tmux(["kill-window", "-t", f"{session}:{w}"], capture=True, check=False)
    except Exception:
        pass


def _archive_or_remove_agent_dirs(
    root: Path, agent: Dict[str, Any], *, purge: bool = False
) -> None:
    ts = ts_for_filename(now_iso())
    agent_dir = root / agent["dir_rel"]
    mail_dir = root / DIR_MAIL / agent["name"]

    def _move_or_remove(src: Path, dest_root: Path) -> None:
        if not src.exists():
            return
        if purge:
            shutil.rmtree(src, ignore_errors=True)
            return
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / f"{ts}_{src.name}"
        try:
            shutil.move(str(src), str(dest))
        except Exception:
            shutil.rmtree(src, ignore_errors=True)

    _move_or_remove(agent_dir, root / DIR_AGENTS / "_removed")
    _move_or_remove(mail_dir, root / DIR_MAIL / "_removed")


def _unassign_tickets_for_agent(
    root: Path, state: Dict[str, Any], agent_name: str
) -> None:
    with ticket_lock(root, state):
        store = load_tickets(root, state)
        changed = False
        for t in store["tickets"]:
            if t.get("assignee") == agent_name:
                t["assignee"] = None
                t["updated_at"] = now_iso()
                _add_history(
                    t,
                    {
                        "ts": t["updated_at"],
                        "type": "unassigned",
                        "user": "oteam",
                        "from": agent_name,
                        "to": None,
                        "note": "agent removed",
                    },
                )
                changed = True
        if changed:
            save_tickets(root, state, store)


# -----------------------------
# Coordination logic: autostart selection
# -----------------------------


def autostart_agent_names(state: Dict[str, Any]) -> List[str]:
    mode = state.get("coordination", {}).get("autostart", DEFAULT_AUTOSTART)
    names = [a["name"] for a in state["agents"]]
    if mode == "all":
        return names
    if mode == "pm+architect":
        return [n for n in names if n in ("pm", "architect")]
    return [n for n in names if n == "pm"]


# -----------------------------
# Messaging
# -----------------------------


def mailbox_paths(root: Path, agent_name: str) -> Tuple[Path, Path, Path]:
    base = root / DIR_MAIL / agent_name
    return (base / "message.md", base / "inbox", base / "outbox")


def format_message(
    ts: str,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    *,
    msg_type: str = "MESSAGE",
    task: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> str:
    subject_line = subject.strip() if subject.strip() else "(no subject)"
    lines = [
        f"## {ts} — From: {sender} → To: {recipient}",
        f"**Type:** {msg_type}",
        f"**Subject:** {subject_line}",
    ]
    if ticket_id:
        lines.append(f"**Ticket:** {ticket_id}")
    if task:
        lines.append(f"**Task:** {task}")
    lines.append("")  # blank line before body
    lines.append(body.rstrip())
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


MESSAGE_SEPARATOR_RE = re.compile(r"\n\s*---\s*\n")


def read_last_entry(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    parts = MESSAGE_SEPARATOR_RE.split(text)
    for part in reversed(parts):
        if part.strip():
            return part.strip() + "\n"
    return ""


def _split_reasons(reason_text: str) -> Set[str]:
    parts = [p.strip() for p in (reason_text or "").split("/") if p.strip()]
    if not parts:
        parts = [reason_text or "NUDGE"]
    return set(parts)


def _normalize_reasons(reasons: Set[str]) -> Set[str]:
    """
    Normalize a set of reasons so variants of the same nudge (e.g., mailbox updates with/without metadata)
    compare equal for deduplication.
    """
    if not reasons:
        return {"NUDGE"}
    normalized: Set[str] = set()
    for r in reasons:
        r_clean = (r or "").strip()
        if not r_clean:
            continue
        upper_r = r_clean.upper()
        if "MAILBOX UPDATED" in upper_r or "MAILBOX MAY BE UPDATED" in upper_r:
            normalized.add("MAILBOX UPDATED")
            continue
        normalized.add(r_clean)
    return normalized or {"NUDGE"}


def _mailbox_nudge_reason(ticket_id: Optional[str], msg_type: Optional[str]) -> str:
    reason = "ACT NOW, DO YOUR JOB. mailbox may be updated:"
    if ticket_id:
        reason = f"{ticket_id} {reason}"
    if msg_type:
        reason = f"{reason} ({msg_type})"
    return reason


def write_message(
    root: Path,
    state: Dict[str, Any],
    *,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    msg_type: str = "MESSAGE",
    task: Optional[str] = None,
    nudge: bool = True,
    start_if_needed: bool = False,
    ticket_id: Optional[str] = None,
    skip_inbox: bool = False,
) -> None:
    ts = now_iso()
    if ticket_id:
        ticket_id = ticket_id.strip()
    entry = format_message(
        ts,
        sender,
        recipient,
        subject,
        body,
        msg_type=msg_type,
        task=task,
        ticket_id=ticket_id or None,
    )
    if not entry.endswith("\n"):
        entry += "\n"

    agent_names = {a["name"] for a in state["agents"]}
    allowed = set(agent_names) | {"customer"}
    allowed_senders = allowed | {"oteam"}
    if recipient not in allowed:
        raise OTeamError(f"unknown recipient: {recipient}")
    if sender not in allowed_senders:
        raise OTeamError(f"unknown sender: {sender}")
    if sender == "customer" and recipient != "pm":
        raise OTeamError("customer can only message pm")
    if recipient == "customer" and sender not in {"pm", "oteam"}:
        raise OTeamError("only pm can message customer")
    if msg_type == ASSIGNMENT_TYPE and sender not in {"pm", "oteam"}:
        raise OTeamError("assignments must come from pm")
    if ticket_id:
        with ticket_lock(root, state):
            store = load_tickets(root, state)
            t = _find_ticket(store, ticket_id)
            if not t:
                raise OTeamError(f"unknown ticket: {ticket_id}")
            snippet = (body or "").strip().splitlines()[0] if body else ""
            _add_history(
                t,
                {
                    "ts": ts,
                    "type": "message",
                    "user": sender,
                    "note": f"{subject}".strip(),
                    "snippet": snippet[:200],
                },
            )
            t["updated_at"] = ts
            save_tickets(root, state, store)

    msg_path, inbox_dir, _ = mailbox_paths(root, recipient)
    mkdirp(inbox_dir)

    if not skip_inbox:
        atomic_write_text(inbox_dir / f"{ts_for_filename(ts)}_{sender}.md", entry)

    append_text(msg_path, entry)

    append_text(root / DIR_SHARED / "MESSAGES.log.md", entry)

    if msg_type == ASSIGNMENT_TYPE:
        append_text(root / DIR_SHARED / "ASSIGNMENTS.log.md", entry)

    if sender == "pm" and recipient == "customer":
        cust_state = load_customer_state(root)
        cust_state["last_pm_reply_ts"] = ts
        cust_state["pending_messages"] = []
        save_customer_state(root, cust_state)

    if sender == "customer" and recipient == "pm":
        cust_state = load_customer_state(root)
        cust_state["last_customer_msg_ts"] = ts
        msg_id = ts_for_filename(ts)
        pending = cust_state.get("pending_messages") or []
        existing_ids = [p.get("id") for p in pending]
        if msg_id not in existing_ids:
            pending.append({"id": msg_id, "subject": subject, "added_at": ts})
            cust_state["pending_messages"] = pending

        # Burst buffering: customers often send message-per-thought. Buffer entries so the router can
        # send one digest after a short quiet window, instead of spamming the PM.
        try:
            buf_path = (root / CUSTOMER_BURST_BUFFER_REL).resolve()
            mkdirp(buf_path.parent)
            append_text(buf_path, entry)
            cust_state["burst_pending"] = True
            cust_state["burst_last_ts"] = ts
            cust_state["burst_count"] = int(cust_state.get("burst_count", 0) or 0) + 1
            if not cust_state.get("burst_start_ts"):
                cust_state["burst_start_ts"] = ts
        except Exception:
            pass

        # Throttled automated acknowledgement to the customer (useful for IM).
        if (state.get("telegram") or {}).get("enabled", False):
            try:
                sup = state.get("supervision") or {}
                cooldown = int(sup.get("customer_ack_cooldown_seconds", 25) or 25)
                last_ack = iso_to_unix(str(cust_state.get("last_customer_ack_ts", "")))
                last_pm_reply = iso_to_unix(str(cust_state.get("last_pm_reply_ts", "")))
                # Only send acknowledgement if enough time since last AND PM hasn't replied yet
                if not last_ack or (time.time() - last_ack) >= cooldown:
                    if not (last_pm_reply and last_pm_reply >= last_ack):
                        ack_body = (
                            "Automated acknowledgement: received. "
                            "You can keep sending details; rapid messages will be batched. "
                            "The PM will reply here when available."
                        )
                        write_message(
                            root,
                            state,
                            sender="oteam",
                            recipient="customer",
                            subject="Received — PM notified",
                            body=ack_body,
                            msg_type="MESSAGE",
                            task=None,
                            nudge=False,
                            start_if_needed=False,
                            skip_inbox=True,
                        )
                        cust_state["last_customer_ack_ts"] = ts
            except Exception:
                pass

        save_customer_state(root, cust_state)

        # For imported workspaces, treat this as the signal that the customer
        # is ready for the team to start working autonomously.
        try:
            _maybe_trigger_import_kickoff_on_customer_message(root, state)
        except Exception:
            pass

    # Save a copy in sender outbox (if sender is known) for local context.
    if sender in (allowed):
        _, _, outbox_dir = mailbox_paths(root, sender)
        mkdirp(outbox_dir)
        atomic_write_text(outbox_dir / f"{ts_for_filename(ts)}_{recipient}.md", entry)

    if recipient != "customer":
        rec = next(a for a in state["agents"] if a["name"] == recipient)
        agent_dir_msg = root / rec["dir_rel"] / "message.md"
        repo_msg = root / rec["repo_dir_rel"] / "message.md"
        if agent_dir_msg.exists() and not samefile(agent_dir_msg, msg_path):
            append_text(agent_dir_msg, entry)
        if repo_msg.exists() and not samefile(repo_msg, msg_path):
            append_text(repo_msg, entry)

    # Mirror to sender inbox for conversational context (if it exists and is distinct).
    if sender and sender != recipient and sender not in {"customer", "oteam"}:
        try:
            sender_msg_path, _, _ = mailbox_paths(root, sender)
            if sender_msg_path.exists() and not samefile(sender_msg_path, msg_path):
                append_text(sender_msg_path, entry)
        except Exception:
            pass

    if not nudge:
        return

    if recipient != "customer":
        if start_if_needed:
            maybe_start_agent_on_message(
                root, state, recipient, sender=sender, msg_type=msg_type
            )
        nudge_reason = _mailbox_nudge_reason(ticket_id, msg_type)
        nudge_agent(root, state, recipient, reason=nudge_reason)


def write_broadcast_message(
    root: Path,
    state: Dict[str, Any],
    *,
    sender: str,
    recipients: List[str],
    subject: str,
    body: str,
    msg_type: str = "MESSAGE",
    task: Optional[str] = None,
    nudge: bool = True,
    start_if_needed: bool = False,
    ticket_id: Optional[str] = None,
) -> None:
    ts = now_iso()
    if ticket_id:
        ticket_id = ticket_id.strip()
    entry = format_message(
        ts,
        sender,
        ", ".join(recipients),
        subject,
        body,
        msg_type=msg_type,
        task=task,
        ticket_id=ticket_id or None,
    )
    if not entry.endswith("\n"):
        entry += "\n"

    agent_names = {a["name"] for a in state["agents"]}
    allowed = set(agent_names) | {"customer"}
    allowed_senders = allowed | {"oteam"}
    for r in recipients:
        if r not in allowed:
            raise OTeamError(f"unknown recipient: {r}")
    if sender not in allowed_senders:
        raise OTeamError(f"unknown sender: {sender}")
    if sender == "customer":
        raise OTeamError("customer cannot broadcast")
    for r in recipients:
        if r == "customer" and sender not in {"pm", "oteam"}:
            raise OTeamError("only pm can broadcast to customer")

    if msg_type == ASSIGNMENT_TYPE and sender not in {"pm", "oteam"}:
        raise OTeamError("assignments must come from pm")

    if ticket_id:
        with ticket_lock(root, state):
            store = load_tickets(root, state)
            t = _find_ticket(store, ticket_id)
            if not t:
                raise OTeamError(f"unknown ticket: {ticket_id}")
            snippet = (body or "").strip().splitlines()[0] if body else ""
            _add_history(
                t,
                {
                    "ts": ts,
                    "type": "message",
                    "user": sender,
                    "note": f"{subject}".strip(),
                    "snippet": snippet[:200],
                },
            )
            t["updated_at"] = ts
            save_tickets(root, state, store)

    for r in recipients:
        msg_path, inbox_dir, _ = mailbox_paths(root, r)
        mkdirp(inbox_dir)
        atomic_write_text(inbox_dir / f"{ts_for_filename(ts)}_{sender}.md", entry)

    main_msg_path = mailbox_paths(root, recipients[0])[0]
    append_text(main_msg_path, entry)

    broadcast_log_entry = (
        f"BROADCAST from {sender} to {', '.join(recipients)}:\n{entry}\n"
    )
    append_text(root / DIR_SHARED / "MESSAGES.log.md", broadcast_log_entry)

    if msg_type == ASSIGNMENT_TYPE:
        append_text(root / DIR_SHARED / "ASSIGNMENTS.log.md", broadcast_log_entry)

    if not nudge:
        return

    for r in recipients:
        if r != "customer":
            if start_if_needed:
                maybe_start_agent_on_message(
                    root, state, r, sender=sender, msg_type=msg_type
                )

    nudge_agents_parallel(
        root, state, recipients, reason=_mailbox_nudge_reason(ticket_id, msg_type)
    )


def nudge_agents_parallel(
    root: Path,
    state: Dict[str, Any],
    agent_names: List[str],
    *,
    reason: str = "MAILBOX UPDATED",
    interrupt: bool = False,
) -> Dict[str, bool]:
    session = state["tmux"]["session"]
    if not tmux_has_session(session):
        return {n: False for n in agent_names}

    results: Dict[str, bool] = {}
    lock = threading.Lock()
    active_threads: List[threading.Thread] = []

    def nudge_one(agent_name: str) -> None:
        _router_noise_until[agent_name] = time.time() + 2.0
        try:
            if agent_name not in set(tmux_list_windows(session)):
                ensure_agent_windows(root, state, launch_opencode=True)
        except Exception:
            pass

        extra = ""
        if agent_name == "pm":
            if "CUSTOMER" in reason.upper():
                extra = "\n\nIMMEDIATE ACTION: Reply to the customer FIRST."
            else:
                extra = "\n\nPM PRIORITIES:\n1) Reply to customer if waiting\n2) Check agent status updates\n3) Triage tickets"
        if "CUSTOMER" in reason.upper():
            extra += "\n\n**CUSTOMER IS WAITING - REPLY NOW**"

        if interrupt:
            msg = f"INTERRUPT — {reason}\n\nAction required: check message.md and reply as needed."
        else:
            if agent_name == "pm":
                msg = f"{reason}\n\nCheck message.md now."
            else:
                msg = f"{reason}\n\nCheck message.md now."

        try:
            if interrupt:
                try:
                    tmux_send_keys(session, agent_name, ["Escape"])
                    time.sleep(0.05)
                except Exception:
                    pass
            tmux_send_keys(session, agent_name, ["C-u"])
            ok = False
            if is_opencode_running(state, agent_name):
                wait_success = wait_for_pane_quiet(
                    session, agent_name, quiet_for=0.5, timeout=10.0
                )
                if not wait_success:
                    with lock:
                        results[agent_name] = False
                    return
                ok = tmux_send_line(session, agent_name, msg)
            else:
                ok = tmux_send_line_when_quiet(
                    session, agent_name, f"echo {shlex.quote(msg)}", quiet_for=0.5
                )
            if ok:
                _nudge_history[agent_name] = (
                    time.time(),
                    _normalize_reasons(_split_reasons(reason)),
                )
            with lock:
                results[agent_name] = ok
        except Exception:
            with lock:
                results[agent_name] = False

    for name in agent_names:
        if name != "customer":
            t = threading.Thread(target=nudge_one, args=(name,))
            t.start()
            active_threads.append(t)

    for t in active_threads:
        t.join()

    for name in agent_names:
        if name not in results:
            results[name] = False

    return results


# -----------------------------
# Starting / nudging agents in tmux
# -----------------------------


def ensure_tmux_session(root: Path, state: Dict[str, Any]) -> None:
    ensure_executable("tmux", hint="Install tmux (apt/brew).")
    session = state["tmux"]["session"]
    if not tmux_has_session(session):
        tmux_new_session(session, root)
        # Spawn router immediately as first window for visibility
        try:
            install_self_into_root(root)
            cmd_args = router_window_command(root)
            tmux_new_window(session, ROUTER_WINDOW, root, command_args=cmd_args)
        except Exception:
            pass
    tmux(
        ["set-option", "-t", session, "remain-on-exit", "on"], capture=True, check=False
    )
    tmux(
        ["set-option", "-t", session, "allow-rename", "off"], capture=True, check=False
    )


def router_window_command(root: Path) -> List[str]:
    shell = pick_shell()
    cmd = f"cd {shlex.quote(str(root))} && exec python3 oteam.py . watch"
    return shell + [cmd]


def customer_window_command(root: Path) -> List[str]:
    shell = pick_shell()
    cmd = f"cd {shlex.quote(str(root))} && exec python3 oteam.py . customer-chat"
    return shell + [cmd]


def ensure_router_window(root: Path, state: Dict[str, Any]) -> None:
    session = state["tmux"]["session"]
    if not state.get("tmux", {}).get("router", True):
        return
    windows = set(tmux_list_windows(session))
    if ROUTER_WINDOW not in windows:
        install_self_into_root(root)
        cmd_args = router_window_command(root)
        tmux_new_window(session, ROUTER_WINDOW, root, command_args=cmd_args)
        return

    cmd = tmux_pane_current_command(session, ROUTER_WINDOW)
    if cmd not in ("python", "python3"):
        cmd_args = router_window_command(root)
        tmux_respawn_window(session, ROUTER_WINDOW, root, command_args=cmd_args)


def ensure_customer_window(root: Path, state: Dict[str, Any]) -> None:
    session = state["tmux"]["session"]
    windows = set(tmux_list_windows(session))
    if CUSTOMER_WINDOW in windows:
        return
    install_self_into_root(root)
    cmd_args = customer_window_command(root)
    tmux_new_window(session, CUSTOMER_WINDOW, root, command_args=cmd_args)


def standby_window_command(
    root: Path, state: Dict[str, Any], agent: Dict[str, Any]
) -> List[str]:
    shell = pick_shell()
    banner = standby_banner(agent, state).replace("'", "'\"'\"'")
    sh = os.environ.get("SHELL", "bash")
    cmd = f"cd {shlex.quote(str(Path(agent['dir_abs'])))} && printf '%s\n' '{banner}' && exec {shlex.quote(sh)}"
    return shell + [cmd]


def opencode_window_command(
    root: Path, state: Dict[str, Any], agent: Dict[str, Any], *, boot: bool
) -> List[str]:
    shell = pick_shell()
    repo_dir = Path(agent["repo_dir_abs"])
    base_args = build_opencode_args_for_agent(state, agent, start_dir=repo_dir)

    opencode_bin = shutil.which("opencode") or "opencode"
    exports = (
        f"export CTEAM_ROOT={shlex.quote(str(root))} "
        f"&& export CTEAM_AGENT={shlex.quote(agent['name'])} "
        f"&& export PATH={shlex.quote(str(Path(opencode_bin).parent))}:$PATH"
    )
    cmd = (
        exports
        + " && cd "
        + shlex.quote(str(repo_dir))
        + " && "
        + " ".join(shlex.quote(x) for x in base_args)
    )
    return shell + [cmd]


def is_opencode_running(state: Dict[str, Any], window: str) -> bool:
    session = state["tmux"]["session"]
    cmd = tmux_pane_current_command(session, window).lower()
    return "opencode" in cmd


def start_opencode_in_window(
    root: Path, state: Dict[str, Any], agent: Dict[str, Any], *, boot: bool
) -> None:
    def _cmd_candidates() -> Set[str]:
        return {"opencode"}

    def _wait_for_opencode(session: str, window: str, timeout: float = 8.0) -> str:
        deadline = time.time() + max(timeout, 0.1)
        cands = _cmd_candidates()
        last_cmd = ""
        while time.time() < deadline:
            last_cmd = tmux_pane_current_command(session, window).lower()
            if any(c in last_cmd for c in cands):
                return last_cmd
            time.sleep(0.1)
        return last_cmd

    session = state["tmux"]["session"]
    w = agent["name"]
    repo_dir = Path(agent["repo_dir_abs"])
    if not repo_dir.exists():
        create_agent_dirs(root, state, agent)
        repo_dir = Path(agent["repo_dir_abs"])
    cmd_args = opencode_window_command(root, state, agent, boot=boot)

    windows = set(tmux_list_windows(session))
    if w not in windows:
        tmux_new_window(session, w, repo_dir, command_args=cmd_args)
        time.sleep(1.0)
        windows_after_create = set(tmux_list_windows(session))
        if w not in windows_after_create:
            raise OTeamError(f"failed to create tmux window: {w}")
    elif not is_opencode_running(state, w):
        tmux_respawn_window(session, w, repo_dir, command_args=cmd_args)
    else:
        return

    if boot:
        if agent["name"] == "pm":
            first_prompt = initial_prompt_for_pm(state)
        else:
            if state.get("mode") == "import":
                first_prompt = (
                    f"You are {agent['title']} ({agent['name']}). Imported project; wait for ticketed assignments. "
                    f"Read AGENTS.md first - it explains how to communicate with PM using `oteam msg`. "
                    f"When assigned, analyse, devise, plan, implement, verify the solution. "
                    f"Do NOT change code without a ticket (Type: {ASSIGNMENT_TYPE}). "
                    f'To ask PM questions or propose work: `oteam msg --to pm --subject "..." --body "..."`. '
                    f"Tickets are managed via `oteam tickets ...`. oteam.py: {state.get('root_abs', '')}/oteam.py"
                )
            else:
                first_prompt = (
                    f"You are {agent['title']} ({agent['name']}). "
                    f"Read AGENTS.md first - it explains how to communicate using `oteam msg`. "
                    f"When assigned, analyse, devise, plan, implement, verify the solution. "
                    f"If you do not have an assignment (Type: {ASSIGNMENT_TYPE}), do NOT start coding. "
                    f'To communicate with PM: `oteam msg --to pm --subject "..." --body "..."`. '
                    f"oteam.py: {state.get('root_abs', '')}/oteam.py"
                )
    else:
        first_prompt = prompt_on_mail(agent["name"])

    print(f"DEBUG: start_opencode_in_window for {w}, boot={boot}", file=sys.stderr)

    pane_cmd = _wait_for_opencode(session, w, timeout=8.0)
    wait_for_pane_quiet(session, w, quiet_for=5.0, timeout=8.0)
    if w not in set(tmux_list_windows(session)):
        raise OTeamError(f"tmux window {w} no longer exists before send-keys")

    if not pane_cmd or not any(c in pane_cmd for c in _cmd_candidates()):
        raise OTeamError(
            f"opencode is not running in window {w}: pane command is '{pane_cmd}'"
        )
    windows_before = set(tmux_list_windows(session))
    if w not in windows_before:
        time.sleep(0.5)
        windows_before = set(tmux_list_windows(session))
        if w not in windows_before:
            tmux_new_window(session, w, repo_dir, command_args=cmd_args)
    try:
        print(f"DEBUG: Sending C-u to {w}", file=sys.stderr)
        tmux_send_keys(session, w, ["C-u"])
        time.sleep(0.5)
        print(f"DEBUG: C-u sent, calling tmux_send_line_when_quiet", file=sys.stderr)
    except OTeamError as first_err:
        print(f"DEBUG: C-u failed: {first_err}", file=sys.stderr)
        time.sleep(0.5)
        windows_after = set(tmux_list_windows(session))
        if w not in windows_after:
            tmux_new_window(session, w, repo_dir, command_args=cmd_args)
        try:
            tmux_send_keys(session, w, ["C-u"])
            time.sleep(0.5)
        except OTeamError as second_err:
            raise OTeamError(
                f"failed to send C-u to {w}: {first_err}; recovery also failed: {second_err}"
            ) from first_err
    print(f"DEBUG: Calling tmux_send_line_when_quiet for {w}", file=sys.stderr)
    ok = tmux_send_line_when_quiet(session, w, first_prompt)
    print(f"DEBUG: tmux_send_line_when_quiet returned {ok}", file=sys.stderr)
    if not ok:
        raise OTeamError(f"failed to send startup prompt to {w}")


def nudge_agent(
    root: Path,
    state: Dict[str, Any],
    agent_name: str,
    *,
    reason: str = "MAILBOX UPDATED",
    interrupt: bool = False,
) -> bool:
    session = state["tmux"]["session"]
    if not tmux_has_session(session):
        return False
    _router_noise_until[agent_name] = time.time() + 5.0
    try:
        if agent_name not in set(tmux_list_windows(session)):
            ensure_agent_windows(root, state, launch_opencode=True)
    except Exception:
        pass

    extra = ""
    if agent_name == "pm":
        if "CUSTOMER" in reason.upper():
            extra = "\n\nIMMEDIATE ACTION: Reply to the customer FIRST. Use `oteam msg --to customer` or `oteam customer-chat`."
        else:
            extra = (
                "\n\nPM PRIORITIES (in order):"
                "\n1) Reply to customer if waiting"
                "\n2) Check agent status updates"
                "\n3) Triage tickets"
                "\n4) Unblock agents"
            )
    if "CUSTOMER" in reason.upper():
        extra += "\n\n**CUSTOMER IS WAITING - REPLY NOW**"
    if interrupt:
        if agent_name == "pm":
            msg = f"INTERRUPT — {reason}\n\nAction required: check message.md and reply to customer/agents as needed."
        else:
            msg = f"INTERRUPT — {reason}\n\nAction: check message.md, analyse the problem, devise a solution, plan the steps, implement the change, verify it works, reply to PM if needed, update STATUS.md."
    else:
        if agent_name == "pm":
            msg = f"{reason}\n\nCheck message.md now. Possibly reply to waiting customer, then handle other pending PM work."
        else:
            msg = f"{reason}\n\nCheck message.md now. Analyse, devise, plan, implement, verify the solution, send PM status, update STATUS.md."
    try:
        if interrupt:
            try:
                tmux_send_keys(session, agent_name, ["Escape"])
                time.sleep(0.1)
            except Exception:
                pass
        tmux_send_keys(session, agent_name, ["C-u"])
        ok = False
        if is_opencode_running(state, agent_name):
            wait_success = wait_for_pane_quiet(
                session, agent_name, quiet_for=5.0, timeout=30.0
            )
            if not wait_success:
                return False
            ok = tmux_send_line(session, agent_name, msg)
        else:
            ok = tmux_send_line_when_quiet(
                session, agent_name, f"echo {shlex.quote(msg)}", quiet_for=5.0
            )
        if ok:
            _nudge_history[agent_name] = (
                time.time(),
                _normalize_reasons(_split_reasons(reason)),
            )
        return ok
    except Exception:
        return False


class NudgeQueue:
    """Batch and deduplicate nudges before sending to tmux windows; defers until pane idle."""

    def __init__(
        self, root: Path, *, min_interval: float = 2.0, idle_for: float = 5.0
    ) -> None:
        self.root = root
        self.min_interval = max(0.0, float(min_interval))
        self.idle_for = max(0.0, float(idle_for))
        self.pending: Dict[str, Dict[str, Any]] = {}
        self.last_sent: Dict[str, Tuple[float, Set[str]]] = {}
        self._pane_state: Dict[str, Tuple[Tuple[int, str], float]] = {}

    def request(self, agent_name: str, reason: str, *, interrupt: bool = False) -> None:
        entry = self.pending.setdefault(agent_name, {"reasons": [], "interrupt": False})
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)
        if interrupt:
            entry["interrupt"] = True

    def _is_idle(self, state: Dict[str, Any], agent_name: str) -> bool:
        session = (state.get("tmux") or {}).get("session")
        if not session or not tmux_has_session(session):
            return True
        sig = _pane_sig(session, agent_name)
        now = time.time()
        prev_sig, prev_ts = self._pane_state.get(agent_name, (None, now))  # type: ignore
        if sig == prev_sig and (now - prev_ts) >= self.idle_for:
            return True
        self._pane_state[agent_name] = (sig, now)
        return False

    def flush(self, state: Dict[str, Any]) -> List[Tuple[str, bool, str, str]]:
        results: List[Tuple[str, bool, str, str]] = []
        now = time.time()
        busy_agents: List[str] = []

        # First pass: check which agents are idle
        for agent_name in list(self.pending.keys()):
            if not self._is_idle(state, agent_name):
                busy_agents.append(agent_name)

        # Second pass: process only idle agents
        for agent_name in list(self.pending.keys()):
            if agent_name in busy_agents:
                # Agent is busy - update backoff state and skip
                backoff_count, backoff_until = _nudge_backoff.get(agent_name, (0, 0.0))
                if now < backoff_until:
                    # Still in backoff period, skip entirely
                    entry = self.pending.get(agent_name, {})
                    reasons = entry.get("reasons") or []
                    reason_text = " / ".join(reasons) if reasons else "NUDGE"
                    results.append((agent_name, False, reason_text, "busy"))
                    continue
                # Increment backoff: exponential increase (30s, 60s, 120s, etc.)
                new_backoff_count = backoff_count + 1
                backoff_seconds = min(30 * (2 ** (new_backoff_count - 1)), 300)
                _nudge_backoff[agent_name] = (new_backoff_count, now + backoff_seconds)
                entry = self.pending.get(agent_name, {})
                reasons = entry.get("reasons") or []
                reason_text = " / ".join(reasons) if reasons else "NUDGE"
                results.append((agent_name, False, reason_text, "busy"))
                continue

            # Reset backoff when agent becomes idle
            if agent_name in _nudge_backoff:
                del _nudge_backoff[agent_name]

            entry = self.pending.pop(agent_name, {})
            reasons = entry.get("reasons") or []
            interrupt = bool(entry.get("interrupt", False))
            reason_text = " / ".join(reasons) if reasons else "NUDGE"

            last = self.last_sent.get(agent_name) or _nudge_history.get(agent_name)
            last_ts = None
            last_reasons: Set[str] = set()
            if last:
                try:
                    last_ts = float(last[0])
                    last_reasons = _normalize_reasons(set(last[1]))
                except Exception:
                    last_ts, last_reasons = None, set()

            current_reasons_raw = (
                set(reasons) if reasons else _split_reasons(reason_text)
            )
            current_reasons = _normalize_reasons(current_reasons_raw)
            if last_ts is not None and (now - last_ts) < self.min_interval:
                if current_reasons.issubset(last_reasons):
                    results.append((agent_name, False, reason_text, "duplicate"))
                    continue

            ok = nudge_agent(
                self.root, state, agent_name, reason=reason_text, interrupt=interrupt
            )
            if ok:
                reason_set = current_reasons or {"NUDGE"}
                self.last_sent[agent_name] = (now, reason_set)
                _nudge_history[agent_name] = (now, reason_set)
            results.append((agent_name, ok, reason_text, ""))
        return results


def maybe_start_agent_on_message(
    root: Path,
    state: Dict[str, Any],
    recipient: str,
    *,
    sender: str,
    msg_type: str,
) -> None:
    coord = state.get("coordination", {})
    if not coord.get("start_agents_on_assignment", True):
        return
    if recipient == "pm":
        return

    allowed_from = set(coord.get("assignment_from", ["pm", "oteam"]))
    if sender not in allowed_from:
        return

    if msg_type != coord.get("assignment_type", ASSIGNMENT_TYPE):
        return

    agent = next((a for a in state["agents"] if a["name"] == recipient), None)
    if not agent:
        return

    ensure_tmux_session(root, state)
    if not is_opencode_running(state, recipient):
        start_opencode_in_window(root, state, agent, boot=False)


# -----------------------------

# -----------------------------
# Telegram integration (customer channel bridge)
# -----------------------------


def telegram_config_path(root: Path, state: Optional[Dict[str, Any]] = None) -> Path:
    rel = None
    if state:
        rel = (state.get("telegram") or {}).get("config_rel")
    rel = rel or TELEGRAM_CONFIG_REL
    return (root / rel).resolve()


def _chmod_private(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _normalize_phone(p: str) -> str:
    return "".join(ch for ch in (p or "") if ch.isdigit())


def telegram_load_config(root: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    cfg_path = telegram_config_path(root, state)
    if not cfg_path.exists():
        raise OTeamError(
            f"telegram not configured: missing {cfg_path}. Run: python3 oteam.py {shlex.quote(str(root))} telegram-configure"
        )
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise OTeamError(f"invalid telegram config JSON: {cfg_path} ({e})") from e

    token = (cfg.get("bot_token") or "").strip()
    phone = (cfg.get("authorized_phone") or "").strip()
    if not token or not phone:
        raise OTeamError(
            f"telegram config incomplete (need bot_token + authorized_phone): {cfg_path}"
        )
    cfg["authorized_phone_norm"] = _normalize_phone(phone)
    return cfg


def telegram_save_config(
    root: Path, state: Dict[str, Any], cfg: Dict[str, Any]
) -> None:
    cfg_path = telegram_config_path(root, state)
    cfg2 = dict(cfg)
    cfg2.pop("authorized_phone_norm", None)
    atomic_write_text(cfg_path, json.dumps(cfg2, indent=2, sort_keys=True) + "\n")
    _chmod_private(cfg_path)


def _telegram_api(
    token: str, method: str, payload: Dict[str, Any], *, timeout: float = 30.0
) -> Any:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise OTeamError(
            f"telegram HTTP error calling {method}: {e.code} {e.reason}\n{detail}".strip()
        ) from e
    except Exception as e:
        raise OTeamError(f"telegram error calling {method}: {e}") from e

    try:
        obj = json.loads(raw)
    except Exception as e:
        raise OTeamError(
            f"telegram non-JSON response from {method}: {raw[:400]}"
        ) from e
    if not obj.get("ok"):
        raise OTeamError(f"telegram API returned ok=false for {method}: {raw[:600]}")
    return obj.get("result")


def _telegram_send_text(cfg: Dict[str, Any], text: str) -> None:
    chat_id = cfg.get("chat_id")
    token = cfg.get("bot_token")
    if not chat_id or not token:
        return
    t = (text or "").rstrip()
    if not t:
        t = "(empty message)"
    chunks: List[str] = []
    while len(t) > TELEGRAM_MAX_MESSAGE:
        chunks.append(t[:TELEGRAM_MAX_MESSAGE])
        t = t[TELEGRAM_MAX_MESSAGE:]
    if t:
        chunks.append(t)
    for c in chunks:
        _telegram_api(
            token, "sendMessage", {"chat_id": chat_id, "text": c}, timeout=20.0
        )


def _telegram_initial_offset(cfg: Dict[str, Any]) -> int:
    try:
        offset = int(cfg.get("update_offset") or 0)
    except Exception:
        offset = 0
    if offset < 0:
        offset = 0
    # If chat is not yet authorized, start from 0 so the /start + contact flow works.
    if not cfg.get("chat_id") or not cfg.get("user_id"):
        return 0
    return offset


def _telegram_request_contact(cfg: Dict[str, Any], chat_id: int) -> None:
    token = cfg.get("bot_token")
    if not token:
        return
    markup = {
        "keyboard": [[{"text": "Share phone number", "request_contact": True}]],
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }
    _telegram_api(
        token,
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": "To link this bot to your oteam customer channel, please share your phone number.",
            "reply_markup": markup,
        },
        timeout=20.0,
    )


def _parse_entry(
    entry: str,
) -> Tuple[
    Optional[str], Optional[str], Optional[str], str, str, Optional[str], Optional[str]
]:
    ts = None
    sender = None
    recipient = None
    subject = ""
    ticket_id: Optional[str] = None
    msg_type: Optional[str] = None
    lines = entry.splitlines()
    if lines:
        m = re.match(
            r"^##\s+(.+?)\s+—\s+From:\s*([^\s→]+)\s+→\s+To:\s*([^\s]+)\s*$",
            lines[0].strip(),
        )
        if m:
            ts = m.group(1).strip()
            sender = m.group(2).strip()
            recipient = m.group(3).strip()

    m2 = re.search(r"\*\*Subject:\*\*\s*(.+)", entry)
    if m2:
        subject = m2.group(1).strip()
    m_type = re.search(r"\*\*Type:\*\*\s*([A-Z_]+)", entry)
    if m_type:
        msg_type = m_type.group(1).strip()
    m3 = re.search(r"\*\*Ticket:\*\*\s*([A-Za-z0-9_-]+)", entry)
    if m3:
        ticket_id = m3.group(1).strip()

    body_start = 0
    for i, line in enumerate(lines[:25]):
        if line.strip().startswith("**Subject:**"):
            body_start = i + 1
            break
    while body_start < len(lines) and lines[body_start].strip().startswith("**Task:**"):
        body_start += 1
    while body_start < len(lines) and not lines[body_start].strip():
        body_start += 1

    body_lines: List[str] = []
    for line in lines[body_start:]:
        if line.strip() == "---":
            break
        body_lines.append(line)
    body = "\n".join(body_lines).rstrip()
    return ts, sender, recipient, subject, body, ticket_id, msg_type


def _sanitize_filename(name: str, fallback: str = "file") -> str:
    base = Path(name or "").name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    if not base or base in {".", ".."}:
        base = fallback
    if len(base) > 120:
        stem, ext = os.path.splitext(base)
        base = (stem[:100] or fallback) + ext[:15]
    return base or fallback


def _is_image_document(doc: Dict[str, Any]) -> bool:
    mime = str(doc.get("mime_type") or "").lower()
    if mime.startswith("image/"):
        return True
    name = str(doc.get("file_name") or "").lower()
    return any(
        name.endswith(ext)
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif")
    )


class TelegramBridge:
    """Bridges Telegram <-> oteam customer channel."""

    def __init__(self, root: Path, state: Dict[str, Any]) -> None:
        self.root = root
        self.state = state
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self._customer_pos: int = 0
        self._customer_buf: str = ""
        self._logged_offset_reset = False
        self._telegram_drive = self.root / DIR_SHARED_DRIVE / "telegram"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def run(self) -> None:
        try:
            cfg = telegram_load_config(self.root, self.state)
        except Exception as e:
            log_line(self.root, f"[telegram] disabled: {e}")
            return

        cfg_path = telegram_config_path(self.root, self.state)
        customer_file = self.root / DIR_MAIL / "customer" / "message.md"
        try:
            self._customer_pos = customer_file.stat().st_size
        except Exception:
            self._customer_pos = 0

        offset = _telegram_initial_offset(cfg)
        if offset == 0 and (not cfg.get("chat_id") or not cfg.get("user_id")):
            if not self._logged_offset_reset:
                log_line(
                    self.root,
                    "[telegram] reset update_offset to 0 (chat not authorized yet; send /start and share contact)",
                )
                self._logged_offset_reset = True
        last_saved = time.time()
        log_line(self.root, f"[telegram] bridge started (config={cfg_path})")

        while not self.stop_event.is_set():
            self._maybe_forward_outbound(cfg, customer_file)

            try:
                updates = _telegram_api(
                    cfg["bot_token"],
                    "getUpdates",
                    {"timeout": 6, "offset": offset, "allowed_updates": ["message"]},
                    timeout=12.0,
                )
                if not isinstance(updates, list):
                    updates = []
            except Exception as e:
                log_line(self.root, f"[telegram] getUpdates error: {e}")
                time.sleep(1.5)
                continue

            for upd in updates:
                try:
                    offset = max(offset, int(upd.get("update_id", 0)) + 1)
                    msg = upd.get("message") or {}
                    if msg:
                        self._handle_inbound(cfg, msg)
                except Exception as e:
                    log_line(self.root, f"[telegram] inbound handling error: {e}")

            if time.time() - last_saved >= 3.0:
                try:
                    cfg["update_offset"] = offset
                    telegram_save_config(self.root, self.state, cfg)
                    last_saved = time.time()
                except Exception:
                    pass

        try:
            cfg["update_offset"] = offset
            telegram_save_config(self.root, self.state, cfg)
        except Exception:
            pass
        log_line(self.root, "[telegram] bridge stopped")

    def _is_authorized_sender(self, cfg: Dict[str, Any], msg: Dict[str, Any]) -> bool:
        chat = msg.get("chat") or {}
        frm = msg.get("from") or {}
        if cfg.get("chat_id") and cfg.get("user_id"):
            return int(chat.get("id", 0)) == int(cfg["chat_id"]) and int(
                frm.get("id", 0)
            ) == int(cfg["user_id"])
        return False

    def _rel_path(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except Exception:
            return str(p)

    def _save_telegram_file(
        self,
        cfg: Dict[str, Any],
        file_id: Optional[str],
        *,
        prefer_name: Optional[str] = None,
    ) -> Optional[Path]:
        token = cfg.get("bot_token")
        if not token or not file_id:
            return None
        try:
            info = (
                _telegram_api(token, "getFile", {"file_id": file_id}, timeout=20.0)
                or {}
            )
            file_path = info.get("file_path")
            if not file_path:
                raise OTeamError("missing file_path")
            fname = _sanitize_filename(prefer_name or Path(file_path).name or "file")
            mkdirp(self._telegram_drive)
            ts_prefix = ts_for_filename(now_iso())
            dest = self._telegram_drive / f"{ts_prefix}_{fname}"
            counter = 1
            while dest.exists():
                dest = self._telegram_drive / f"{ts_prefix}_{counter}_{fname}"
                counter += 1
            url = f"https://api.telegram.org/file/bot{token}/{file_path}"
            with urllib.request.urlopen(url, timeout=60.0) as resp:
                data = resp.read()
            dest.write_bytes(data)
            _chmod_private(dest)
            return dest
        except Exception as e:
            log_line(self.root, f"[telegram] failed to save file {file_id}: {e}")
            return None

    def _handle_inbound(self, cfg: Dict[str, Any], msg: Dict[str, Any]) -> None:
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        text = (msg.get("text") or "").strip()
        caption = (msg.get("caption") or "").strip()
        contact = msg.get("contact")
        photos = msg.get("photo") or []
        document = msg.get("document") or {}

        # First-time setup: require contact share that matches configured authorized phone number.
        if contact and chat_id:
            phone_norm = _normalize_phone(str(contact.get("phone_number") or ""))
            if phone_norm and phone_norm == cfg.get("authorized_phone_norm"):
                frm = msg.get("from") or {}
                cfg["chat_id"] = int(chat_id)
                cfg["user_id"] = int(frm.get("id", 0) or 0)
                telegram_save_config(self.root, self.state, cfg)
                _telegram_send_text(
                    cfg,
                    "✅ Linked! This chat is now authorized for the oteam customer channel.",
                )
                log_line(
                    self.root,
                    f"[telegram] authorized chat_id={cfg['chat_id']} user_id={cfg['user_id']}",
                )
            else:
                tmp = dict(cfg)
                tmp["chat_id"] = int(chat_id)
                _telegram_send_text(
                    tmp, "⛔ Not authorized (phone number did not match)."
                )
            return

        if text == "/start" and chat_id:
            _telegram_request_contact(cfg, int(chat_id))
            return

        if not self._is_authorized_sender(cfg, msg):
            return

        if caption and not text:
            text = caption

        attachments: List[str] = []
        if photos:
            best = max(photos, key=lambda p: p.get("file_size", 0))
            saved = self._save_telegram_file(
                cfg,
                best.get("file_id"),
                prefer_name=f"photo_{best.get('file_unique_id') or 'image'}.jpg",
            )
            if saved:
                attachments.append(f"Image saved to {self._rel_path(saved)}")
            else:
                attachments.append("Image received but failed to save.")

        doc_is_image = False
        if document:
            doc_is_image = _is_image_document(document)
            if doc_is_image:
                saved = self._save_telegram_file(
                    cfg,
                    document.get("file_id"),
                    prefer_name=document.get("file_name") or "image",
                )
                if saved:
                    attachments.append(f"Image saved to {self._rel_path(saved)}")
                else:
                    name = document.get("file_name") or "image"
                    attachments.append(f"Image {name} received but failed to save.")

        body_lines: List[str] = []
        if text:
            body_lines.append(text)
        else:
            if photos or doc_is_image:
                body_lines.append("[sent an image]")
            elif msg.get("sticker"):
                body_lines.append("[sent a sticker]")
            elif document:
                body_lines.append("[sent a document]")
            else:
                body_lines.append("[sent a non-text message]")

        if attachments:
            body_lines.append("")
            body_lines.extend(attachments)

        body = "\n".join(body_lines).strip()

        try:
            state = load_state(self.root)
        except Exception:
            state = self.state

        # Record for PM mailbox
        write_message(
            self.root,
            state,
            sender="customer",
            recipient="pm",
            subject="Telegram",
            body=body,
            msg_type="MESSAGE",
            task=None,
            nudge=True,
            start_if_needed=True,
        )

        # Mirror into customer channel for visibility/logging (customer chat tails this file).
        try:
            ts = now_iso()
            entry = format_message(
                ts, "customer", "customer", "Telegram", body, msg_type="MESSAGE"
            )
            cust_msg, cust_inbox, _ = mailbox_paths(self.root, "customer")
            mkdirp(cust_inbox)
            append_text(cust_msg, entry)
            atomic_write_text(cust_inbox / f"{ts_for_filename(ts)}_customer.md", entry)
        except Exception as e:
            log_line(
                self.root,
                f"[telegram] failed to mirror inbound to customer mailbox: {e}",
            )

    def _maybe_forward_outbound(self, cfg: Dict[str, Any], customer_file: Path) -> None:
        if not cfg.get("chat_id"):
            return
        try:
            if not customer_file.exists():
                return
            with customer_file.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self._customer_pos)
                new_txt = f.read()
                self._customer_pos = f.tell()
            if not new_txt:
                return
        except Exception:
            return

        self._customer_buf += new_txt
        parts = MESSAGE_SEPARATOR_RE.split(self._customer_buf)
        complete = parts[:-1]
        self._customer_buf = parts[-1]
        for entry in complete:
            entry = entry.strip("\n")
            if not entry:
                continue
            entry = entry + "\n---\n"
            ts, sender, recipient, subject, body, _, _ = _parse_entry(entry)
            if recipient != "customer":
                if recipient is None:
                    log_line(
                        self.root,
                        f"[telegram] debug: failed to parse recipient from entry, skipping: {entry[:100]!r}",
                    )
                continue
            if sender == "customer":
                continue  # avoid echoing inbound Telegram messages back to Telegram
            if body and "\\n" in body:
                body = body.replace("\\n", "\n")
            if subject and "\\n" in subject:
                subject = subject.replace("\\n", "\n")
            out = f"[{ts or now_iso()}] {sender or 'cteam'}\n"
            if subject:
                out += f"Subject: {subject}\n"
            out += body or "(no body)"
            try:
                log_line(
                    self.root, f"[telegram] forwarding to customer: {subject[:50]!r}"
                )
                _telegram_send_text(cfg, out)
            except Exception as e:
                log_line(self.root, f"[telegram] send error: {e}")


# Router / watch loop
# -----------------------------


def parse_sender_and_type_from_message(
    text: str,
) -> Tuple[Optional[str], Optional[str]]:
    sender = None
    m = re.search(r"From:\s*([^\s→]+)", text)
    if m:
        sender = m.group(1).strip()
    m2 = re.search(r"\*\*Type:\*\*\s*([A-Z_]+)", text)
    msg_type = m2.group(1).strip() if m2 else None
    return sender, msg_type


# -----------------------------
# Router helpers (dashboard, spawn approvals, IM commands)


@contextmanager
def file_lock(lock_path: Path):
    """Exclusive lock using a lock file."""
    mkdirp(lock_path.parent)
    f = lock_path.open("a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        f.close()


def dashboard_json_path(root: Path) -> Path:
    return (root / DASHBOARD_JSON_REL).resolve()


def status_board_path(root: Path) -> Path:
    return (root / STATUS_BOARD_MD_REL).resolve()


def customer_burst_buffer_path(root: Path) -> Path:
    return (root / CUSTOMER_BURST_BUFFER_REL).resolve()


def spawn_requests_path(root: Path) -> Path:
    return (root / SPAWN_REQUESTS_REL).resolve()


def _spawn_store_default() -> Dict[str, Any]:
    return {"meta": {"next": 1}, "requests": []}


@contextmanager
def spawn_lock(root: Path):
    lock_path = spawn_requests_path(root).with_suffix(".lock")
    with file_lock(lock_path):
        yield


def load_spawn_requests(root: Path) -> Dict[str, Any]:
    p = spawn_requests_path(root)
    if not p.exists():
        return _spawn_store_default()
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("meta", {"next": 1})
    data.setdefault("requests", [])
    return data


def save_spawn_requests(root: Path, store: Dict[str, Any]) -> None:
    p = spawn_requests_path(root)
    mkdirp(p.parent)
    atomic_write_text(p, json.dumps(store, indent=2, sort_keys=True) + "\n")


def _new_approval_token(n: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(max(6, int(n))))


def create_spawn_request(
    root: Path,
    *,
    requested_by: str,
    role: str,
    name: Optional[str],
    title: Optional[str],
    persona: Optional[str],
    start_opencode: bool,
    reason: str,
) -> Dict[str, Any]:
    ts = now_iso()
    with spawn_lock(root):
        store = load_spawn_requests(root)
        nxt = int((store.get("meta") or {}).get("next", 1) or 1)
        rid = f"S{nxt:03d}"
        (store.setdefault("meta", {}))["next"] = nxt + 1
        req = {
            "id": rid,
            "token": _new_approval_token(),
            "created_at": ts,
            "requested_by": requested_by,
            "status": "pending",  # pending|approved|denied|fulfilled
            "role": role,
            "name": name,
            "title": title,
            "persona": persona,
            "start_opencode": bool(start_opencode),
            "reason": reason,
        }
        store.setdefault("requests", []).append(req)
        save_spawn_requests(root, store)
        return req


def find_spawn_request_by_token(
    store: Dict[str, Any], token: str
) -> Optional[Dict[str, Any]]:
    token = (token or "").strip().upper()
    for r in store.get("requests") or []:
        if str(r.get("token", "")).strip().upper() == token:
            return r
    return None


def tmux_capture_tail(session: str, window: str, *, lines: int = 200) -> str:
    try:
        n = max(1, int(lines))
    except Exception:
        n = 200
    try:
        cp = tmux(
            ["capture-pane", "-p", "-t", f"{session}:{window}", "-J", "-S", str(-n)],
            capture=True,
            check=False,
        )
        return cp.stdout or ""
    except Exception:
        return ""


def _human_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    try:
        s = int(max(0, seconds))
    except Exception:
        return "-"
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _ticket_priority_value(t: Dict[str, Any]) -> int:
    p = str(t.get("priority") or "").strip().upper()
    if p.startswith("P") and p[1:].isdigit():
        return int(p[1:])
    return 9


def ticket_is_ready(store: Dict[str, Any], t: Dict[str, Any]) -> bool:
    if (t.get("status") or TICKET_STATUS_OPEN) != TICKET_STATUS_OPEN:
        return False
    if t.get("assignee"):
        return False
    tags = set((t.get("tags") or []) if isinstance(t.get("tags"), list) else [])
    ready_flag = bool(t.get("ready", False)) or ("ready" in tags) or ("auto" in tags)
    if not ready_flag:
        return False
    deps = t.get("depends_on") or t.get("dependencies") or []
    if isinstance(deps, str):
        deps = [deps]
    if isinstance(deps, list):
        for dep in deps:
            dt = _find_ticket(store, str(dep))
            if dt and (dt.get("status") or TICKET_STATUS_OPEN) != TICKET_STATUS_CLOSED:
                return False
    return True


def compute_dashboard_snapshot(
    root: Path,
    state: Dict[str, Any],
    *,
    session: str,
    last_pane_change: Dict[str, float],
    last_mail_activity: Dict[str, float],
) -> Dict[str, Any]:
    now_ts = time.time()
    stage = (state.get("coordination") or {}).get("stage", "discovery")
    auto_dispatch = bool((state.get("coordination") or {}).get("auto_dispatch", False))
    windows = set(tmux_list_windows(session))

    try:
        store = load_tickets(root, state)
    except Exception:
        store = _ticket_store_default()
    tickets = store.get("tickets") or []
    active = [
        t
        for t in tickets
        if t.get("status") in {TICKET_STATUS_OPEN, TICKET_STATUS_BLOCKED}
    ]
    unassigned = [t for t in active if not t.get("assignee")]
    blocked = [t for t in active if t.get("status") == TICKET_STATUS_BLOCKED]
    open_tickets = [t for t in active if t.get("status") == TICKET_STATUS_OPEN]

    by_assignee: Dict[str, List[str]] = {}
    for tt in active:
        a = tt.get("assignee") or ""
        if not a:
            continue
        by_assignee.setdefault(a, []).append(tt.get("id", ""))

    agents_out: List[Dict[str, Any]] = []

    opencode_basename = "opencode"

    for a in state["agents"]:
        name = a["name"]
        window_exists = name in windows
        pane_cmd = tmux_pane_current_command(session, name) if window_exists else ""
        opencode_running = bool(
            window_exists and pane_cmd in {opencode_basename, "opencode"}
        )
        idle_for = None
        if window_exists:
            idle_for = now_ts - float(last_pane_change.get(name, now_ts))
        status_path = root / a["dir_rel"] / "STATUS.md"
        try:
            status_age = now_ts - status_path.stat().st_mtime
        except Exception:
            status_age = None
        mail_age = None
        try:
            mail_age = now_ts - float(last_mail_activity.get(name, now_ts))
        except Exception:
            mail_age = None
        agents_out.append(
            {
                "name": name,
                "role": a.get("role", ""),
                "window": bool(window_exists),
                "pane_cmd": pane_cmd,
                "opencode": bool(opencode_running),
                "idle_for_s": idle_for,
                "status_age_s": status_age,
                "mail_age_s": mail_age,
                "tickets": sorted([x for x in by_assignee.get(name, []) if x]),
            }
        )

    dash = {
        "ts": now_iso(),
        "stage": stage,
        "auto_dispatch": auto_dispatch,
        "tickets": {
            "open": len(open_tickets),
            "blocked": len(blocked),
            "active": len(active),
            "unassigned": len(unassigned),
            "unassigned_list": [
                {
                    "id": tt.get("id", ""),
                    "title": tt.get("title", ""),
                    "priority": tt.get("priority"),
                }
                for tt in sorted(
                    unassigned,
                    key=lambda x: (_ticket_priority_value(x), x.get("id", "")),
                )[:12]
            ],
        },
        "agents": agents_out,
    }
    return dash


def format_status_board_md(dash: Dict[str, Any]) -> str:
    ts = dash.get("ts", "")
    stage = dash.get("stage", "")
    auto_dispatch = dash.get("auto_dispatch", False)
    t = dash.get("tickets") or {}
    agents = dash.get("agents") or []
    lines: List[str] = []
    lines.append(f"# Status board — {ts}")
    lines.append("")
    lines.append(
        f"Stage: `{stage}` | Auto-dispatch: `{str(bool(auto_dispatch)).lower()}`"
    )
    lines.append("")
    lines.append("## Tickets")
    lines.append(
        f"Active: {t.get('active', 0)}  | Open: {t.get('open', 0)}  | Blocked: {t.get('blocked', 0)}  | Unassigned: {t.get('unassigned', 0)}"
    )
    ul = t.get("unassigned_list") or []
    if ul:
        lines.append("")
        lines.append("Top unassigned:")
        for it in ul:
            tid = it.get("id", "")
            title = it.get("title", "")
            pr = it.get("priority") or ""
            pr_txt = f" {pr}" if pr else ""
            lines.append(f"- {tid}{pr_txt} — {title}")
    lines.append("")
    lines.append("## Agents")
    lines.append(
        "| agent | role | tmux | opencode | idle | tickets | status_age | mail_age | pane |"
    )
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---|")
    for a in agents:
        name = a.get("name", "")
        role = a.get("role", "")
        window = "yes" if a.get("window") else "no"
        opencode = "yes" if a.get("opencode") else "no"
        idle = _human_duration(a.get("idle_for_s"))
        tickets = ",".join(a.get("tickets") or []) or "-"
        s_age = _human_duration(a.get("status_age_s"))
        m_age = _human_duration(a.get("mail_age_s"))
        pane = (a.get("pane_cmd") or "-")[:40]
        lines.append(
            f"| {name} | {role} | {window} | {opencode} | {idle} | {tickets} | {s_age} | {m_age} | {pane} |"
        )
    lines.append("")
    lines.append("_Generated by router. Do not edit by hand._")
    return "\n".join(lines) + "\n"


def format_dashboard_for_customer(dash: Dict[str, Any]) -> str:
    stage = dash.get("stage", "")
    t = dash.get("tickets") or {}
    agents = dash.get("agents") or []
    lines: List[str] = []
    lines.append(f"Stage: {stage}")
    lines.append(
        f"Tickets: active={t.get('active', 0)} open={t.get('open', 0)} blocked={t.get('blocked', 0)} unassigned={t.get('unassigned', 0)}"
    )
    ul = t.get("unassigned_list") or []
    if ul:
        lines.append("Unassigned (top):")
        for it in ul[:8]:
            lines.append(f"- {it.get('id', '?')} — {it.get('title', '')}")
    lines.append("Agents:")
    for a in agents:
        idle = a.get("idle_for_s")
        status = "active" if idle is not None and idle < 30 else "idle"
        ticket_txt = ",".join(a.get("tickets") or []) or "(no ticket)"
        lines.append(f"- {a.get('name')} ({a.get('role')}): {status} — {ticket_txt}")
    return "\n".join(lines)


def _split_slash_commands(text: str) -> List[Tuple[str, List[str]]]:
    cmds: List[Tuple[str, List[str]]] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s.startswith("/"):
            continue
        parts = s[1:].split()
        if not parts:
            continue
        cmds.append((parts[0].lower(), parts[1:]))
    return cmds


def _format_ticket_detail(store: Dict[str, Any], ticket_id: str) -> str:
    t = _find_ticket(store, ticket_id)
    if not t:
        return f"Ticket not found: {ticket_id}"
    summary = _ticket_summary_block(t)
    desc = (t.get("description") or "").rstrip()
    hist = t.get("history") or []
    tail = hist[-5:] if isinstance(hist, list) else []
    hist_lines = []
    for e in tail:
        try:
            hist_lines.append(
                f"- {e.get('ts', '')}: {e.get('type', '')} {e.get('by', '')} {e.get('note', '') or e.get('snippet', '')}"
            )
        except Exception:
            pass
    extra = ("\n\nRecent history:\n" + "\n".join(hist_lines)) if hist_lines else ""
    return f"{summary}\n\nDescription:\n{desc if desc else '(none)'}{extra}"


def cmd_watch(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    if not (root / STATE_FILENAME).exists():
        raise OTeamError(
            "watch: could not find oteam.json in this directory or its parents"
        )
    state = load_state(root)

    router_lock_path = root / DIR_RUNTIME / "router.lock"
    mkdirp(router_lock_path.parent)
    router_pid: Optional[int] = None
    try:
        existing = router_lock_path.read_text(encoding="utf-8").strip()
        if existing:
            router_pid = int(existing)
    except Exception:
        pass
    if router_pid:
        try:
            os.kill(router_pid, 0)
            raise OTeamError(
                f"router already running (PID {router_pid}). "
                f"Stop it first with `oteam <workdir> kill` or `kill {router_pid}`."
            )
        except (OSError, ProcessLookupError):
            pass
    router_lock_path.write_text(str(os.getpid()))

    agent_names = {a["name"] for a in state["agents"]}
    ensure_shared_scaffold(root, state)
    ensure_tmux_session(root, state)

    interval = max(0.5, float(args.interval))

    telegram_bridge: Optional[TelegramBridge] = None
    telegram_enabled = False

    seen_inbox: Dict[str, set[str]] = {}
    last_mail_sig: Dict[str, Tuple[int, int]] = {}
    nudge_queue = NudgeQueue(root)

    # Activity tracking: last time the tmux pane content changed.
    pane_sig: Dict[str, Tuple[int, str]] = {}
    last_pane_change: Dict[str, float] = {}
    last_mail_activity: Dict[str, float] = {}
    last_agent_reply: Dict[str, float] = {}
    last_resume_prompt: Dict[str, float] = {}
    resume_prompt_count: Dict[str, int] = {}
    last_status_mtime: Dict[str, float] = {}
    last_resume_escalation: Dict[str, float] = {}
    last_seen_mail_ts: Dict[str, str] = {}
    last_seen_mail_sig: Dict[str, str] = {}
    last_mailbox_nudge: Dict[str, float] = {}
    last_assigned_reminder: Dict[str, float] = {}

    # Debounce / rate limits
    last_dashboard_write = 0.0
    last_pm_digest = 0.0
    last_agent_stall_nudge: Dict[str, float] = {}
    last_agent_unassigned_nudge: Dict[str, float] = {}

    now_ts = time.time()
    for a in state["agents"]:
        msg_path, inbox_dir, _ = mailbox_paths(root, a["name"])
        try:
            current = set(p.name for p in inbox_dir.glob("*.md"))
        except FileNotFoundError:
            current = set()
        seen_inbox[a["name"]] = current

        try:
            st = msg_path.stat()
            last_mail_sig[a["name"]] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            last_mail_sig[a["name"]] = (0, 0)

        last_pane_change[a["name"]] = now_ts
        last_mail_activity[a["name"]] = now_ts
        last_agent_reply[a["name"]] = now_ts
        last_resume_prompt[a["name"]] = 0.0
        resume_prompt_count[a["name"]] = 0
        last_resume_escalation[a["name"]] = 0.0
        try:
            last_status_mtime[a["name"]] = (
                (root / a["dir_rel"] / "STATUS.md").stat().st_mtime
            )
        except FileNotFoundError:
            last_status_mtime[a["name"]] = now_ts
        try:
            entry = read_last_entry(msg_path)
            ts0, *_ = _parse_entry(entry)
            if ts0:
                last_seen_mail_ts[a["name"]] = ts0
            if entry and entry.strip():
                last_seen_mail_sig[a["name"]] = hashlib.sha256(
                    entry.strip().encode("utf-8", "ignore")
                ).hexdigest()
        except Exception:
            pass
        last_mailbox_nudge[a["name"]] = 0.0
        last_assigned_reminder[a["name"]] = 0.0

    log_line(root, f"[router] watching mailboxes under {root} (interval={interval}s)")

    def maybe_write_dashboard(session: str) -> Optional[Dict[str, Any]]:
        nonlocal last_dashboard_write
        sup = state.get("supervision") or {}
        every = float(sup.get("dashboard_write_seconds", 10) or 10)
        if every <= 0:
            return None
        if time.time() - last_dashboard_write < every:
            return None
        last_dashboard_write = time.time()
        try:
            dash = compute_dashboard_snapshot(
                root,
                state,
                session=session,
                last_pane_change=last_pane_change,
                last_mail_activity=last_mail_activity,
            )
            atomic_write_text(
                dashboard_json_path(root),
                json.dumps(dash, indent=2, sort_keys=True) + "\n",
            )
            atomic_write_text(status_board_path(root), format_status_board_md(dash))
            return dash
        except Exception as e:
            log_line(root, f"[router] failed to write dashboard: {e}")
            return None

    def maybe_send_pm_digest(dash: Dict[str, Any]) -> None:
        nonlocal last_pm_digest
        sup = state.get("supervision") or {}
        every = float(sup.get("pm_digest_interval_seconds", 600) or 600)
        if every <= 0:
            return
        now = time.time()
        if now - last_pm_digest < every:
            return

        idle_thresh = float(sup.get("agent_idle_seconds", 120) or 120)
        agents = dash.get("agents") or []
        idle_agents = [
            a
            for a in agents
            if a.get("name") != "pm"
            and a.get("window")
            and (a.get("idle_for_s") or 0) >= idle_thresh
        ]
        unassigned = int((dash.get("tickets") or {}).get("unassigned", 0) or 0)
        blocked = int((dash.get("tickets") or {}).get("blocked", 0) or 0)
        if not idle_agents and unassigned == 0 and blocked == 0:
            return

        last_pm_digest = now
        lines: List[str] = []
        lines.append("Supervisor digest (router)")
        lines.append(f"Stage: {dash.get('stage', '')}")
        lines.append(
            f"Tickets: unassigned={unassigned} blocked={blocked} active={int((dash.get('tickets') or {}).get('active', 0) or 0)}"
        )
        if idle_agents:
            lines.append("Idle agents:")
            for a in idle_agents[:10]:
                lines.append(
                    f"- {a.get('name')} ({a.get('role')}): idle {_human_duration(a.get('idle_for_s'))} — tickets: {','.join(a.get('tickets') or []) or '(none)'}"
                )
        lines.append("")
        lines.append("Tip: open shared/STATUS_BOARD.md for the live board.")
        write_message(
            root,
            state,
            sender="oteam",
            recipient="pm",
            subject="Supervisor digest",
            body="\n".join(lines),
            msg_type="MESSAGE",
            nudge=True,
            start_if_needed=True,
        )

    def handle_customer_commands(last_body: str) -> bool:
        cmds = _split_slash_commands(last_body or "")
        if not cmds:
            return False

        handled_any = False
        try:
            store = load_tickets(root, state)
        except Exception:
            store = _ticket_store_default()

        for cmd, argv in cmds:
            if cmd in {"help", "commands"}:
                help_txt = "\n".join(
                    [
                        "Customer commands:",
                        "- /tickets            (list open/blocked tickets)",
                        "- /ticket T001        (show one ticket)",
                        "- /status             (team + ticket snapshot)",
                        "- /tail <agent> [N]   (capture last N lines from tmux pane, default 120)",
                        "- /approve <TOKEN>    (approve a pending 'spawn agent' request)",
                        "- /deny <TOKEN>       (deny a pending 'spawn agent' request)",
                    ]
                )
                write_message(
                    root,
                    state,
                    sender="oteam",
                    recipient="customer",
                    subject="Available commands",
                    body=help_txt,
                    nudge=False,
                )
                handled_any = True
            elif cmd == "tickets":
                write_message(
                    root,
                    state,
                    sender="oteam",
                    recipient="customer",
                    subject="Ticket summary (open/blocked)",
                    body=_format_ticket_summary(store),
                    nudge=False,
                )
                handled_any = True
            elif cmd == "ticket":
                body = (
                    "Usage: /ticket T001"
                    if not argv
                    else _format_ticket_detail(store, argv[0])
                )
                write_message(
                    root,
                    state,
                    sender="oteam",
                    recipient="customer",
                    subject="Ticket detail",
                    body=body,
                    nudge=False,
                )
                handled_any = True
            elif cmd in {"status", "dashboard"}:
                try:
                    dash = compute_dashboard_snapshot(
                        root,
                        state,
                        session=state["tmux"]["session"],
                        last_pane_change=last_pane_change,
                        last_mail_activity=last_mail_activity,
                    )
                    body = format_dashboard_for_customer(dash)
                except Exception:
                    body = "Dashboard unavailable."
                write_message(
                    root,
                    state,
                    sender="oteam",
                    recipient="customer",
                    subject="Status",
                    body=body,
                    nudge=False,
                )
                handled_any = True
            elif cmd in {"tail", "pane"}:
                if not argv:
                    body = "Usage: /tail <agent> [N]"
                else:
                    agent = argv[0]
                    try:
                        n = int(argv[1]) if len(argv) > 1 else 120
                    except Exception:
                        n = 120
                    txt = tmux_capture_tail(state["tmux"]["session"], agent, lines=n)
                    if not txt.strip():
                        body = f"(no pane output captured for {agent})"
                    else:
                        body = f"Last {n} lines from tmux:{agent}\n\n```\n{txt.rstrip()}\n```"
                write_message(
                    root,
                    state,
                    sender="oteam",
                    recipient="customer",
                    subject="tmux capture",
                    body=body,
                    nudge=False,
                )
                handled_any = True
            elif cmd in {"approve", "deny"}:
                if not argv:
                    body = f"Usage: /{cmd} <TOKEN>"
                    write_message(
                        root,
                        state,
                        sender="oteam",
                        recipient="customer",
                        subject="Spawn approval",
                        body=body,
                        nudge=False,
                    )
                    handled_any = True
                else:
                    token = argv[0].strip().upper()
                    with spawn_lock(root):
                        store_sr = load_spawn_requests(root)
                        req = find_spawn_request_by_token(store_sr, token)
                        if not req:
                            write_message(
                                root,
                                state,
                                sender="oteam",
                                recipient="customer",
                                subject="Spawn approval",
                                body=f"Unknown token: {token}",
                                nudge=False,
                            )
                            handled_any = True
                        elif req.get("status") != "pending":
                            write_message(
                                root,
                                state,
                                sender="oteam",
                                recipient="customer",
                                subject="Spawn approval",
                                body=f"Request {req.get('id')} is already {req.get('status')}.",
                                nudge=False,
                            )
                            handled_any = True
                        else:
                            if cmd == "deny":
                                req["status"] = "denied"
                                req["decided_at"] = now_iso()
                                req["decided_by"] = "customer"
                                save_spawn_requests(root, store_sr)
                                note = f"Denied request {req.get('id')} (role={req.get('role')})."
                                write_message(
                                    root,
                                    state,
                                    sender="oteam",
                                    recipient="customer",
                                    subject="Spawn denied",
                                    body=note,
                                    nudge=False,
                                )
                                write_message(
                                    root,
                                    state,
                                    sender="oteam",
                                    recipient="pm",
                                    subject="Spawn denied by customer",
                                    body=note,
                                    nudge=True,
                                )
                                handled_any = True
                            else:
                                req["status"] = "approved"
                                req["decided_at"] = now_iso()
                                req["decided_by"] = "customer"
                                save_spawn_requests(root, store_sr)
                                try:
                                    agent = add_agent_to_workspace(
                                        root,
                                        state,
                                        role=req.get("role") or "developer",
                                        name=req.get("name"),
                                        title=req.get("title"),
                                        persona=req.get("persona"),
                                        start_opencode=bool(
                                            req.get("start_opencode", True)
                                        ),
                                    )
                                    req["status"] = "fulfilled"
                                    req["fulfilled_at"] = now_iso()
                                    req["fulfilled_agent"] = agent.get("name")
                                    save_spawn_requests(root, store_sr)
                                    note = f"Approved {req.get('id')} — spawned agent {agent.get('name')} ({agent.get('role')})."
                                    write_message(
                                        root,
                                        state,
                                        sender="oteam",
                                        recipient="customer",
                                        subject="Spawn approved",
                                        body=note,
                                        nudge=False,
                                    )
                                    write_message(
                                        root,
                                        state,
                                        sender="oteam",
                                        recipient="pm",
                                        subject="Spawn approved — agent created",
                                        body=note,
                                        nudge=True,
                                    )
                                    handled_any = True
                                except Exception as e:
                                    note = f"Spawn failed after approval: {e}"
                                    write_message(
                                        root,
                                        state,
                                        sender="oteam",
                                        recipient="customer",
                                        subject="Spawn failed",
                                        body=note,
                                        nudge=False,
                                    )
                                    write_message(
                                        root,
                                        state,
                                        sender="oteam",
                                        recipient="pm",
                                        subject="Spawn failed",
                                        body=note,
                                        nudge=True,
                                    )
                                    handled_any = True

        if handled_any:
            # Treat automated command responses as "reply" to avoid PM nag spam.
            try:
                cust_state = load_customer_state(root)
                cust_state["last_pm_reply_ts"] = now_iso()
                save_customer_state(root, cust_state)
            except Exception:
                pass
        return handled_any

    def maybe_send_customer_burst_digest() -> None:
        sup = state.get("supervision") or {}
        quiet_s = float(sup.get("customer_burst_seconds", 20) or 20)
        if quiet_s <= 0:
            return
        try:
            cust_state = load_customer_state(root)
        except Exception:
            return
        if not cust_state.get("burst_pending"):
            return
        last_ts = iso_to_unix(str(cust_state.get("burst_last_ts") or ""))
        if not last_ts:
            return
        if time.time() - last_ts < quiet_s:
            return

        buf_path = customer_burst_buffer_path(root)
        try:
            buf = buf_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            buf = ""
        if not buf.strip():
            cust_state["burst_pending"] = False
            cust_state["burst_count"] = 0
            cust_state.pop("burst_start_ts", None)
            cust_state.pop("burst_last_ts", None)
            save_customer_state(root, cust_state)
            return

        parts = re.split(r"(?m)^(?=## )", buf)
        entries = [p for p in (x.strip() for x in parts) if p.startswith("## ")]
        blocks: List[str] = []
        for e in entries[-12:]:
            ts0, sender, _, subj, body, _, _ = _parse_entry(e[:4000])
            # Filter out pure slash-command messages from the digest; those are handled immediately.
            if (body or "").lstrip().startswith("/"):
                continue
            blocks.append(
                f"---\n{ts0 or ''} | {subj or '(no subject)'}\n{(body or '').strip()}"
            )

        digest = "\n\n".join(blocks).strip()
        if not digest:
            digest = f"(customer burst had {len(entries)} message(s), mostly commands)"
        if len(digest) > 6500:
            digest = digest[:6500] + "\n...(truncated)"

        write_message(
            root,
            state,
            sender="oteam",
            recipient="pm",
            subject=f"[CUSTOMER] {len(entries)} new messages - reply now",
            body=digest
            + "\n\n**IMMEDIATE ACTION REQUIRED**: Reply to the customer FIRST.\n"
            + "Use: `oteam msg --to customer --subject 'Re: ...' --body '...'`\n"
            + "Then update shared/BRIEF.md with any new information.",
            msg_type="MESSAGE",
            nudge=True,
            start_if_needed=True,
        )

        # Keep the team moving: ask a non-PM agent to draft a response if available.
        try:
            helper = None
            for a in state["agents"]:
                if a["name"] == "pm":
                    continue
                if a.get("role") in {"researcher", "architect"}:
                    helper = a["name"]
                    break
            if helper:
                write_message(
                    root,
                    state,
                    sender="oteam",
                    recipient=helper,
                    subject="Assist PM: draft response + propose tickets",
                    body=digest
                    + "\n\nTask: draft a customer reply and propose concrete tickets; send your draft to PM.",
                    msg_type="MESSAGE",
                    nudge=True,
                    start_if_needed=True,
                )
        except Exception:
            pass

        cust_state["burst_pending"] = False
        cust_state["burst_count"] = 0
        cust_state["burst_digest_ts"] = now_iso()
        cust_state.pop("burst_start_ts", None)
        cust_state.pop("burst_last_ts", None)
        save_customer_state(root, cust_state)
        try:
            atomic_write_text(buf_path, "")
        except Exception:
            pass

    def maybe_autodispatch() -> None:
        coord = state.get("coordination") or {}
        if not bool(coord.get("auto_dispatch", False)):
            return
        if str(coord.get("stage", "discovery")).lower() not in {
            "execution",
            "implement",
            "implementation",
        }:
            return
        try:
            store = load_tickets(root, state)
        except Exception:
            return
        ready = [
            tt for tt in (store.get("tickets") or []) if ticket_is_ready(store, tt)
        ]
        if not ready:
            return

        active_by_assignee: Dict[str, int] = {}
        for tt in store.get("tickets") or []:
            if tt.get("status") in {
                TICKET_STATUS_OPEN,
                TICKET_STATUS_BLOCKED,
            } and tt.get("assignee"):
                a = str(tt.get("assignee"))
                active_by_assignee[a] = active_by_assignee.get(a, 0) + 1

        sup = state.get("supervision") or {}
        idle_thresh = float(sup.get("agent_idle_seconds", 120) or 120)
        candidates: List[str] = []
        for a in state["agents"]:
            if a["name"] == "pm":
                continue
            if active_by_assignee.get(a["name"], 0) > 0:
                continue
            idle_for = time.time() - float(last_pane_change.get(a["name"], time.time()))
            if idle_for < idle_thresh:
                continue
            candidates.append(a["name"])

        if not candidates:
            return

        ready_sorted = sorted(
            ready, key=lambda x: (_ticket_priority_value(x), x.get("id", ""))
        )
        assignee = candidates[0]
        tt = ready_sorted[0]
        tid = tt.get("id", "")
        if not tid:
            return
        try:
            ticket_assign(
                root,
                state,
                ticket_id=tid,
                assignee=assignee,
                user="oteam",
                note="auto-dispatch",
            )
        except Exception:
            return
        body = (
            _ticket_summary_block(tt)
            + "\n\nYou were auto-assigned this READY ticket.\n"
            + "Protocol: implement, run tests, update STATUS.md, and report back to PM with results + next steps."
        )
        write_message(
            root,
            state,
            sender="oteam",
            recipient=assignee,
            subject=f"Auto-assignment: {tid}",
            body=body,
            msg_type=ASSIGNMENT_TYPE,
            ticket_id=tid,
            nudge=True,
            start_if_needed=True,
        )
        nudge_queue.request("pm", reason=f"AUTO-DISPATCHED {tid} to @{assignee}")

    while True:
        customer_nag_pending = False
        customer_state_for_nag: Optional[Dict[str, Any]] = None
        pm_idle_pending = False

        try:
            state = load_state(root)
            agent_names = {a["name"] for a in state["agents"]}
        except Exception:
            pass
        now_for_new = time.time()
        for a in state["agents"]:
            n = a["name"]
            last_agent_reply.setdefault(n, now_for_new)
            last_resume_prompt.setdefault(n, 0.0)
            resume_prompt_count.setdefault(n, 0)
            last_resume_escalation.setdefault(n, 0.0)
            try:
                mt = (root / a["dir_rel"] / "STATUS.md").stat().st_mtime
                last_status_mtime[n] = mt
            except FileNotFoundError:
                last_status_mtime[n] = last_status_mtime.get(n, now_for_new)

        # Start/stop Telegram bridge based on state.
        want_enabled = bool((state.get("telegram") or {}).get("enabled", False))
        if want_enabled != telegram_enabled:
            telegram_enabled = want_enabled
            if telegram_bridge:
                try:
                    telegram_bridge.stop()
                except Exception:
                    pass
                telegram_bridge = None
            if telegram_enabled:
                try:
                    telegram_bridge = TelegramBridge(root, state)
                    telegram_bridge.start()
                    log_line(root, "[router] telegram bridge enabled")
                except Exception as e:
                    log_line(root, f"[router] telegram bridge failed to start: {e}")
                    telegram_bridge = None

        session = state["tmux"]["session"]
        if not tmux_has_session(session):
            time.sleep(interval)
            continue

        windows = set(tmux_list_windows(session))
        now = time.time()

        # Update pane activity signatures (for true idleness).
        for a in state["agents"]:
            name = a["name"]
            if name not in windows:
                continue
            sig = _pane_sig(session, name)
            noise_until = float(_router_noise_until.get(name, 0.0) or 0.0)
            if noise_until and now < noise_until:
                # Skip router-injected noise so we don't reset idleness falsely.
                continue
            if name not in pane_sig:
                pane_sig[name] = sig
                last_pane_change[name] = now
            elif sig != pane_sig[name]:
                pane_sig[name] = sig
                last_pane_change[name] = now

        # Scan inboxes + message.md changes.
        for a in state["agents"]:
            name = a["name"]
            msg_path, inbox_dir, _ = mailbox_paths(root, name)

            new_files: List[Path] = []
            try:
                current = set(p.name for p in inbox_dir.glob("*.md"))
            except FileNotFoundError:
                mkdirp(inbox_dir)
                current = set()
            prev = seen_inbox.get(name, set())
            for fn in sorted(current - prev):
                new_files.append(inbox_dir / fn)
            seen_inbox[name] = current

            try:
                st = msg_path.stat()
                sig = (st.st_mtime_ns, st.st_size)
            except FileNotFoundError:
                sig = (0, 0)
            if last_mail_sig.get(name) != sig:
                last_mail_sig[name] = sig
                new_files.append(msg_path)
                last_mail_activity[name] = time.time()

            if not new_files:
                continue

            last_txt = ""
            last_sender: Optional[str] = None
            last_type: Optional[str] = None
            last_ticket: Optional[str] = None
            last_body: str = ""
            latest_ts: Optional[str] = None
            latest_entry: str = ""
            last_sig: Optional[str] = None
            last_ts: Optional[str] = None
            for fp in sorted(new_files, reverse=True):
                try:
                    txt = read_last_entry(fp)
                except Exception:
                    txt = ""
                if not txt or not txt.strip():
                    continue
                ts, sender, _, _, body, ticket, msg_type = _parse_entry(txt)
                if ts and (not latest_ts or ts > latest_ts):
                    latest_ts = ts
                    latest_entry = txt
                    last_sender = sender
                    last_body = body
                    last_ticket = ticket
                    last_type = msg_type
            if latest_entry and latest_entry.strip():
                last_txt = latest_entry
                last_ts = latest_ts
                if last_txt and last_txt.strip():
                    last_sig = hashlib.sha256(
                        last_txt.strip().encode("utf-8", "ignore")
                    ).hexdigest()

            # Deduplicate mailbox nudges: only react when the newest entry changes.
            if last_sig and last_seen_mail_sig.get(name) == last_sig:
                continue
            if last_ts:
                last_seen_mail_ts[name] = last_ts
            if last_sig:
                last_seen_mail_sig[name] = last_sig

            if last_sender and last_sender in agent_names:
                last_agent_reply[last_sender] = time.time()
                resume_prompt_count[last_sender] = 0
                last_resume_escalation[last_sender] = 0.0

            # Start agent only on assignment.
            if last_sender and last_type:
                if last_type == ASSIGNMENT_TYPE or (
                    bool(
                        (state.get("coordination") or {}).get(
                            "start_agents_on_assignment", True
                        )
                    )
                    and str(last_sender)
                    in set(
                        (state.get("coordination") or {}).get(
                            "assignment_from", ["pm", "oteam"]
                        )
                    )
                ):
                    maybe_start_agent_on_message(
                        root, state, name, sender=last_sender, msg_type=last_type
                    )

            # Customer-to-PM: handle commands immediately.
            if name == "pm" and (last_sender or "").lower() == "customer":
                try:
                    handled = handle_customer_commands(last_body or "")
                except Exception:
                    handled = False
                if handled:
                    continue

            reason = _mailbox_nudge_reason(last_ticket, last_type)
            sup = state.get("supervision") or {}
            cooldown = float(sup.get("mailbox_nudge_cooldown_seconds", 90) or 90)
            if name == "pm" and last_sender and last_sender == "customer":
                cooldown = 30
            now_nudge = time.time()
            if now_nudge - float(last_mailbox_nudge.get(name, 0.0)) >= cooldown:
                last_mailbox_nudge[name] = now_nudge
                nudge_queue.request(name, reason)

        # Digest bursty customer messages once they stop typing.
        try:
            maybe_send_customer_burst_digest()
        except Exception:
            pass

        # Auto-dispatch ready tickets if enabled.
        try:
            maybe_autodispatch()
        except Exception:
            pass

        # Dashboard output + PM digest
        dash = maybe_write_dashboard(session)
        if dash:
            try:
                maybe_send_pm_digest(dash)
            except Exception:
                pass

        # Customer wait loop (nag PM if customer hasn't been replied to).
        try:
            cust_state = load_customer_state(root)
            last_in = iso_to_unix(str(cust_state.get("last_customer_msg_ts", "")))
            last_out = iso_to_unix(str(cust_state.get("last_pm_reply_ts", "")))
            last_nag = iso_to_unix(str(cust_state.get("last_pm_nag_ts", "")))
            nag_count = int(cust_state.get("pm_nag_count", 0) or 0)
            if last_in and (not last_out or last_out < last_in):
                now_ts2 = time.time()
                if not last_nag or last_nag < last_in:
                    nag_count = 0
                if nag_count < 3 and (not last_nag or (now_ts2 - last_nag) >= 60):
                    nudge_queue.request(
                        "pm", reason="CUSTOMER WAITING — REPLY NOW", interrupt=True
                    )
                    customer_nag_pending = True
                    customer_state_for_nag = cust_state
        except Exception:
            pass

        # Agent-level stall/idle nudges based on pane idleness + ticket assignment.
        by_assignee: Dict[str, List[str]] = {}
        try:
            store = load_tickets(root, state)
            tickets = store.get("tickets") or []
            active = [
                tt
                for tt in tickets
                if tt.get("status") in {TICKET_STATUS_OPEN, TICKET_STATUS_BLOCKED}
            ]
            for tt in active:
                ass = tt.get("assignee") or ""
                if not ass:
                    continue
                by_assignee.setdefault(ass, []).append(tt.get("id", ""))

            sup = state.get("supervision") or {}
            stalled_s = float(sup.get("agent_stalled_nudge_seconds", 10) or 10)
            unassigned_s = float(sup.get("agent_unassigned_nudge_seconds", 600) or 600)
            # Track STATUS.md freshness for progress signals.
            for a in state["agents"]:
                try:
                    mt = (root / a["dir_rel"] / "STATUS.md").stat().st_mtime
                    if mt > last_status_mtime.get(a["name"], 0.0):
                        last_status_mtime[a["name"]] = mt
                        resume_prompt_count[a["name"]] = 0
                        last_resume_escalation[a["name"]] = 0.0
                except FileNotFoundError:
                    pass

            for a in state["agents"]:
                name = a["name"]
                if name == "pm":
                    continue
                idle_for = time.time() - float(last_pane_change.get(name, time.time()))
                assigned = [x for x in by_assignee.get(name, []) if x]
                if assigned:
                    if (
                        idle_for >= stalled_s
                        and (time.time() - float(last_agent_stall_nudge.get(name, 0.0)))
                        >= stalled_s
                    ):
                        last_agent_stall_nudge[name] = time.time()
                        nudge_queue.request(
                            name,
                            reason=f"STALLED — tickets {','.join(assigned)}. Analyse the problem, devise a solution, plan the steps, implement the change, verify it works. Deliver the next concrete step now; if blocked, mark ticket blocked and tell PM immediately.",
                            interrupt=True,
                        )
                else:
                    if (
                        idle_for >= unassigned_s
                        and (
                            time.time()
                            - float(last_agent_unassigned_nudge.get(name, 0.0))
                        )
                        >= unassigned_s
                    ):
                        last_agent_unassigned_nudge[name] = time.time()
                        nudge_queue.request(
                            name,
                            reason="NO WORK IN PROGRESS — analyse ready tickets, devise solutions, propose 1–2 concrete tasks to PM now.",
                            interrupt=True,
                        )
        except Exception:
            pass

        # Auto-resume assigned agents (OpenCode is single-shot; keep them moving).
        try:
            sup = state.get("supervision") or {}
            resume_s = float(sup.get("agent_resume_seconds", 180) or 180)
            resume_cooldown = float(
                sup.get("agent_resume_cooldown_seconds", resume_s) or resume_s
            )
            resume_max = int(sup.get("agent_resume_max_prompts", 2) or 2)
            remind_s = float(sup.get("assigned_ticket_reminder_seconds", 60) or 60)
            now_resume = time.time()
            for agent_name, assigned in by_assignee.items():
                if not assigned or agent_name == "pm":
                    continue
                # Idle reminder listing assigned tickets.
                idle_for = now_resume - float(
                    last_pane_change.get(agent_name, now_resume)
                )
                if (
                    idle_for >= remind_s
                    and (
                        now_resume - float(last_assigned_reminder.get(agent_name, 0.0))
                    )
                    >= remind_s
                ):
                    assigned_list = ", ".join(sorted(assigned))
                    body = (
                        "You still own these tickets:\n"
                        + "\n".join(f"- {tid}" for tid in sorted(assigned))
                        + "\n\nPick one now, analyse it, devise a solution, plan the steps, implement the change, verify it works, and send PM a brief status (result, next step, ETA). "
                        "If blocked, mark the ticket blocked with the reason and tell PM immediately."
                    )
                    write_message(
                        root,
                        state,
                        sender="oteam",
                        recipient=agent_name,
                        subject=f"Reminder: assigned tickets ({assigned_list})",
                        body=body,
                        msg_type="MESSAGE",
                        nudge=True,
                        start_if_needed=True,
                    )
                    last_assigned_reminder[agent_name] = now_resume
                    # Avoid double messaging in the same loop.
                    continue
                # If already escalated beyond threshold and no new progress, pause re-prompts until STATUS or message.
                if resume_prompt_count.get(agent_name, 0) >= resume_max:
                    if (
                        now_resume - float(last_resume_escalation.get(agent_name, 0.0))
                    ) >= resume_cooldown:
                        pm_body = (
                            f"Agent @{agent_name} has exceeded resume prompts ({resume_prompt_count.get(agent_name, 0)}x) "
                            f"on tickets {', '.join(sorted(assigned))} with no STATUS.md update. "
                            "Intervene: ask for concrete progress, reassign, or adjust scope."
                        )
                        write_message(
                            root,
                            state,
                            sender="oteam",
                            recipient="pm",
                            subject=f"Agent may be stuck: {agent_name}",
                            body=pm_body,
                            msg_type="MESSAGE",
                            nudge=True,
                            start_if_needed=True,
                        )
                        last_resume_escalation[agent_name] = now_resume
                    continue
                idle_since_pane = now_resume - float(
                    last_pane_change.get(agent_name, now_resume)
                )
                if idle_since_pane < resume_s:
                    continue
                if (
                    now_resume - float(last_agent_reply.get(agent_name, 0.0))
                ) < resume_s:
                    continue
                if (
                    now_resume - float(last_resume_prompt.get(agent_name, 0.0))
                ) < resume_cooldown:
                    continue
                status_age = now_resume - float(
                    last_status_mtime.get(agent_name, now_resume)
                )

                assigned_list = ", ".join(sorted(assigned))
                body = (
                    f"You are still assigned: {assigned_list}.\n\n"
                    "Resume work now. Pick the first ticket, do the next concrete change/test, then send PM a brief status "
                    "(what you did, result, next step, ETA). If blocked, mark the ticket blocked with the reason and notify PM. "
                    "Do not just acknowledge; ship the next increment in this run."
                )
                write_message(
                    root,
                    state,
                    sender="oteam",
                    recipient=agent_name,
                    subject="Resume assigned work",
                    body=body,
                    msg_type="MESSAGE",
                    nudge=True,
                    start_if_needed=True,
                )
                last_resume_prompt[agent_name] = now_resume
                resume_prompt_count[agent_name] = (
                    resume_prompt_count.get(agent_name, 0) + 1
                )

                # Escalate to PM if an agent keeps looping without visible progress (stale STATUS).
                if (
                    resume_prompt_count[agent_name] >= max(resume_max, 2)
                    and status_age >= resume_s
                    and (
                        now_resume - float(last_resume_escalation.get(agent_name, 0.0))
                    )
                    >= resume_cooldown
                ):
                    pm_body = (
                        f"Agent @{agent_name} has been re-prompted {resume_prompt_count[agent_name]} times "
                        f"for tickets {assigned_list} without a recent STATUS.md update "
                        f"(stale for ~{int(status_age)}s). Consider intervening: ask for concrete progress, "
                        "reassign, or adjust scope."
                    )
                    write_message(
                        root,
                        state,
                        sender="oteam",
                        recipient="pm",
                        subject=f"Agent may be looping: {agent_name}",
                        body=pm_body,
                        msg_type="MESSAGE",
                        nudge=True,
                        start_if_needed=True,
                    )
                    last_resume_escalation[agent_name] = now_resume
        except Exception:
            pass

        # PM-level idle: nudge PM when PM is idle (regardless of team status)
        try:
            sup = state.get("supervision") or {}
            idle_s = float(sup.get("agent_idle_seconds", 120) or 120)
            if not state["tmux"].get("paused", False):
                pm_idle = (
                    time.time() - float(last_pane_change.get("pm", time.time()))
                ) >= idle_s
                if pm_idle:
                    nudge_queue.request(
                        "pm",
                        reason="PM IDLE — analyse what's pending, devise priorities, plan the next steps, assign tickets, unblock agents",
                    )
                    pm_idle_pending = True
        except Exception:
            pass

        # Flush pending nudges (dedup + defer while panes are busy).
        try:
            for agent_name, ok, reason_text, skip_reason in nudge_queue.flush(state):
                if skip_reason in {"duplicate", "busy"}:
                    continue

                if (
                    ok
                    and customer_nag_pending
                    and agent_name == "pm"
                    and "CUSTOMER WAITING" in reason_text
                ):
                    try:
                        cust_state = customer_state_for_nag or load_customer_state(root)
                        cust_state["last_pm_nag_ts"] = now_iso()
                        cust_state["pm_nag_count"] = (
                            int(cust_state.get("pm_nag_count", 0) or 0) + 1
                        )
                        save_customer_state(root, cust_state)
                    except Exception:
                        pass
                if (
                    ok
                    and pm_idle_pending
                    and agent_name == "pm"
                    and "TEAM IDLE" in reason_text
                ):
                    # Acknowledge for logging
                    pass
        except Exception:
            pass

        time.sleep(interval)


def cmd_customer_chat(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    if not (root / STATE_FILENAME).exists():
        raise OTeamError(
            "customer-chat: could not find oteam.json in this directory or its parents"
        )
    state = load_state(root)

    history_path = root / DIR_RUNTIME / "customer_chat.history"
    mkdirp(history_path.parent)
    configure_readline(history_path)

    chat_dir = root / DIR_MAIL / "customer"
    mkdirp(chat_dir / "inbox")
    mkdirp(chat_dir / "outbox")
    mkdirp(chat_dir / "sent")
    chat_file = chat_dir / "message.md"
    if not chat_file.exists():
        atomic_write_text(
            chat_file,
            "# Customer channel\n\nUse this window to chat with the PM/team.\n\n",
        )

    print("=== Customer chat ===")
    print("Type a message and press Enter to send to PM. Ctrl+C to exit.\n")

    try:
        existing = chat_file.read_text(encoding="utf-8", errors="replace")
        if existing.strip():
            print(existing.rstrip())
    except Exception:
        pass

    stop = threading.Event()

    def tail_chat() -> None:
        pos = chat_file.stat().st_size if chat_file.exists() else 0
        buf = ""
        while not stop.is_set():
            try:
                with chat_file.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    new_txt = f.read()
                    pos = f.tell()
                if not new_txt:
                    stop.wait(0.5)
                    continue
                buf += new_txt
                parts = MESSAGE_SEPARATOR_RE.split(buf)
                complete = parts[:-1]
                buf = parts[-1]
                for entry in complete:
                    entry = entry.strip("\n")
                    if not entry:
                        continue
                    entry = entry + "\n---\n"
                    _, sender, recipient, _, _, _, _ = _parse_entry(entry)
                    if recipient and recipient != "customer":
                        continue
                    print("\n" + entry, end="", flush=True)
            except Exception:
                stop.wait(0.5)
                continue
            stop.wait(0.5)

    t = threading.Thread(target=tail_chat, daemon=True)
    t.start()

    try:
        while True:
            try:
                text = input("you> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                print("")
                break
            text = text.rstrip("\n")
            if not text.strip():
                continue
            write_message(
                root,
                state,
                sender="customer",
                recipient="pm",
                subject="Customer chat",
                body=text,
                msg_type="MESSAGE",
                task=None,
                nudge=True,
                start_if_needed=True,
            )
            try:
                readline.add_history(text)
                _readline_save_history(history_path)
            except Exception:
                pass
            print(f"[you → pm] {text}")
    finally:
        stop.set()
        t.join(timeout=1.0)
        print("Exiting customer chat.")


def cmd_chat(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    if not (root / STATE_FILENAME).exists():
        raise OTeamError(
            "chat: could not find oteam.json in this directory or its parents"
        )
    state = load_state(root)

    history_path = root / DIR_RUNTIME / f"chat_{slugify(args.to)}.history"
    configure_readline(history_path)

    agent_names = {a["name"] for a in state["agents"]}
    allowed = set(agent_names) | {"customer"}
    recipient = args.to
    if recipient not in allowed:
        raise OTeamError(f"unknown recipient: {recipient}")

    msg_path, inbox_dir, _ = mailbox_paths(root, recipient)
    mkdirp(inbox_dir)
    if not msg_path.exists():
        atomic_write_text(msg_path, f"# Inbox — {recipient}\n\n(append-only)\n\n")

    print(f"=== Chat with {recipient} ===")
    print("Type a message and press Enter to send. Ctrl+C to exit.\n")

    try:
        existing = msg_path.read_text(encoding="utf-8", errors="replace")
        if existing.strip():
            print(existing.rstrip())
    except Exception:
        pass

    stop = threading.Event()

    def tail_chat() -> None:
        pos = msg_path.stat().st_size if msg_path.exists() else 0
        while not stop.is_set():
            try:
                with msg_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    new_txt = f.read()
                    pos = f.tell()
                    if new_txt:
                        print("\n" + new_txt, end="", flush=True)
            except Exception:
                pass
            stop.wait(0.5)

    t = threading.Thread(target=tail_chat, daemon=True)
    t.start()

    try:
        while True:
            try:
                text = input("you> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                print("")
                break
            text = text.rstrip("\n")
            if not text.strip():
                continue
            write_message(
                root,
                state,
                sender=args.sender or "pm",
                recipient=recipient,
                subject=args.subject or "chat",
                body=text,
                msg_type="MESSAGE",
                task=None,
                nudge=not args.no_nudge,
                start_if_needed=args.start_if_needed,
            )
            print(f"[you → {recipient}] {text}")
    finally:
        stop.set()
        t.join(timeout=1.0)
        print("Exiting chat.")


def cmd_upload(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    if not (root / STATE_FILENAME).exists():
        raise OTeamError(
            "upload: could not find oteam.json in this directory or its parents"
        )
    state = load_state(root)

    dest_root = root / DIR_SHARED_DRIVE
    mkdirp(dest_root)

    uploaded: List[Tuple[Path, Path]] = []
    for src_str in args.paths:
        src = Path(src_str).expanduser()
        if not src.exists():
            print(f"skip: source does not exist: {src}", file=sys.stderr)
            continue
        dest = dest_root / (args.dest or src.name)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        uploaded.append((src, dest))
        print(f"Uploaded {src} -> {dest}")

    if uploaded:
        lines = ["Files uploaded to shared drive:", ""]
        for s, d in uploaded:
            lines.append(f"- {s} → {d}")
        lines.append("")
        lines.append("Check `shared/drive/` for the uploaded artifacts.")
        body = "\n".join(lines)
        write_message(
            root,
            state,
            sender=args.sender or "pm",
            recipient="pm",
            subject="Upload to shared drive",
            body=body,
            msg_type="MESSAGE",
            task=None,
            nudge=True,
            start_if_needed=True,
        )


def cmd_pause(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    state["tmux"]["paused"] = True
    save_state(root, state)

    session = state["tmux"]["session"]
    if tmux_has_session(session):
        # Move agent windows to standby shells.
        for agent in state["agents"]:
            try:
                try:
                    tmux_send_keys(
                        session, agent["name"], ["Escape", "Escape", "Escape", "C-c"]
                    )
                except Exception:
                    pass
                cmd_args = standby_window_command(root, state, agent)
                tmux_respawn_window(
                    session,
                    agent["name"],
                    Path(agent["dir_abs"]),
                    command_args=cmd_args,
                )
            except Exception:
                pass
        # Quiet router window.
        try:
            try:
                tmux_send_keys(
                    session, ROUTER_WINDOW, ["Escape", "Escape", "Escape", "C-c"]
                )
            except Exception:
                pass
            tmux_respawn_window(
                session,
                ROUTER_WINDOW,
                root,
                command_args=pick_shell()
                + ["printf 'oteam paused; router idle\\n'; exec $SHELL"],
            )
        except Exception:
            pass
    print(f"paused tmux session: {state['tmux']['session']}")


# -----------------------------
# High-level operations: init/import/resume/open
# -----------------------------


def create_root_structure(root: Path, state: Dict[str, Any]) -> bool:
    mkdirp(root)
    install_self_into_root(root)
    created_tickets = ensure_shared_scaffold(root, state)
    update_roster(root, state)
    save_state(root, state)
    return created_tickets


def ensure_tmux(root: Path, state: Dict[str, Any], *, launch_opencode: bool) -> None:
    print(
        f"DEBUG: ensure_tmux called with launch_opencode={launch_opencode}",
        file=sys.stderr,
    )
    ensure_tmux_session(root, state)
    ensure_router_window(root, state)
    ensure_customer_window(root, state)
    ensure_agent_windows(root, state, launch_opencode=launch_opencode)


def ensure_agent_windows(
    root: Path, state: Dict[str, Any], *, launch_opencode: bool
) -> None:
    session = state["tmux"]["session"]
    auto = set(autostart_agent_names(state))
    print(
        f"DEBUG: ensure_agent_windows: launch_opencode={launch_opencode}, auto={auto}",
        file=sys.stderr,
    )

    agents = sorted(state["agents"], key=lambda a: (a["name"] != "pm", a["name"]))
    for agent in agents:
        w = agent["name"]
        agent_dir = Path(agent["dir_abs"])
        if launch_opencode and w in auto:
            if is_opencode_running(state, w):
                continue
            start_opencode_in_window(root, state, agent, boot=True)
            continue

        windows = set(tmux_list_windows(session))
        if w not in windows:
            cmd_args = standby_window_command(root, state, agent)
            tmux_new_window(session, w, agent_dir, command_args=cmd_args)


# -----------------------------
# Commands
# -----------------------------


def _init_wizard(
    root: Path, state: Dict[str, Any], *, ask_telegram: bool = True
) -> Dict[str, Any]:
    """Interactive init wizard: capture seed text and optionally configure Telegram."""
    history_path = root / DIR_RUNTIME / "init_wizard.history"
    configure_readline(history_path)

    seed_file = root / DIR_SEED / "SEED.md"
    if (
        not seed_file.exists()
        or not seed_file.read_text(encoding="utf-8", errors="replace").strip()
    ):
        seed_txt = prompt_multiline(
            "Paste customer/seed requirements (optional).", history_path=history_path
        )
        if seed_txt.strip():
            atomic_write_text(seed_file, "# Seed input\n\n" + seed_txt.strip() + "\n")

    extras_file = root / DIR_SEED_EXTRAS / "EXTRAS.md"
    if (
        not extras_file.exists()
        or not extras_file.read_text(encoding="utf-8", errors="replace").strip()
    ):
        extras_txt = prompt_multiline(
            "Paste seed-extras (links, research notes) (optional).",
            history_path=history_path,
        )
        if extras_txt.strip():
            atomic_write_text(
                extras_file, "# Seed extras\n\n" + extras_txt.strip() + "\n"
            )

    if ask_telegram and prompt_yes_no(
        "Configure Telegram customer chat now?", default=False
    ):
        token = prompt_line("Telegram bot token (from @BotFather)", default="")
        phone = prompt_line(
            "Authorized phone number (digits; E.164 preferred, e.g. +15551234567)",
            default="",
        )
        cfg = {
            "configured_at": now_iso(),
            "bot_token": token.strip(),
            "authorized_phone": phone.strip(),
            "chat_id": None,
            "user_id": None,
            "update_offset": 0,
        }
        telegram_save_config(root, state, cfg)

        if prompt_yes_no("Enable Telegram integration now?", default=True):
            state.setdefault("telegram", {})
            state["telegram"]["enabled"] = True
            save_state(root, state)
            print(
                "Telegram enabled. After startup, open your bot in Telegram and send /start, then share contact."
            )

    _readline_save_history(history_path)
    return state


def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.workdir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.force:
        raise OTeamError(f"directory not empty: {root} (use --force to reuse)")
    mkdirp(root)

    ensure_executable("git", hint="Install git")
    if not args.no_tmux:
        ensure_executable("tmux", hint="Install tmux")
    if not args.no_opencode:
        if shutil.which(DEFAULT_OPENCODE_CMD) is None:
            print(
                f"warning: opencode executable not found: {DEFAULT_OPENCODE_CMD} (windows will be shells)",
                file=sys.stderr,
            )

    project_name = args.name or root.name
    state = build_state(
        root,
        project_name,
        devs=args.devs,
        mode="new",
        imported_from=None,
        opencode_cmd=DEFAULT_OPENCODE_CMD,
        opencode_model=args.model,
        sandbox=args.sandbox,
        approval=args.ask_for_approval,
        search=not args.no_search,
        full_auto=args.full_auto,
        yolo=args.yolo,
        autostart=args.autostart,
        router=not args.no_router,
    )
    save_state(root, state)

    _ = create_root_structure(root, state)
    create_git_scaffold_new(root, state)
    ensure_agents_created(root, state)
    update_roster(root, state)
    save_state(root, state)

    state = load_state(root)
    if sys.stdin.isatty() and not args.no_interactive:
        state = _init_wizard(root, state, ask_telegram=True)
        state = load_state(root)
    write_message(
        root,
        state,
        sender="oteam",
        recipient="pm",
        subject="Kickoff",
        body="Start by filling shared/GOALS.md and shared/PLAN.md (high-level). Track and assign all work via tickets (`cteam tickets ...`, `oteam assign ...`).",
        msg_type=ASSIGNMENT_TYPE,
        task="PM-KICKOFF",
        nudge=True,
        start_if_needed=True,
    )

    if args.no_tmux:
        print(f"Initialized workspace at {root}")
        print(f"Open tmux later: python3 oteam.py {shlex.quote(str(root))} open")
        return

    launch_opencode = not args.no_opencode and not state["tmux"].get("paused", False)
    ensure_tmux(root, state, launch_opencode=launch_opencode)
    print(f"tmux session: {state['tmux']['session']}")
    if not args.no_attach:
        tmux_attach(state["tmux"]["session"], window=args.window)


def cmd_import(args: argparse.Namespace) -> None:
    root = Path(args.workdir).expanduser().resolve()
    src = args.src

    if root.exists() and any(root.iterdir()) and not args.force:
        raise OTeamError(f"directory not empty: {root} (use --force to reuse)")
    mkdirp(root)

    ensure_executable("git", hint="Install git")
    if not args.no_tmux:
        ensure_executable("tmux", hint="Install tmux")
    if not args.no_opencode:
        if shutil.which(DEFAULT_OPENCODE_CMD) is None:
            print(
                f"warning: opencode executable not found: {DEFAULT_OPENCODE_CMD} (windows will be shells)",
                file=sys.stderr,
            )

    project_name = args.name or (Path(src).name if src else root.name)
    state = build_state(
        root,
        project_name,
        devs=args.devs,
        mode="import",
        imported_from=src,
        opencode_cmd=DEFAULT_OPENCODE_CMD,
        opencode_model=args.model,
        sandbox=args.sandbox,
        approval=args.ask_for_approval,
        search=not args.no_search,
        full_auto=args.full_auto,
        yolo=args.yolo,
        autostart=args.autostart,
        router=not args.no_router,
    )
    # For imports, start in a paused state so agents/OpenCode do nothing
    # until the customer actually asks for work.
    state.setdefault("tmux", {})["paused"] = True
    save_state(root, state)

    _ = create_root_structure(root, state)

    if args.seed:
        seed_file = root / DIR_SEED / "SEED.md"
        seed_path = Path(args.seed).expanduser().resolve()
        if not seed_path.exists():
            raise OTeamError(f"seed file not found: {seed_path}")
        mkdirp(seed_file.parent)
        atomic_write_text(seed_file, seed_path.read_text(encoding="utf-8"))

    import_seed_text = create_git_scaffold_import(root, state, src)
    if import_seed_text and not args.seed:
        seed_file = root / DIR_SEED / "SEED.md"
        existing = seed_file.read_text(encoding="utf-8") if seed_file.exists() else ""
        if not existing.strip():
            mkdirp(seed_file.parent)
            atomic_write_text(seed_file, import_seed_text)

    ensure_agents_created(root, state)
    update_roster(root, state)
    save_state(root, state)

    # Record that this imported workspace should stay idle until the
    # first customer message arrives, at which point kickoff/recon
    # assignments will be sent and OpenCode will be started.
    try:
        cust_state = load_customer_state(root)
        cust_state["import_pending"] = True
        if args.recon:
            cust_state["import_recon"] = True
        save_customer_state(root, cust_state)
    except Exception:
        pass

    if args.no_tmux:
        print(f"Imported workspace at {root}")
        print(f"Open tmux later: python3 oteam.py {shlex.quote(str(root))} open")
        return

    # Do not launch OpenCode on import; wait for the customer to ask.
    launch_opencode = False
    ensure_tmux(root, state, launch_opencode=launch_opencode)
    print(f"tmux session: {state['tmux']['session']}")
    if not args.no_attach:
        tmux_attach(state["tmux"]["session"], window=args.window)


def cmd_resume(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    if args.autostart:
        state["coordination"]["autostart"] = args.autostart
    if args.no_router:
        state["tmux"]["router"] = False
    if args.router:
        state["tmux"]["router"] = True
    save_state(root, state)

    if (
        not (root / DIR_PROJECT_BARE).exists()
        or not (root / DIR_PROJECT_BARE / "HEAD").exists()
    ):
        git_init_bare(root / DIR_PROJECT_BARE, branch=state["git"]["default_branch"])
    if not (root / DIR_PROJECT_CHECKOUT).exists():
        git_clone(root / DIR_PROJECT_BARE, root / DIR_PROJECT_CHECKOUT)

    ensure_agents_created(root, state)
    update_roster(root, state)
    state["tmux"]["paused"] = False
    save_state(root, state)

    if not (root / TICKET_MIGRATION_FLAG).exists():
        try:
            write_message(
                root,
                state,
                sender="oteam",
                recipient="pm",
                subject="Ticket system initialized",
                body=(
                    "A ticket database was created automatically (shared/TICKETS.json). "
                    "Please migrate any outstanding tasks/plan items into tickets and retire duplicate entries in legacy files. "
                    "Use `oteam tickets` to manage tickets and `oteam assign --ticket ...` for all assignments."
                ),
                msg_type="MESSAGE",
                nudge=True,
                start_if_needed=True,
            )
            atomic_write_text(root / TICKET_MIGRATION_FLAG, now_iso())
        except Exception:
            pass

    if args.no_tmux:
        print(f"Workspace ready at {root} (tmux disabled)")
        return

    session = state["tmux"]["session"]
    if tmux_has_session(session):
        print(f"tmux session already running: {session}")
        if not args.no_attach:
            tmux_attach(session, window=args.window)
        return

    launch_opencode = not args.no_opencode
    ensure_tmux(root, state, launch_opencode=launch_opencode)
    print(f"tmux session: {session}")
    if not args.no_attach:
        tmux_attach(session, window=args.window)
        return

    launch_opencode = not args.no_opencode
    ensure_tmux(root, state, launch_opencode=launch_opencode)
    print(f"tmux session: {session}")
    if not args.no_attach:
        tmux_attach(session, window=args.window)


def cmd_open(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    new_tickets = create_root_structure(root, state)
    if (
        not (root / DIR_PROJECT_BARE).exists()
        or not (root / DIR_PROJECT_BARE / "HEAD").exists()
    ):
        git_init_bare(root / DIR_PROJECT_BARE, branch=state["git"]["default_branch"])
    if not (root / DIR_PROJECT_CHECKOUT).exists():
        git_clone(root / DIR_PROJECT_BARE, root / DIR_PROJECT_CHECKOUT)

    ensure_agents_created(root, state)
    update_roster(root, state)
    save_state(root, state)

    if new_tickets and not (root / TICKET_MIGRATION_FLAG).exists():
        try:
            write_message(
                root,
                state,
                sender="oteam",
                recipient="pm",
                subject="Ticket system initialized",
                body=(
                    "A ticket database was created automatically (shared/TICKETS.json). "
                    "Please migrate any outstanding tasks/plan items into tickets and retire duplicate entries in legacy files. "
                    "Use `oteam tickets` to manage tickets and `oteam assign --ticket ...` for all assignments."
                ),
                msg_type="MESSAGE",
                nudge=True,
                start_if_needed=True,
            )
            atomic_write_text(root / TICKET_MIGRATION_FLAG, now_iso())
        except Exception:
            pass
    if not args.no_tmux:
        launch_opencode = not args.no_opencode and not state["tmux"].get(
            "paused", False
        )
        ensure_tmux(root, state, launch_opencode=launch_opencode)
        print(f"tmux session: {state['tmux']['session']}")
        if not args.no_attach:
            tmux_attach(state["tmux"]["session"], window=args.window)
    else:
        print(f"tmux disabled. Workspace at {root}")


def cmd_attach(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    ensure_executable("tmux")
    session = state["tmux"]["session"]
    if not tmux_has_session(session):
        raise OTeamError(
            f"tmux session not running: {session} (run: python3 oteam.py {root} open)"
        )
    tmux_attach(session, window=args.window)


def cmd_kill(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)
    ensure_executable("tmux")
    tmux_kill_session(state["tmux"]["session"])
    print(f"killed tmux session: {state['tmux']['session']}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    state = load_state(root)

    dash: Optional[Dict[str, Any]] = None
    p = dashboard_json_path(root)
    if p.exists() and not getattr(args, "fresh", False):
        try:
            dash = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            dash = None

    if dash is None:
        session = state["tmux"]["session"]
        if not tmux_has_session(session):
            raise OTeamError(
                f"tmux session not running: {session} (start router with: oteam watch)"
            )
        now_ts = time.time()
        dummy = {a["name"]: now_ts for a in state["agents"]}
        dash = compute_dashboard_snapshot(
            root,
            state,
            session=session,
            last_pane_change=dummy,
            last_mail_activity=dummy,
        )

    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(dash, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(format_status_board_md(dash))


def cmd_capture(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    state = load_state(root)

    session = state["tmux"]["session"]
    if not tmux_has_session(session):
        raise OTeamError(f"tmux session not running: {session}")

    window = args.agent
    txt = tmux_capture_tail(session, window, lines=int(args.lines or 200))
    if not txt.strip():
        txt = "(no pane output captured)"

    if getattr(args, "to", None):
        body = f"Capture from tmux:{window} (last {int(args.lines or 200)} lines)\n\n```\n{txt.rstrip()}\n```"
        write_message(
            root,
            state,
            sender="oteam",
            recipient=args.to,
            subject=f"tmux capture: {window}",
            body=body,
            msg_type="MESSAGE",
            nudge=(args.to != "customer"),
            start_if_needed=True,
        )
    else:
        sys.stdout.write(txt)


def cmd_request_agent(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    state = load_state(root)

    requested_by = os.environ.get("CTEAM_AGENT") or "pm"
    reason = (args.reason or "").strip()
    if not reason:
        raise OTeamError("--reason is required")

    req = create_spawn_request(
        root,
        requested_by=requested_by,
        role=args.role,
        name=args.name,
        title=args.title,
        persona=args.persona,
        start_opencode=bool(args.start_opencode),
        reason=reason,
    )

    customer_body = textwrap.dedent(
        f"""\
        PM requests spawning a new agent.

        Request: {req.get("id")}
        Role: {req.get("role")}
        Name: {req.get("name") or "(auto)"}
        Start opencode immediately: {str(bool(req.get("start_opencode"))).lower()}

        Reason:
        {reason}

        To approve:  /approve {req.get("token")}
        To deny:     /deny {req.get("token")}
        """
    ).strip()

    write_message(
        root,
        state,
        sender="oteam",
        recipient="customer",
        subject="Approval requested: spawn new agent",
        body=customer_body,
        msg_type="MESSAGE",
        nudge=False,
        start_if_needed=False,
    )
    write_message(
        root,
        state,
        sender="oteam",
        recipient="pm",
        subject="Spawn request sent to customer",
        body=f"Sent spawn request {req.get('id')} (role={req.get('role')}, token={req.get('token')}).",
        msg_type="MESSAGE",
        nudge=True,
        start_if_needed=True,
    )


def cmd_stage(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    state = load_state(root)
    stage = (args.stage or "").strip().lower()
    if stage not in {"discovery", "planning", "execution"}:
        raise OTeamError("stage must be one of: discovery, planning, execution")
    coord = state.get("coordination") or {}
    if not isinstance(coord, dict):
        coord = {}
    coord["stage"] = stage
    state["coordination"] = coord
    save_state(root, state)
    sys.stdout.write(f"Set coordination.stage = {stage}\n")


def cmd_autodispatch(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    state = load_state(root)
    mode = (args.mode or "").strip().lower()
    if mode in {"on", "true", "1", "yes"}:
        val = True
    elif mode in {"off", "false", "0", "no"}:
        val = False
    else:
        raise OTeamError("mode must be on/off (or true/false)")
    coord = state.get("coordination") or {}
    if not isinstance(coord, dict):
        coord = {}
    coord["auto_dispatch"] = bool(val)
    state["coordination"] = coord
    save_state(root, state)
    sys.stdout.write(f"Set coordination.auto_dispatch = {str(bool(val)).lower()}\n")


def cmd_status(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    print(f"Project: {state['project_name']}")
    print(f"Root: {root}")
    print(f"Mode: {state.get('mode', 'new')}")
    print(f"tmux session: {state['tmux']['session']}")
    print(f"autostart: {state.get('coordination', {}).get('autostart')}")
    tg = state.get("telegram") or {}
    print(
        f"telegram: enabled={bool(tg.get('enabled', False))} config={tg.get('config_rel', TELEGRAM_CONFIG_REL)}"
    )
    print("")

    for a in state["agents"]:
        a_dir = root / a["dir_rel"]
        status = a_dir / "STATUS.md"
        msg = a_dir / "message.md"
        s_mtime = (
            _dt.datetime.fromtimestamp(status.stat().st_mtime)
            if status.exists()
            else None
        )
        m_mtime = (
            _dt.datetime.fromtimestamp(msg.stat().st_mtime) if msg.exists() else None
        )
        print(f"== {a['name']} — {a['title']} ==")
        print(
            f"  status: {s_mtime.isoformat(timespec='seconds') if s_mtime else 'missing'}"
        )
        print(
            f"  inbox : {m_mtime.isoformat(timespec='seconds') if m_mtime else 'missing'}"
        )
        if status.exists():
            snippet = "\n".join(
                status.read_text(encoding="utf-8", errors="replace").splitlines()[:10]
            )
            print("  --- STATUS snippet ---")
            print(textwrap.indent(snippet, "  "))
        print("")


def cmd_sync(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    ensure_executable("git", hint="Install git")

    bare = root / DIR_PROJECT_BARE
    checkout = root / DIR_PROJECT_CHECKOUT

    print(f"Project: {state['project_name']}")
    print(f"Root: {root}")
    print("")

    if args.show_branches:
        heads = git_list_heads(bare)
        print("== Origin branches (project.git) ==")
        if not heads:
            print("  (no branches yet)")
        else:
            for line in heads:
                parts = line.split("\t", 3)
                if len(parts) == 4:
                    ref, sha, date, subj = parts
                    print(f"  {ref:20} {sha}  {date}  {subj}")
                else:
                    print("  " + line)
        print("")

    if checkout.exists():
        if args.fetch:
            try:
                git_fetch(checkout)
            except Exception as e:
                print(
                    f"warning: fetch failed in integration checkout: {e}",
                    file=sys.stderr,
                )
        print("== Integration checkout (project/) ==")
        if args.pull:
            print(
                "  pull:",
                git_try_pull_ff(checkout, branch=state["git"]["default_branch"]),
            )
        print(textwrap.indent(git_status_short(checkout), "  "))
        print("")
    else:
        print("== Integration checkout (project/) ==\n  (missing)\n")

    if args.all:
        agent_map = {a["name"]: a for a in state["agents"]}
        targets = (
            [p.strip() for p in args.agent.split(",")]
            if args.agent
            else list(agent_map.keys())
        )
        print("== Agent repos ==")
        for name in targets:
            a = agent_map.get(name)
            if not a:
                print(f"  {name}: (unknown)")
                continue
            repo = root / a["repo_dir_rel"]
            if not repo.exists():
                print(f"  {name}: (missing)")
                continue
            if args.fetch:
                try:
                    git_fetch(repo)
                except Exception as e:
                    print(f"  {name}: warning fetch failed: {e}")
            print(f"  -- {name} ({a['title']}) --")
            print(textwrap.indent(git_status_short(repo), "    "))
        print("")


def cmd_seed_sync(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    for a in state["agents"]:
        a_dir = root / a["dir_rel"]
        if args.clean:
            if (a_dir / "seed").exists():
                shutil.rmtree(a_dir / "seed")
            if (a_dir / "seed-extras").exists():
                shutil.rmtree(a_dir / "seed-extras")
        copytree_merge(root / DIR_SEED, a_dir / "seed")
        if a["role"] in ("researcher", "architect", "project_manager"):
            copytree_merge(root / DIR_SEED_EXTRAS, a_dir / "seed-extras")
    print("seed synced")


def cmd_update_workdir(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    ensure_shared_scaffold(root, state)
    install_self_into_root(root)
    ensure_agents_created(root, state)
    sync_oteam_into_agents(root, state)

    updated: List[str] = []
    for a in state["agents"]:
        a_dir = root / a["dir_rel"]
        atomic_write_text(a_dir / "AGENTS.md", render_agent_agents_md(state, a))
        status_path = a_dir / "STATUS.md"
        if not status_path.exists():
            atomic_write_text(status_path, render_agent_status_template(a))

        repo_dir = root / a["repo_dir_rel"]
        try:
            safe_link_dir(Path("..") / "inbox", repo_dir / "inbox")
            safe_link_dir(Path("..") / "outbox", repo_dir / "outbox")
            safe_link_file(Path("..") / "AGENTS.md", repo_dir / "AGENTS.md")
            safe_link_file(Path("..") / "STATUS.md", repo_dir / "STATUS.md")
            safe_link_file(Path("..") / "message.md", repo_dir / "message.md")
            safe_link_dir(Path("..") / ".." / ".." / DIR_SHARED, repo_dir / "shared")
            safe_link_dir(
                Path("..") / ".." / ".." / DIR_SHARED_DRIVE, repo_dir / "shared-drive"
            )
        except Exception:
            pass
        updated.append(a["name"])

    print("updated AGENTS.md for agents: " + ", ".join(updated))


def _print_ticket_cli(t: Dict[str, Any]) -> str:
    blocked_on = t.get("blocked_on") or ""
    status = t.get("status", TICKET_STATUS_OPEN)
    assignee = t.get("assignee") or "unassigned"
    bits = [t.get("id", ""), f"[{status}]", f"@{assignee}", "—", t.get("title", "")]
    if status == TICKET_STATUS_BLOCKED and blocked_on:
        bits.append(f"(blocked on {blocked_on})")
    return " ".join(str(b) for b in bits if b != "")


def cmd_tickets(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    cmd = args.ticket_cmd
    if not cmd:
        raise OTeamError("tickets: missing subcommand")

    if cmd == "list":
        store = load_tickets(root, state)
        tickets = store.get("tickets", [])
        if args.status:
            tickets = [t for t in tickets if t.get("status") == args.status]
        if args.assignee:
            tickets = [
                t
                for t in tickets
                if (t.get("assignee") or "").lower() == args.assignee.lower()
            ]
        if args.json:
            print(json.dumps(tickets, indent=2, sort_keys=True))
            return
        if not tickets:
            print("no tickets match filters")
            return
        for t in tickets:
            print(_print_ticket_cli(t))
        return

    if cmd == "create":
        tags = _parse_tags(args.tags)
        ticket = ticket_create(
            root,
            state,
            title=args.title,
            description=args.desc,
            creator=args.creator or "pm",
            assignee=args.assignee,
            tags=tags,
            assign_note=args.assign_note,
        )
        print(
            f"created ticket {ticket['id']} (assignee: {ticket.get('assignee') or 'unassigned'})"
        )
        return

    if cmd == "assign":
        ticket = ticket_assign(
            root,
            state,
            ticket_id=args.id,
            assignee=args.assignee,
            user=args.user or "pm",
            note=args.note,
        )
        print(f"assigned {ticket['id']} to {ticket.get('assignee') or 'unassigned'}")
        if args.notify:
            _notify_ticket_change(
                root,
                state,
                ticket,
                recipients=[ticket.get("assignee") or ""],
                subject=f"{ticket['id']} assigned",
                body=args.note or "New assignment.",
                nudge=True,
                start_if_needed=True,
            )
        return

    if cmd == "block":
        ticket = ticket_block(
            root,
            state,
            ticket_id=args.id,
            user=args.user or "pm",
            blocked_on=args.on,
            note=args.note,
        )
        print(f"blocked {ticket['id']} on {args.on}")
        recipients = [x for x in [ticket.get("assignee"), "pm"] if x]
        _notify_ticket_change(
            root,
            state,
            ticket,
            recipients=recipients,
            subject=f"{ticket['id']} blocked",
            body=f"Blocked on: {args.on}\nNote: {args.note or ''}",
            nudge=True,
            start_if_needed=False,
        )
        return

    if cmd == "reopen":
        ticket = ticket_reopen(
            root,
            state,
            ticket_id=args.id,
            user=args.user or "pm",
            note=args.note,
        )
        print(f"reopened {ticket['id']}")
        if ticket.get("assignee"):
            _notify_ticket_change(
                root,
                state,
                ticket,
                recipients=[ticket.get("assignee") or ""],
                subject=f"{ticket['id']} reopened",
                body=args.note or "Ticket reopened.",
                nudge=True,
                start_if_needed=False,
            )
        return

    if cmd == "close":
        ticket = ticket_close(
            root,
            state,
            ticket_id=args.id,
            user=args.user or "pm",
            note=args.note,
        )
        print(f"closed {ticket['id']}")
        recipients = list(dict.fromkeys([ticket.get("assignee"), "pm"]))
        _notify_ticket_change(
            root,
            state,
            ticket,
            recipients=recipients,
            subject=f"{ticket['id']} closed",
            body=args.note or "Ticket closed.",
            nudge=False,
            start_if_needed=False,
        )
        return

    if cmd == "show":
        store = load_tickets(root, state)
        ticket = _ticket_required(store, args.id)
        print(_ticket_summary_block(ticket))
        hist = ticket.get("history") or []
        if hist:
            print("")
            print("History:")
            for h in hist:
                h_ts = h.get("ts", "")
                h_type = h.get("type", "")
                note = h.get("note", "")
                from_to = ""
                if "from" in h or "to" in h:
                    from_to = f" (from {h.get('from')} to {h.get('to')})"
                extra = (
                    f" blocked_on={h.get('blocked_on')}" if h_type == "blocked" else ""
                )
                print(f"- {h_ts} {h_type}{from_to}{extra} — {h.get('user', '')} {note}")
        return

    raise OTeamError(f"unknown tickets subcommand: {cmd}")


def cmd_msg(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    body = args.body
    if args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    if not body.strip():
        raise OTeamError("message body is empty (provide --body TEXT or --file)")

    write_message(
        root,
        state,
        sender=args.sender or "pm",
        recipient=args.to,
        subject=args.subject or "",
        body=body,
        msg_type="MESSAGE",
        task=None,
        nudge=not args.no_nudge,
        start_if_needed=args.start_if_needed,
        ticket_id=getattr(args, "ticket", None),
    )
    if args.start_if_needed:
        ensure_tmux_session(root, state)
        ensure_router_window(root, state)
        agent = next((a for a in state["agents"] if a["name"] == args.to), None)
        if agent and not is_opencode_running(state, args.to):
            start_opencode_in_window(root, state, agent, boot=False)

    print(f"sent to {args.to}")


def cmd_broadcast(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    body = args.text
    if args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    if not body.strip():
        raise OTeamError("message body is empty (provide TEXT or --file)")

    sender = args.sender or "pm"
    subject = args.subject or "broadcast"
    recipients = [a["name"] for a in state["agents"]]

    write_broadcast_message(
        root,
        state,
        sender=sender,
        recipients=recipients,
        subject=subject,
        body=body,
        msg_type="MESSAGE",
        nudge=not args.no_nudge,
        start_if_needed=args.start_if_needed,
    )

    if args.start_if_needed:
        ensure_tmux_session(root, state)
        ensure_router_window(root, state)

    print("broadcast sent")


def cmd_assign(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    body = getattr(args, "body", "") or args.text
    if args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    if not body.strip():
        raise OTeamError("assignment body is empty (provide TEXT, --body, or --file)")

    sender = args.sender or "pm"
    ticket: Optional[Dict[str, Any]] = None
    created = False
    ticket_id = (args.ticket or "").strip() if hasattr(args, "ticket") else ""

    if ticket_id:
        store = load_tickets(root, state)
        ticket = _find_ticket(store, ticket_id)
        if not ticket:
            raise OTeamError(f"ticket not found: {ticket_id}")
    else:
        if not args.title or not args.desc:
            raise OTeamError(
                "assignment requires --ticket or --title/--desc to auto-create a ticket"
            )
        ticket = ticket_create(
            root,
            state,
            title=args.title,
            description=args.desc,
            creator=sender,
            assignee=args.to,
            tags=_parse_tags(getattr(args, "tags", None)),
            assign_note=getattr(args, "assign_note", None),
        )
        created = True
        ticket_id = ticket["id"]

    if not created:
        ticket = ticket_assign(
            root,
            state,
            ticket_id=ticket["id"],
            assignee=args.to,
            user=sender,
            note=getattr(args, "assign_note", None) or args.subject,
        )

    ticket_id = ticket["id"]
    summary = _ticket_summary_block(ticket)
    full_body = f"{summary}\n\n{body.strip()}"
    subject = args.subject or f"{ticket_id} — {ticket.get('title', '')}"

    write_message(
        root,
        state,
        sender=sender,
        recipient=args.to,
        subject=subject,
        body=full_body,
        msg_type=ASSIGNMENT_TYPE,
        task=args.task,
        nudge=not args.no_nudge,
        start_if_needed=True,
        ticket_id=ticket_id,
    )
    print(f"assigned {ticket_id} to {args.to}")


def cmd_nudge(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    targets: List[str]
    if args.to == "all":
        targets = [a["name"] for a in state["agents"]]
    else:
        targets = [p.strip() for p in args.to.split(",") if p.strip()]
    for t in targets:
        ok = nudge_agent(
            root, state, t, reason=args.reason or "NUDGE", interrupt=args.interrupt
        )
        print(f"{t}: {'nudged' if ok else 'could not nudge'}")


def cmd_restart(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    if args.window:
        if args.window.strip() == "all":
            targets = [a["name"] for a in state["agents"]]
        else:
            targets = [p.strip() for p in args.window.split(",") if p.strip()]
    else:
        targets = [a["name"] for a in state["agents"]]

    ensure_tmux_session(root, state)
    ensure_router_window(root, state)

    for name in targets:
        agent = next((a for a in state["agents"] if a["name"] == name), None)
        if not agent:
            print(f"warning: unknown agent '{name}', skipping", file=sys.stderr)
            continue
        if args.hard:
            try:
                tmux_send_keys(state["tmux"]["session"], name, ["C-c"])
            except Exception:
                pass
        start_opencode_in_window(root, state, agent, boot=(name == "pm"))
        print(f"restarted opencode in window: {name}")


def notify_pm_new_agent(
    root: Path, state: Dict[str, Any], agent: Dict[str, Any]
) -> None:
    """
    Inform the PM that a new agent was added and remind them to balance workloads.
    """
    if not any(a["name"] == "pm" for a in state["agents"]):
        return
    role = agent.get("role", "agent").replace("_", " ")
    subject = (
        f"New agent added: {agent.get('name', '(unknown)')} ({agent.get('title', '')})"
    )
    lines = [
        f"A new {role} joined: **{agent.get('name', '')}** — {agent.get('title', '')}.",
        "Please balance work across the team: onboard them, delegate tasks, and adjust the plan if needed.",
        "Keep assignments explicit (`Type: ASSIGNMENT`) and update PLAN/tickets to reflect the new capacity.",
    ]
    body = "\n".join(lines)
    try:
        write_message(
            root,
            state,
            sender="oteam",
            recipient="pm",
            subject=subject,
            body=body,
            msg_type="MESSAGE",
            task=None,
            nudge=True,
            start_if_needed=True,
        )
    except Exception:
        pass


def notify_pm_agent_removed(
    root: Path, state: Dict[str, Any], agent: Dict[str, Any], *, purged: bool
) -> None:
    """
    Inform the PM that an agent was removed so roster/tickets stay aligned.
    """
    if not any(a["name"] == "pm" for a in state["agents"]):
        return
    subject = f"Agent removed: {agent.get('name', '(unknown)')}"
    note = (
        "purged from disk"
        if purged
        else "archived under agents/_removed and mail/_removed"
    )
    lines = [
        f"Agent **{agent.get('name', '')}** ({agent.get('title', '') or agent.get('role', 'agent')}) was removed ({note}).",
        "Tickets assigned to this agent were unassigned; reassign or close as needed.",
    ]
    body = "\n".join(lines)
    try:
        write_message(
            root,
            state,
            sender="oteam",
            recipient="pm",
            subject=subject,
            body=body,
            msg_type="MESSAGE",
            task=None,
            nudge=True,
            start_if_needed=True,
        )
    except Exception:
        pass


def add_agent_to_workspace(
    root: Path,
    state: Dict[str, Any],
    *,
    role: str,
    name: Optional[str] = None,
    title: Optional[str] = None,
    persona: Optional[str] = None,
    start_opencode: bool = False,
    no_tmux: bool = False,
) -> Dict[str, Any]:
    """Internal helper to add an agent (used by CLI and router approvals)."""
    if role == "project_manager" or (name and name.strip().lower() == "pm"):
        raise OTeamError(
            "cannot add another project manager; workspace already has a PM"
        )

    existing = {a["name"] for a in state["agents"]}
    if not name:
        if role == "developer":
            i = 1
            while f"dev{i}" in existing:
                i += 1
            name = f"dev{i}"
        elif role == "tester":
            i = 1
            while f"test{i}" in existing:
                i += 1
            name = f"test{i}"
        elif role == "researcher":
            i = 1
            while f"res{i}" in existing:
                i += 1
            name = f"res{i}"
        elif role == "architect":
            i = 1
            while f"arch{i}" in existing:
                i += 1
            name = f"arch{i}"
        else:
            base = (role or "agent")[:4]
            i = 1
            while f"{base}{i}" in existing:
                i += 1
            name = f"{base}{i}"

    if name in existing:
        raise OTeamError(f"agent already exists: {name}")

    title = title or role.replace("_", " ").title()
    agent = {
        "name": name,
        "role": role,
        "title": title,
        "persona": persona or role_persona(role),
        "dir_rel": f"{DIR_AGENTS}/{name}",
        "repo_dir_rel": f"{DIR_AGENTS}/{name}/proj",
    }
    state["agents"].append(agent)
    compute_agent_abspaths(state)
    save_state(root, state)

    ensure_shared_scaffold(root, state)
    create_agent_dirs(root, state, agent)
    update_roster(root, state)
    save_state(root, state)

    if not no_tmux:
        ensure_tmux_session(root, state)
        cmd_args = standby_window_command(root, state, agent)
        tmux_new_window(
            state["tmux"]["session"],
            name,
            Path(agent["dir_abs"]),
            command_args=cmd_args,
        )
        if start_opencode:
            start_opencode_in_window(root, state, agent, boot=False)

    notify_pm_new_agent(root, state, agent)
    return agent


def cmd_add_agent(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    interactive = sys.stdout.isatty() and sys.stdin.isatty()

    def _prompt_choice(prompt: str, options: List[str], default_idx: int = 0) -> str:
        if not options:
            return ""
        for i, opt in enumerate(options, 1):
            print(f"{i}) {opt}")
        raw = input(f"{prompt} [{default_idx + 1}]: ").strip()
        if not raw:
            return options[default_idx]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except Exception:
            pass
        return options[default_idx]

    def _prompt_text(prompt: str, default: str = "") -> str:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw or default

    def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
        suffix = "Y/n" if default else "y/N"
        raw = input(f"{prompt} ({suffix}): ").strip().lower()
        if not raw:
            return default
        return raw in {"y", "yes"}

    role = args.role
    if role == "project_manager" or (args.name and args.name.strip().lower() == "pm"):
        raise OTeamError(
            "cannot add another project manager; workspace already has a PM"
        )
    existing = {a["name"] for a in state["agents"]}

    # Interactive Q/A when running in a TTY.
    if interactive:
        roles = ["developer", "tester", "researcher", "architect"]
        if role not in roles:
            role = "developer"
        role = _prompt_choice("Role", roles, roles.index(role))

        def _default_name_for(role_val: str) -> str:
            if role_val == "developer":
                i = 1
                while f"dev{i}" in existing:
                    i += 1
                return f"dev{i}"
            base = role_val
            if base not in existing:
                return base
            i = 2
            while f"{base}{i}" in existing:
                i += 1
            return f"{base}{i}"

        name_default = args.name or _default_name_for(role)
        name = _prompt_text("Agent name", name_default)
        title_default = args.title or role.replace("_", " ").title()
        title = _prompt_text("Title", title_default)
        persona_default = args.persona or role_persona(role)
        persona = _prompt_text("Persona (optional)", persona_default)

        tmux_running = tmux_has_session(state["tmux"]["session"])
        create_window_default = (not args.no_tmux) and tmux_running
        if not tmux_running:
            print(
                "(tmux session not running; will skip window creation regardless of choice)"
            )
            create_window_default = False
        no_tmux = not _prompt_yes_no("Create tmux window now?", create_window_default)
        start_opencode = _prompt_yes_no(
            "Start opencode immediately?", args.start_opencode
        )
    else:
        name = args.name
        title = args.title
        persona = args.persona
        no_tmux = args.no_tmux
        start_opencode = args.start_opencode

    if not name:
        if role == "developer":
            i = 1
            while f"dev{i}" in existing:
                i += 1
            name = f"dev{i}"
        else:
            base = role
            if base not in existing:
                name = base
            else:
                i = 2
                while f"{base}{i}" in existing:
                    i += 1
                name = f"{base}{i}"

    if any(a["name"] == name for a in state["agents"]):
        raise OTeamError(f"agent already exists: {name}")

    title = title or role.replace("_", " ").title()
    agent = {
        "name": name,
        "role": role,
        "title": title,
        "persona": persona or role_persona(role),
        "dir_rel": f"{DIR_AGENTS}/{name}",
        "repo_dir_rel": f"{DIR_AGENTS}/{name}/proj",
    }
    state["agents"].append(agent)
    compute_agent_abspaths(state)
    save_state(root, state)

    ensure_shared_scaffold(root, state)
    create_agent_dirs(root, state, agent)
    update_roster(root, state)
    save_state(root, state)

    if not no_tmux:
        if tmux_has_session(state["tmux"]["session"]):
            cmd_args = standby_window_command(root, state, agent)
            tmux_new_window(
                state["tmux"]["session"],
                name,
                Path(agent["dir_abs"]),
                command_args=cmd_args,
            )
            if start_opencode:
                start_opencode_in_window(root, state, agent, boot=False)

    notify_pm_new_agent(root, state, agent)
    print(f"added agent: {name} ({title}) role={role}")


def cmd_remove_agent(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    name = args.name.strip()
    if name == "pm":
        raise OTeamError("cannot remove the project manager")

    agent = next((a for a in state["agents"] if a["name"] == name), None)
    if not agent:
        raise OTeamError(f"agent not found: {name}")

    # Best-effort cleanup before state mutation; do not abort on errors.
    try:
        if tmux_has_session(state["tmux"]["session"]):
            _kill_agent_window_if_present(state, name)
    except Exception:
        pass
    try:
        _archive_or_remove_agent_dirs(root, agent, purge=args.purge)
    except Exception:
        pass
    try:
        _unassign_tickets_for_agent(root, state, name)
    except Exception:
        pass
    _nudge_history.pop(name, None)

    state["agents"] = [a for a in state["agents"] if a["name"] != name]
    compute_agent_abspaths(state)
    update_roster(root, state)
    save_state(root, state)

    try:
        notify_pm_agent_removed(root, state, agent, purged=args.purge)
    except Exception:
        pass

    print(f"removed agent: {name}")


def cmd_doc_walk(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise OTeamError("could not find oteam.json in this directory or its parents")
    state = load_state(root)

    write_message(
        root,
        state,
        sender=args.sender or "pm",
        recipient="pm",
        subject=args.subject or "Doc-walk kickoff",
        body=(
            "Please run a documentation sprint over the existing code:\n"
            "- Fill shared/GOALS.md (infer goals)\n"
            "- Build a doc plan in shared/PLAN.md (high-level milestones only)\n"
            "- Assign module ownership to devs for documentation\n"
            "- Ensure we have: how-to-run, how-to-test, architecture overview, ADRs/decisions\n"
        ),
        msg_type=ASSIGNMENT_TYPE,
        task=args.task or "DOC-WALK",
        nudge=True,
        start_if_needed=True,
    )

    if args.auto:
        for who, subj, task, body in [
            (
                "architect",
                "Doc-walk: architecture doc",
                "DOC-ARCH",
                "Create docs/architecture.md and a decisions summary. Coordinate with PM.",
            ),
            (
                "tester",
                "Doc-walk: run/test guide",
                "DOC-TEST",
                "Create docs/runbook.md with how to run and test. Coordinate with PM.",
            ),
            (
                "researcher",
                "Doc-walk: dependencies inventory",
                "DOC-DEPS",
                "Create shared/research/deps.md with dependencies and risks. Coordinate with PM.",
            ),
        ]:
            if any(a["name"] == who for a in state["agents"]):
                write_message(
                    root,
                    state,
                    sender=args.sender or "pm",
                    recipient=who,
                    subject=subj,
                    body=body,
                    msg_type=ASSIGNMENT_TYPE,
                    task=task,
                    nudge=True,
                    start_if_needed=True,
                )

    print("doc-walk kickoff sent (PM-led)")


# -----------------------------
# Telegram commands
# -----------------------------


def cmd_telegram_configure(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    if not (root / STATE_FILENAME).exists():
        raise OTeamError(
            "telegram-configure: could not find oteam.json in this directory or its parents"
        )
    state = load_state(root)

    history_path = root / DIR_RUNTIME / "telegram_configure.history"
    if sys.stdin.isatty() and not args.no_interactive:
        configure_readline(history_path)

    token = (args.token or "").strip()
    phone = (args.phone or "").strip()

    if sys.stdin.isatty() and not args.no_interactive:
        if not token:
            token = prompt_line("Telegram bot token (from @BotFather)", default="")
        if not phone:
            phone = prompt_line(
                "Authorized phone number (digits; E.164 preferred, e.g. +15551234567)",
                default="",
            )

    token = token.strip()
    phone = phone.strip()
    if not token or not phone:
        raise OTeamError(
            "telegram-configure: need --token and --phone (or run interactively)"
        )

    cfg = {
        "configured_at": now_iso(),
        "bot_token": token,
        "authorized_phone": phone,
        "chat_id": None,
        "user_id": None,
        "update_offset": 0,
    }
    telegram_save_config(root, state, cfg)

    print(f"Wrote Telegram config: {telegram_config_path(root, state)}")
    print("")
    print("Next steps:")
    print("1) In Telegram, open your bot chat and send /start.")
    print("2) Tap 'Share phone number' to authorize (must match the configured phone).")
    print("3) Enable integration:")
    print(f"   python3 oteam.py {shlex.quote(str(root))} telegram-enable")


def cmd_telegram_enable(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    if not (root / STATE_FILENAME).exists():
        raise OTeamError(
            "telegram-enable: could not find oteam.json in this directory or its parents"
        )
    state = load_state(root)

    _ = telegram_load_config(root, state)  # validate

    state.setdefault("telegram", {})
    state["telegram"]["enabled"] = True

    if not state.get("tmux", {}).get("router", True):
        state["tmux"]["router"] = True

    save_state(root, state)

    print("Telegram integration enabled.")
    print(
        "If the router is running (`cteam <workdir> watch`), it will start bridging immediately."
    )
    print("If not, open/start the workspace (tmux router runs the watcher):")
    print(f"  python3 oteam.py {shlex.quote(str(root))} open")


def cmd_telegram_disable(args: argparse.Namespace) -> None:
    root = (
        find_project_root(Path(args.workdir))
        or Path(args.workdir).expanduser().resolve()
    )
    if not (root / STATE_FILENAME).exists():
        raise OTeamError(
            "telegram-disable: could not find oteam.json in this directory or its parents"
        )
    state = load_state(root)

    state.setdefault("telegram", {})
    state["telegram"]["enabled"] = False
    save_state(root, state)
    print("Telegram integration disabled.")


# -----------------------------
# CLI
# -----------------------------

COMMAND_NAMES = [
    "init",
    "import",
    "resume",
    "open",
    "attach",
    "kill",
    "watch",
    "pause",
    "chat",
    "upload",
    "customer-chat",
    "status",
    "sync",
    "seed-sync",
    "update-workdir",
    "tickets",
    "msg",
    "broadcast",
    "assign",
    "nudge",
    "restart",
    "add-agent",
    "remove-agent",
    "doc-walk",
    "telegram-configure",
    "telegram-enable",
    "telegram-disable",
    "dashboard",
    "capture",
    "request-agent",
    "stage",
    "autodispatch",
]


def rewrite_argv_with_default_workdir(argv: List[str]) -> List[str]:
    """
    Convenience: if the user runs `oteam <command>` from a workspace dir,
    assume workdir="." when the first argument looks like a command and no
    explicit workdir is provided.
    """
    if not argv:
        return argv
    first = argv[0]
    rest = argv[1:]
    if first in COMMAND_NAMES:
        # Only rewrite when the next token looks like a flag (or nothing) to avoid
        # breaking `<workdir> <cmd>` when the workdir happens to match a command name.
        if not rest or rest[0].startswith("-"):
            return [".", *argv]
    return argv


class CTeamHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Readable help with defaults and preserved newlines."""


class CTeamArgumentParser(argparse.ArgumentParser):
    """
    Custom parser that prints full help on usage errors for a friendlier UX.
    """

    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\noteam error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = CTeamArgumentParser(
        prog="oteam",
        add_help=True,
        description=(
            "OpenCode Team (oteam) orchestrates tmux + OpenCode agents around a shared git repo.\n"
            "Common flow: init/import a workspace, open tmux, assign work, chat with agents, and keep the customer updated."
        ),
        epilog=(
            "Popular: init/import/open/resume, msg/assign/broadcast/nudge, watch, add-agent/remove-agent, customer-chat.\n"
            "Git/ops: sync, status, seed-sync, update-workdir, restart, doc-walk, telegram-*. "
            "Run `oteam <workdir> <command> --help` for details. Full guide: README.md."
        ),
        formatter_class=CTeamHelpFormatter,
    )
    p.add_argument("workdir", help="Workspace directory or any path inside it.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_workspace(pp: argparse.ArgumentParser) -> None:
        pp.add_argument(
            "--no-tmux", action="store_true", help="Do not start/manage tmux."
        )
        pp.add_argument(
            "--no-opencode",
            action="store_true",
            help="Do not launch OpenCode in windows.",
        )
        pp.add_argument(
            "--no-attach",
            action="store_true",
            help="Do not attach to tmux after starting.",
        )
        pp.add_argument(
            "--attach-late",
            action="store_true",
            help="Attach after tmux/windows/OpenCode are ready.",
        )
        pp.add_argument("--window", help="Window name to attach/select (e.g., pm).")

    def add_opencode_flags(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--model", default=None, help="OpenCode model (if supported).")
        pp.add_argument(
            "--sandbox",
            default=DEFAULT_OPENCODE_SANDBOX,
            choices=["read-only", "workspace-write", "danger-full-access"],
        )
        pp.add_argument(
            "--ask-for-approval",
            default=DEFAULT_OPENCODE_APPROVAL,
            choices=["untrusted", "on-failure", "on-request", "never"],
        )
        pp.add_argument(
            "--no-search",
            action="store_true",
            help="Disable OpenCode --search (if supported).",
        )
        pp.add_argument(
            "--full-auto",
            action="store_true",
            help="Use OpenCode --full-auto (if supported).",
        )
        pp.add_argument(
            "--yolo",
            action="store_true",
            help="Use OpenCode --yolo (if supported). DANGEROUS.",
        )

    def add_coord_flags(pp: argparse.ArgumentParser) -> None:
        pp.add_argument(
            "--autostart",
            default=DEFAULT_AUTOSTART,
            choices=["pm", "pm+architect", "all"],
            help="Which agents start OpenCode immediately (PM-led coordination).",
        )
        pp.add_argument(
            "--no-router", action="store_true", help="Disable router/watch window."
        )

    p_init = sub.add_parser(
        "init", help="Initialize a new oteam workspace (empty repo)."
    )
    add_common_workspace(p_init)
    p_init.add_argument(
        "--no-interactive", action="store_true", help="Disable interactive init wizard."
    )
    p_init.add_argument("--name", help="Project name (default: folder name).")
    p_init.add_argument(
        "--devs", type=int, default=1, help="Number of developer agents."
    )
    p_init.add_argument(
        "--force", action="store_true", help="Reuse non-empty directory."
    )
    add_opencode_flags(p_init)
    add_coord_flags(p_init)
    p_init.set_defaults(func=cmd_init)

    p_imp = sub.add_parser(
        "import",
        help="Import an existing git repo, directory, or tarball into a new workspace.",
    )
    add_common_workspace(p_imp)
    p_imp.add_argument(
        "--src",
        required=True,
        help="Git repo URL/path, local directory, or tarball to import.",
    )
    p_imp.add_argument("--name", help="Project name override.")
    p_imp.add_argument("--devs", type=int, default=1)
    p_imp.add_argument("--force", action="store_true")
    p_imp.add_argument(
        "--recon",
        action="store_true",
        help="Also send safe coordinated recon assignments (no code changes) to non-PM agents.",
    )
    p_imp.add_argument(
        "--seed",
        help="Path to SEED.md file to use instead of auto-generated seed from import source.",
    )
    add_opencode_flags(p_imp)
    add_coord_flags(p_imp)
    p_imp.set_defaults(func=cmd_import)

    p_resume = sub.add_parser(
        "resume",
        help="Resume/manage an existing workspace (ensure dirs/windows exist).",
    )
    add_common_workspace(p_resume)
    p_resume.add_argument(
        "--autostart",
        choices=["pm", "pm+architect", "all"],
        help="Override autostart in state.",
    )
    p_resume.add_argument("--router", action="store_true", help="Force-enable router.")
    p_resume.add_argument("--no-router", action="store_true", help="Disable router.")
    p_resume.set_defaults(func=cmd_resume)

    p_open = sub.add_parser(
        "open", help="Resume and attach (open tmux in your terminal)."
    )
    add_common_workspace(p_open)
    p_open.add_argument("--no-router", action="store_true")
    p_open.set_defaults(func=cmd_open)

    p_attach = sub.add_parser("attach", help="Attach to tmux session only.")
    add_common_workspace(p_attach)
    p_attach.set_defaults(func=cmd_attach)

    p_kill = sub.add_parser("kill", help="Kill the tmux session.")
    add_common_workspace(p_kill)
    p_kill.set_defaults(func=cmd_kill)

    p_watch = sub.add_parser(
        "watch", help="Router: watch mailboxes and nudge/start agents in tmux."
    )
    p_watch.add_argument("--interval", type=float, default=1.5)
    p_watch.set_defaults(func=cmd_watch)

    p_pause = sub.add_parser(
        "pause", help="Pause tmux: park agent windows and mark workspace paused."
    )
    p_pause.set_defaults(func=cmd_pause)

    p_chat = sub.add_parser("chat", help="Interactive chat with an agent or customer.")
    p_chat.add_argument(
        "--to", required=True, help="Recipient agent name or 'customer'."
    )
    p_chat.add_argument("--sender", help="Override sender name (default: human).")
    p_chat.add_argument("--subject", help="Optional subject.")
    p_chat.add_argument("--no-nudge", action="store_true")
    p_chat.add_argument("--start-if-needed", action="store_true")
    p_chat.set_defaults(func=cmd_chat)

    p_upload = sub.add_parser(
        "upload", help="Copy files/dirs into shared/drive for sharing with the team."
    )
    p_upload.add_argument(
        "paths", nargs="+", help="Files or directories to copy into shared/drive."
    )
    p_upload.add_argument(
        "--dest",
        help="Optional destination name/path under shared/drive (default: use source name).",
    )
    p_upload.add_argument(
        "--from",
        dest="sender",
        help="Sender name for PM notification (default: human).",
    )
    p_upload.set_defaults(func=cmd_upload)

    p_cust = sub.add_parser(
        "customer-chat", help="Run an interactive customer chat window (tmux-friendly)."
    )
    p_cust.set_defaults(func=cmd_customer_chat)

    p_status = sub.add_parser("status", help="Show workspace + agent status.")
    p_status.set_defaults(func=cmd_status)

    p_dashboard = sub.add_parser(
        "dashboard", help="Show the live status board (written by the router)."
    )
    p_dashboard.add_argument("--json", action="store_true", help="print JSON")
    p_dashboard.add_argument(
        "--fresh",
        action="store_true",
        help="compute a best-effort snapshot (ignores router cache)",
    )
    p_dashboard.set_defaults(func=cmd_dashboard)

    p_capture = sub.add_parser(
        "capture", help="Capture the last N lines of an agent's tmux pane."
    )
    p_capture.add_argument("agent", help="tmux window name (agent)")
    p_capture.add_argument(
        "--lines", type=int, default=200, help="number of lines to capture"
    )
    p_capture.add_argument(
        "--to", help="optional recipient to send capture to (pm|customer|agent)"
    )
    p_capture.set_defaults(func=cmd_capture)

    p_request_agent = sub.add_parser(
        "request-agent",
        help="Request spawning a new agent (customer must approve via /approve TOKEN).",
    )
    p_request_agent.add_argument(
        "--role", required=True, choices=AGENT_ROLES, help="agent role"
    )
    p_request_agent.add_argument("--name", help="optional explicit agent name")
    p_request_agent.add_argument("--title", help="optional title label")
    p_request_agent.add_argument("--persona", help="optional persona override")
    p_request_agent.add_argument(
        "--start-opencode",
        dest="start_opencode",
        action="store_true",
        default=True,
        help="start opencode immediately upon approval (default)",
    )
    p_request_agent.add_argument(
        "--no-start-opencode",
        dest="start_opencode",
        action="store_false",
        help="do NOT start opencode automatically",
    )
    p_request_agent.add_argument(
        "--reason", required=True, help="why you need this new agent"
    )
    p_request_agent.set_defaults(func=cmd_request_agent)

    p_stage = sub.add_parser(
        "stage", help="Set the coordination stage (discovery|planning|execution)."
    )
    p_stage.add_argument("stage", choices=["discovery", "planning", "execution"])
    p_stage.set_defaults(func=cmd_stage)

    p_autod = sub.add_parser(
        "autodispatch", help="Toggle router auto-dispatch (on/off)."
    )
    p_autod.add_argument("mode", choices=["on", "off"])
    p_autod.set_defaults(func=cmd_autodispatch)
    p_sync = sub.add_parser("sync", help="Fetch/pull and show git statuses.")
    p_sync.add_argument(
        "--all", action="store_true", help="Also show agent repo statuses."
    )
    p_sync.add_argument("--agent", help="Limit to specific agent(s) with --all.")
    p_sync.add_argument("--fetch", action="store_true")
    p_sync.add_argument("--pull", action="store_true")
    p_sync.add_argument(
        "--no-show-branches", action="store_false", dest="show_branches"
    )
    p_sync.set_defaults(show_branches=True)
    p_sync.set_defaults(func=cmd_sync)

    p_seed = sub.add_parser(
        "seed-sync", help="Copy seed/ and seed-extras/ into agent workdirs."
    )
    p_seed.add_argument("--clean", action="store_true")
    p_seed.set_defaults(func=cmd_seed_sync)

    p_update = sub.add_parser(
        "update-workdir",
        help="Refresh agent AGENTS.md files from current oteam templates.",
    )
    p_update.set_defaults(func=cmd_update_workdir)

    p_tickets = sub.add_parser(
        "tickets",
        help="Ticket management (list/create/assign/block/reopen/close/show).",
    )
    tickets_sub = p_tickets.add_subparsers(dest="ticket_cmd")

    p_t_list = tickets_sub.add_parser("list", help="List tickets.")
    p_t_list.add_argument(
        "--status",
        choices=[TICKET_STATUS_OPEN, TICKET_STATUS_BLOCKED, TICKET_STATUS_CLOSED],
    )
    p_t_list.add_argument("--assignee")
    p_t_list.add_argument("--json", action="store_true")
    p_t_list.set_defaults(func=cmd_tickets)

    p_t_show = tickets_sub.add_parser("show", help="Show one ticket with history.")
    p_t_show.add_argument("--id", required=True)
    p_t_show.set_defaults(func=cmd_tickets)

    p_t_create = tickets_sub.add_parser("create", help="Create a ticket.")
    p_t_create.add_argument("--title", required=True)
    p_t_create.add_argument("--desc", required=True)
    p_t_create.add_argument("--assignee")
    p_t_create.add_argument("--creator", help="Ticket creator (default: pm)")
    p_t_create.add_argument("--tags", help="Comma-separated tags")
    p_t_create.add_argument(
        "--assign-note", help="Optional note for initial assignment"
    )
    p_t_create.set_defaults(func=cmd_tickets)

    p_t_assign = tickets_sub.add_parser("assign", help="Assign an existing ticket.")
    p_t_assign.add_argument("--id", required=True)
    p_t_assign.add_argument("--assignee", required=True)
    p_t_assign.add_argument(
        "--user", help="User performing the assignment (default: pm)"
    )
    p_t_assign.add_argument("--note")
    p_t_assign.add_argument(
        "--notify", action="store_true", help="Also notify the assignee."
    )
    p_t_assign.set_defaults(func=cmd_tickets)

    p_t_block = tickets_sub.add_parser("block", help="Mark a ticket as blocked.")
    p_t_block.add_argument("--id", required=True)
    p_t_block.add_argument(
        "--on", required=True, help="Ticket ID or external issue causing the block."
    )
    p_t_block.add_argument("--user", help="User performing the change (default: pm)")
    p_t_block.add_argument("--note")
    p_t_block.set_defaults(func=cmd_tickets)

    p_t_reopen = tickets_sub.add_parser("reopen", help="Reopen a ticket.")
    p_t_reopen.add_argument("--id", required=True)
    p_t_reopen.add_argument("--user", help="User performing the change (default: pm)")
    p_t_reopen.add_argument("--note")
    p_t_reopen.set_defaults(func=cmd_tickets)

    p_t_close = tickets_sub.add_parser("close", help="Close a ticket.")
    p_t_close.add_argument("--id", required=True)
    p_t_close.add_argument("--user", help="User performing the change (default: pm)")
    p_t_close.add_argument("--note")
    p_t_close.set_defaults(func=cmd_tickets)

    p_msg = sub.add_parser("msg", help="Send a message to one agent.")
    p_msg.add_argument("--to", required=True)
    p_msg.add_argument("--from", dest="sender")
    p_msg.add_argument("--subject")
    p_msg.add_argument("--body", help="Message body text.")
    p_msg.add_argument("--file")
    p_msg.add_argument("--no-nudge", action="store_true")
    p_msg.add_argument(
        "--start-if-needed",
        action="store_true",
        help="If recipient is not running OpenCode, start it in their tmux window.",
    )
    p_msg.add_argument(
        "--no-follow",
        action="store_true",
        help="Do not select recipient window after sending.",
    )
    p_msg.add_argument("--ticket", help="Ticket ID to link this message.")
    p_msg.set_defaults(func=cmd_msg)

    p_b = sub.add_parser("broadcast", help="Broadcast a message to all agents.")
    p_b.add_argument("--from", dest="sender")
    p_b.add_argument("--subject")
    p_b.add_argument("--file")
    p_b.add_argument("--no-nudge", action="store_true")
    p_b.add_argument(
        "--start-if-needed",
        action="store_true",
        help="Start OpenCode in agent windows if needed.",
    )
    p_b.add_argument("text", nargs="?", default="")
    p_b.set_defaults(func=cmd_broadcast)

    p_assign = sub.add_parser(
        "assign", help="Send an ASSIGNMENT to an agent (starts them if needed)."
    )
    p_assign.add_argument("--to", required=True)
    p_assign.add_argument("--task", help="Task ID (e.g., T001).")
    p_assign.add_argument("--from", dest="sender")
    p_assign.add_argument("--subject")
    p_assign.add_argument(
        "--body", help="Assignment body text (alternative to positional TEXT)."
    )
    p_assign.add_argument("--file")
    p_assign.add_argument("--no-nudge", action="store_true")
    p_assign.add_argument(
        "--ticket", help="Existing ticket ID; if omitted, requires --title and --desc."
    )
    p_assign.add_argument("--title", help="Title when auto-creating a ticket.")
    p_assign.add_argument("--desc", help="Description when auto-creating a ticket.")
    p_assign.add_argument("--assign-note", help="Note to record on assignment.")
    p_assign.add_argument(
        "--tags", help="Comma-separated tags when auto-creating a ticket."
    )
    p_assign.add_argument("text", nargs="?", default="")
    p_assign.add_argument(
        "--no-follow",
        action="store_true",
        help="Do not select recipient window after sending.",
    )
    p_assign.set_defaults(func=cmd_assign)

    p_nudge = sub.add_parser("nudge", help="Send a manual nudge to an agent window.")
    p_nudge.add_argument("--to", default="pm", help="agent(s) comma-separated or 'all'")
    p_nudge.add_argument("--reason", default="NUDGE")
    p_nudge.add_argument(
        "--no-follow", action="store_true", help="Do not select the target window."
    )
    p_nudge.add_argument(
        "--interrupt",
        action="store_true",
        help="Send an interrupting nudge (sends Escape before the message).",
    )
    p_nudge.set_defaults(func=cmd_nudge)

    p_restart = sub.add_parser(
        "restart", help="Restart OpenCode in agent tmux windows (respawn)."
    )
    p_restart.add_argument(
        "--window", help="agent window name(s) comma-separated or 'all'"
    )
    p_restart.add_argument(
        "--hard", action="store_true", help="send Ctrl-C before respawn (best-effort)"
    )
    p_restart.set_defaults(func=cmd_restart)

    p_add = sub.add_parser(
        "add-agent", help="Add a new agent to an existing workspace (one PM only)."
    )
    p_add.add_argument(
        "--role",
        default="developer",
        choices=["developer", "tester", "researcher", "architect"],
    )
    p_add.add_argument("--name")
    p_add.add_argument("--title")
    p_add.add_argument("--persona")
    p_add.add_argument("--no-tmux", action="store_true")
    p_add.add_argument(
        "--start-opencode",
        action="store_true",
        help="Start opencode immediately for the new agent.",
    )
    p_add.set_defaults(func=cmd_add_agent)

    p_rm = sub.add_parser(
        "remove-agent", help="Remove an agent from the workspace (non-PM only)."
    )
    p_rm.add_argument("--name", required=True, help="Agent name to remove (not pm).")
    p_rm.add_argument(
        "--purge",
        action="store_true",
        help="Delete agent dirs/mail instead of archiving under _removed/.",
    )
    p_rm.set_defaults(func=cmd_remove_agent)

    p_doc = sub.add_parser(
        "doc-walk", help="Kick off a documentation sprint over the repo (PM-led)."
    )
    p_doc.add_argument("--from", dest="sender")
    p_doc.add_argument("--subject")
    p_doc.add_argument("--task")
    p_doc.add_argument(
        "--auto",
        action="store_true",
        help="Also assign initial doc tasks to architect/tester/researcher.",
    )
    p_doc.set_defaults(func=cmd_doc_walk)

    p_tg_cfg = sub.add_parser(
        "telegram-configure",
        help="Configure Telegram bot credentials for customer chat.",
    )
    p_tg_cfg.add_argument("--token", help="Bot token from @BotFather (stored locally).")
    p_tg_cfg.add_argument(
        "--phone", help="Authorized phone number (digits; E.164 preferred)."
    )
    p_tg_cfg.add_argument(
        "--no-interactive", action="store_true", help="Do not prompt (require flags)."
    )
    p_tg_cfg.set_defaults(func=cmd_telegram_configure)

    p_tg_en = sub.add_parser(
        "telegram-enable",
        help="Enable Telegram customer chat bridge (requires prior configure).",
    )
    p_tg_en.set_defaults(func=cmd_telegram_enable)

    p_tg_dis = sub.add_parser(
        "telegram-disable", help="Disable Telegram customer chat bridge."
    )
    p_tg_dis.set_defaults(func=cmd_telegram_disable)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    argv = rewrite_argv_with_default_workdir(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
        return 0
    except OTeamError as e:
        print(f"oteam error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
