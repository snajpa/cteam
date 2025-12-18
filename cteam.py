#!/usr/bin/env python3
"""
Clanker Team (cteam) — single-file multi-agent manager for tmux + OpenAI Codex CLI + git repos.

Goal
- Run multiple Codex "agents" in a tmux session to collaboratively develop a project.
- Support both:
  - starting from an empty repo (init)
  - importing an existing repo/directory and documenting it (import + doc-walk)

Key design choices (PM-led coordination)
- Default mode is **PM-led**:
  - All agent tmux windows start Codex immediately so they are ready to act.
  - Agents still wait for an **ASSIGNMENT** before implementing work.
  - When the PM (or a human) sends an **ASSIGNMENT** message to an agent, cteam:
      1) ensures Codex is running in that agent's tmux window
      2) nudges them to read `message.md` and execute the task
- This avoids the "everyone does random work without coordination" failure mode while keeping agents live.

Communication
- Canonical mailbox per agent: shared/mail/<agent>/message.md
- `cteam msg` and `cteam assign` append to that mailbox and also drop a copy in:
    shared/mail/<agent>/inbox/<timestamp>_<sender>.md
- A router/watch loop (`cteam <workdir> watch`) runs inside tmux in a `router` window:
  - watches inbox directories and mailbox changes
  - nudges recipients
  - can auto-start Codex for agents when assignments arrive

Major commands
- init, import, resume, open, attach, kill
- msg, broadcast, assign, nudge
- watch (router)
- status, sync, seed-sync
- add-agent, remove-agent, restart
- doc-walk (kick off documentation sprint)

External deps
- tmux, git, codex (Codex CLI). Everything else is Python stdlib.

Security note
- The defaults minimize Codex approval prompts by using:
    --ask-for-approval never
  and a permissive sandbox default:
    --sandbox danger-full-access
  Adjust via CLI flags if you want more safety. When running Codex directly, set
  `approval_policy=never` (or `--ask-for-approval never`) for the expected
  Clanker Team experience.

"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
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


# -----------------------------
# Constants / layout
# -----------------------------

STATE_FILENAME = "cteam.json"
STATE_VERSION = 7

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

DEFAULT_CODEX_CMD = "codex"
DEFAULT_CODEX_SANDBOX = "danger-full-access"   # read-only | workspace-write | danger-full-access
DEFAULT_CODEX_APPROVAL = "never"               # untrusted | on-failure | on-request | never
DEFAULT_CODEX_SEARCH = True

DEFAULT_AUTOSTART = "all"  # pm | pm+architect | all


TELEGRAM_CONFIG_REL = f"{DIR_RUNTIME}/telegram.json"
TELEGRAM_MAX_MESSAGE = 3900  # Telegram hard limit is 4096; keep margin.

# For robust "assignment" detection
ASSIGNMENT_TYPE = "ASSIGNMENT"

TICKET_STATUS_OPEN = "open"
TICKET_STATUS_BLOCKED = "blocked"
TICKET_STATUS_CLOSED = "closed"
TICKET_STATUSES = {TICKET_STATUS_OPEN, TICKET_STATUS_BLOCKED, TICKET_STATUS_CLOSED}

TICKET_MIGRATION_FLAG = f"{DIR_RUNTIME}/tickets_migration_notified"

_nudge_history: Dict[str, Tuple[float, Set[str]]] = {}

class CTeamError(RuntimeError):
    pass


# -----------------------------
# Small utilities
# -----------------------------

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
        atomic_write_text(customer_state_path(root), json.dumps(state, indent=2, sort_keys=True))
    except Exception:
        pass


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "project"


def _looks_like_git_url(src: str) -> bool:
    s = src.strip()
    return bool(re.match(r"^(https?://|ssh://|git@|file://|.+\\.git$)", s))


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


# -----------------------------
# Tickets
# -----------------------------


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
    atomic_write_text(tickets_json_path(root, state), json.dumps(store, indent=2, sort_keys=True) + "\n")


def _next_ticket_id(meta: Dict[str, Any]) -> str:
    try:
        n = int(meta.get("next_id", 1))
    except Exception:
        n = 1
    tid = f"T{n:03d}"
    meta["next_id"] = n + 1
    return tid


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
    active = [t for t in tickets if t.get("status") in {TICKET_STATUS_OPEN, TICKET_STATUS_BLOCKED}]
    if not active:
        return "No open or blocked tickets."
    lines = []
    for t in sorted(active, key=lambda x: x.get("id","")):
        tid = t.get("id", "")
        status = t.get("status", TICKET_STATUS_OPEN)
        assignee = t.get("assignee") or "unassigned"
        title = t.get("title") or ""
        block_note = ""
        if status == TICKET_STATUS_BLOCKED and t.get("blocked_on"):
            block_note = f" (blocked on {t.get('blocked_on')})"
        lines.append(f"- {tid} [{status}] @{assignee} — {title}{block_note}")
    return "\n".join(lines)


def _ticket_required(store: Dict[str, Any], ticket_id: str) -> Dict[str, Any]:
    t = _find_ticket(store, ticket_id)
    if not t:
        raise CTeamError(f"ticket not found: {ticket_id}")
    return t


def _ticket_summary_block(ticket: Dict[str, Any]) -> str:
    assignee = ticket.get("assignee") or "unassigned"
    status = ticket.get("status", TICKET_STATUS_OPEN)
    blocked_on = ticket.get("blocked_on") or ""
    lines = [
        f"Ticket: {ticket.get('id','')} — {ticket.get('title','')}",
        f"Status: {status}" + (f" (blocked on {blocked_on})" if status == TICKET_STATUS_BLOCKED and blocked_on else ""),
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
    return list(dict.fromkeys(tags))  # dedupe preserving order


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
        tid = _next_ticket_id(store["meta"])
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
        _add_history(ticket, {"ts": ts, "type": "created", "user": creator, "note": title.strip()})
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
            raise CTeamError(f"ticket {ticket_id} is closed; reopen before assigning")
        prev = t.get("assignee")
        t["assignee"] = assignee
        t["updated_at"] = now_iso()
        _add_history(t, {"ts": t["updated_at"], "type": "assigned", "user": user, "from": prev, "to": assignee, "note": (note or "").strip()})
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
        _add_history(t, {"ts": t["updated_at"], "type": "blocked", "user": user, "blocked_on": blocked_on, "note": (note or '').strip()})
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
        _add_history(t, {"ts": t["updated_at"], "type": "reopened", "user": user, "note": (note or "").strip()})
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
        _add_history(t, {"ts": t["updated_at"], "type": "closed", "user": user, "note": (note or "").strip()})
        save_tickets(root, state, store)
        return t


# -----------------------------
# Readline helpers (interactive UX)
# -----------------------------

def configure_readline(history_path: Optional[Path] = None) -> None:
    """Best-effort readline configuration + persistent history."""
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


def prompt_line(prompt: str, *, default: Optional[str] = None, secret: bool = False) -> str:
    show = f" [{default}]" if default else ""
    while True:
        try:
            if secret:
                import getpass  # type: ignore
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
    """Capture multiline input. Finish with a line containing only `terminator` or Ctrl-D."""
    print(header.rstrip())
    print(f"(paste multiple lines; finish with a line containing only '{terminator}', or press Ctrl-D)")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            print("")  # newline after Ctrl-D
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
        raise CTeamError(msg)


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
        raise CTeamError(f"missing executable: {cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "").strip()
        err = (e.stderr or "").strip()
        details = "\n".join(x for x in [out, err] if x)
        raise CTeamError(
            f"command failed: {shlex.join(cmd)}" + (f"\n{details}" if details else "")
        ) from e


def pick_shell() -> List[str]:
    """
    Return a shell launcher as argv prefix, e.g. ["bash", "-lc"].
    """
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
    """
    Copy this running cteam.py into root/cteam.py so agents can call it consistently.
    """
    try:
        src = Path(__file__).resolve()
    except Exception:
        return
    dst = root / "cteam.py"
    try:
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            return
        shutil.copy2(src, dst)
    except Exception:
        # non-fatal
        pass


def samefile(a: Path, b: Path) -> bool:
    try:
        if not a.exists() or not b.exists():
            return False
        return os.path.samefile(a, b)
    except Exception:
        return False


def safe_link_file(target: Path, link: Path) -> None:
    """
    Try: symlink -> hardlink -> copy, replacing existing link/path.
    """
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
        abs_target = target if target.is_absolute() else (link.parent / target).resolve()
        if abs_target.exists() and abs_target.is_file():
            os.link(str(abs_target), str(link))
            return
    except OSError:
        pass

    abs_target = target if target.is_absolute() else (link.parent / target).resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(abs_target, link)


def safe_link_dir(target: Path, link: Path) -> None:
    """
    Try: symlink. If not possible, create a local pointer file.
    """
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)

    try:
        os.symlink(str(target), str(link))
        return
    except OSError:
        link.mkdir(parents=True, exist_ok=True)
        abs_target = target if target.is_absolute() else (link.parent / target).resolve()
        atomic_write_text(
            link / "README_cteam_pointer.md",
            textwrap.dedent(
                f"""\
                # cteam pointer

                This directory is a placeholder because the filesystem disallowed a symlink.

                Canonical directory:
                `{abs_target}`

                Please use the canonical path above.
                """
            ),
        )


def copytree_merge(src: Path, dst: Path) -> None:
    """Copy src into dst (merge; does not delete existing). Skips .git."""
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        # skip nested .git
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


def sync_cteam_into_agents(root: Path, state: Dict[str, Any]) -> None:
    """
    Ensure the current cteam.py is present in every agent workspace and repo clone.
    Uses safe_link_file to allow symlink/hardlink/copy based on FS support.
    """
    cteam_path = root / "cteam.py"
    if not cteam_path.exists():
        return
    for agent in state.get("agents", []):
        a_dir = root / agent["dir_rel"]
        repo_dir = root / agent["repo_dir_rel"]
        try:
            mkdirp(a_dir)
            mkdirp(repo_dir)
            safe_link_file(cteam_path, a_dir / "cteam.py")
            safe_link_file(cteam_path, repo_dir / "cteam.py")
        except Exception:
            # Best-effort; skip failures to avoid breaking the rest of update-workdir.
            pass


# -----------------------------
# Git helpers
# -----------------------------

def git_init_bare(path: Path, branch: str = DEFAULT_GIT_BRANCH) -> None:
    mkdirp(path)
    try:
        run_cmd(["git", "init", "--bare", f"--initial-branch={branch}"], cwd=path)
    except CTeamError:
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
    except CTeamError as e:
        return f"(git status failed: {e})"


def git_try_pull_ff(repo: Path, branch: str = DEFAULT_GIT_BRANCH) -> str:
    try:
        git_fetch(repo)
        cp = run_cmd(["git", "-C", str(repo), "pull", "--ff-only", "origin", branch], capture=True)
        out = (cp.stdout or "").strip()
        return out or f"pulled {branch}"
    except CTeamError as e:
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
    except CTeamError:
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


# -----------------------------
# tmux helpers
# -----------------------------

def tmux(cmd: List[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    return run_cmd(["tmux", *cmd], capture=capture, check=check)


def tmux_has_session(session: str) -> bool:
    try:
        tmux(["has-session", "-t", session], capture=True, check=True)
        return True
    except CTeamError:
        return False


def tmux_list_windows(session: str) -> List[str]:
    cp = tmux(["list-windows", "-t", session, "-F", "#{window_name}"], capture=True)
    return [l.strip() for l in (cp.stdout or "").splitlines() if l.strip()]


def tmux_new_session(session: str, root_dir: Path) -> None:
    tmux(["new-session", "-d", "-s", session, "-n", "root", "-c", str(root_dir)], capture=True)
    # Keep windows after command exits (esp. when we exec codex)
    tmux(["set-option", "-t", session, "remain-on-exit", "on"], capture=True, check=False)
    tmux(["set-option", "-t", session, "allow-rename", "off"], capture=True, check=False)


def tmux_new_window(session: str, name: str, start_dir: Path, command_args: Optional[List[str]] = None) -> None:
    cmd = ["new-window", "-t", session, "-n", name, "-c", str(start_dir)]
    if command_args:
        cmd.extend(command_args)
    tmux(cmd, capture=True)


def tmux_respawn_window(session: str, window: str, start_dir: Path, command_args: List[str]) -> None:
    """
    Replace the command running in a window with a new command.

    Falls back to kill-window + new-window if `respawn-window` isn't supported.
    """
    cmd = ["respawn-window", "-t", f"{session}:{window}", "-c", str(start_dir)]
    cmd.extend(command_args)
    try:
        tmux(cmd, capture=True)
        return
    except CTeamError:
        # Older tmux: no respawn-window
        tmux(["kill-window", "-t", f"{session}:{window}"], capture=True, check=False)
        tmux_new_window(session, window, start_dir, command_args=command_args)


def tmux_send_keys(session: str, window: str, keys: List[str]) -> None:
    tmux(["send-keys", "-t", f"{session}:{window}", *keys], capture=True)


def tmux_select_window(session: str, window: str) -> None:
    tmux(["select-window", "-t", f"{session}:{window}"], capture=True, check=False)


def tmux_send_line(session: str, window: str, text: str) -> None:
    """
    Paste text (with trailing newline) into a tmux pane in one shot to avoid
    key parsing issues.
    """
    target = f"{session}:{window}"
    payload = text if text.endswith("\n") else text + "\n"
    tmux(["set-buffer", "--", payload], capture=True)
    tmux(["paste-buffer", "-d", "-t", target], capture=True)


def tmux_send_line_when_quiet(
    session: str,
    window: str,
    text: str,
    *,
    quiet_for: float = 2.0,
    block_timeout: float = 30.0,
) -> bool:
    """
    Block until the pane is quiet, then send text+Enter once.
    If the pane never goes quiet before block_timeout, do nothing.
    """
    payload = text if text.endswith("\n") else text + "\n"
    target = f"{session}:{window}"
    deadline = time.time() + max(block_timeout, 0.5)
    while time.time() < deadline:
        remaining = max(0.5, deadline - time.time())
        if wait_for_pane_quiet(session, window, quiet_for=quiet_for, timeout=min(quiet_for + 1.0, remaining)):
            tmux(["set-buffer", "--", payload], capture=True)
            tmux_send_keys(session, window, ["C-u"])
            tmux(["paste-buffer", "-d", "-t", target], capture=True)
            tmux_send_keys(session, window, ["Enter"])
            return True
    return False


def tmux_pane_current_command(session: str, window: str) -> str:
    """
    Best-effort detection of what's running in the active pane of a window.
    """
    try:
        cp = tmux(["list-panes", "-t", f"{session}:{window}", "-F", "#{pane_current_command}"], capture=True)
        out = (cp.stdout or "").strip().splitlines()
        return out[0].strip() if out else ""
    except Exception:
        return ""


def wait_for_pane_command(session: str, window: str, target_substring: str, timeout: float = 4.0) -> bool:
    """
    Wait until the active pane command contains target_substring (case-insensitive) or timeout.
    """
    deadline = time.time() + max(timeout, 0.1)
    target = target_substring.lower()
    while time.time() < deadline:
        cmd = tmux_pane_current_command(session, window).lower()
        if target in cmd:
            return True
        time.sleep(0.1)
    return False


def _pane_sig(session: str, window: str) -> Tuple[int, str]:
    """
    Lightweight signature of the pane contents: length + tail slice.
    """
    try:
        cp = tmux(["capture-pane", "-p", "-t", f"{session}:{window}", "-J"], capture=True)
        txt = (cp.stdout or "")
        return (len(txt), txt[-400:])
    except Exception:
        return (0, "")


def wait_for_pane_quiet(session: str, window: str, *, quiet_for: float = 1.0, timeout: float = 6.0) -> bool:
    """
    Wait until the pane output is stable (no changes) for quiet_for seconds, or timeout.
    """
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
            tmux(["select-window", "-t", f"{session}:{window}"], capture=True, check=False)
        except Exception:
            pass
    os.execvp("tmux", ["tmux", "attach-session", "-t", session])


def tmux_kill_session(session: str) -> None:
    tmux(["kill-session", "-t", session], capture=True, check=False)


# -----------------------------
# Codex CLI capability detection
# -----------------------------

@dataclass
class CodexCaps:
    cmd: str
    help_text: str

    def supports(self, flag: str) -> bool:
        return flag in self.help_text

    def has_any(self, flags: List[str]) -> Optional[str]:
        for f in flags:
            if self.supports(f):
                return f
        return None


_codex_caps_cache: Dict[str, CodexCaps] = {}


def codex_caps(codex_cmd: str) -> CodexCaps:
    if codex_cmd in _codex_caps_cache:
        return _codex_caps_cache[codex_cmd]
    help_text = ""
    try:
        cp = run_cmd([codex_cmd, "--help"], capture=True, check=False)
        help_text = (cp.stdout or "") + "\n" + (cp.stderr or "")
    except Exception:
        help_text = ""
    caps = CodexCaps(cmd=codex_cmd, help_text=help_text)
    _codex_caps_cache[codex_cmd] = caps
    return caps


def build_codex_base_args(state: Dict[str, Any]) -> List[str]:
    """
    Build Codex CLI args (excluding initial prompt and excluding cwd).
    We intentionally do NOT rely on positional PROMPT support; we will send a prompt via tmux later.
    """
    codex = state["codex"]
    cmd = codex.get("cmd", DEFAULT_CODEX_CMD)
    caps = codex_caps(cmd)

    args: List[str] = [cmd]

    # YOLO / bypass flags (if requested)
    if codex.get("yolo", False):
        flag = caps.has_any(["--dangerously-bypass-approvals-and-sandbox", "--yolo"])
        if flag:
            args.append(flag)
        else:
            pass
    elif codex.get("full_auto", False):
        if caps.supports("--full-auto"):
            args.append("--full-auto")
        else:
            if caps.supports("--ask-for-approval"):
                args += ["--ask-for-approval", "on-failure"]
            if caps.supports("--sandbox"):
                args += ["--sandbox", "workspace-write"]
    else:
        if caps.supports("--sandbox"):
            args += ["--sandbox", codex.get("sandbox", DEFAULT_CODEX_SANDBOX)]
        if caps.supports("--ask-for-approval"):
            args += ["--ask-for-approval", codex.get("ask_for_approval", DEFAULT_CODEX_APPROVAL)]
        elif caps.supports("--approval-policy"):
            args += ["--approval-policy", codex.get("ask_for_approval", DEFAULT_CODEX_APPROVAL)]
        elif caps.supports("--approval-mode"):
            ap = codex.get("ask_for_approval", DEFAULT_CODEX_APPROVAL)
            mode = "full-auto" if ap == "never" else "auto"
            args += ["--approval-mode", mode]

    if codex.get("search", DEFAULT_CODEX_SEARCH) and caps.supports("--search"):
        args.append("--search")

    model = codex.get("model")
    if model and caps.supports("--model"):
        args += ["--model", model]

    return args


def build_codex_args_for_agent(state: Dict[str, Any], agent: Dict[str, Any], start_dir: Path) -> List[str]:
    """
    Build full argv to run Codex for an agent. We set cwd via tmux -c, and optionally also --cd.
    """
    base = build_codex_base_args(state)
    cmd = state["codex"].get("cmd", DEFAULT_CODEX_CMD)
    caps = codex_caps(cmd)

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


# -----------------------------
# State model / defaults
# -----------------------------

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
    codex_cmd: str,
    codex_model: Optional[str],
    sandbox: str,
    approval: str,
    search: bool,
    full_auto: bool,
    yolo: bool,
    autostart: str,
    router: bool,
) -> Dict[str, Any]:
    root_abs = str(root.resolve())
    session = f"cteam_{slugify(project_name)}"

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
        "codex": {
            "cmd": codex_cmd,
            "model": codex_model,
            "sandbox": sandbox,
            "ask_for_approval": approval,
            "search": bool(search),
            "full_auto": bool(full_auto),
            "yolo": bool(yolo),
        },
        "coordination": {
            "autostart": autostart,       # pm | pm+architect | all
            "pm_is_boss": True,
            "start_agents_on_assignment": True,
            "assignment_type": ASSIGNMENT_TYPE,
            "assignment_from": ["pm", "cteam"],   # who can "start" an agent by messaging them
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

    # --- v < 4 upgrades (legacy) ---
    if v < 4:
        state.setdefault("created_at", now_iso())
        state.setdefault("project_name", root.name)
        state.setdefault("mode", state.get("mode", "new"))
        state.setdefault("imported_from", state.get("imported_from"))
        state.setdefault("root_abs", str(root.resolve()))
        state.setdefault("git", {}).setdefault("default_branch", DEFAULT_GIT_BRANCH)

        codex = state.get("codex", {})
        codex.setdefault("cmd", DEFAULT_CODEX_CMD)
        codex.setdefault("model", None)
        codex.setdefault("sandbox", DEFAULT_CODEX_SANDBOX)
        codex.setdefault("ask_for_approval", DEFAULT_CODEX_APPROVAL)
        codex.setdefault("search", DEFAULT_CODEX_SEARCH)
        codex.setdefault("full_auto", False)
        codex.setdefault("yolo", False)
        state["codex"] = codex

        coord = state.get("coordination", {})
        coord.setdefault("autostart", DEFAULT_AUTOSTART)
        coord.setdefault("pm_is_boss", True)
        coord.setdefault("start_agents_on_assignment", True)
        coord.setdefault("assignment_type", ASSIGNMENT_TYPE)
        coord.setdefault("assignment_from", ["pm", "cteam"])
        state["coordination"] = coord

        tm = state.get("tmux", {})
        tm.setdefault("session", f"cteam_{slugify(state.get('project_name', root.name))}")
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

    # --- v < 5 upgrades (Telegram integration state) ---
    if v < 5:
        tg = state.get("telegram", {})
        if not isinstance(tg, dict):
            tg = {}
        tg.setdefault("enabled", False)
        tg.setdefault("config_rel", TELEGRAM_CONFIG_REL)
        state["telegram"] = tg

    # --- v < 6 upgrades (tighten assignment_from defaults) ---
    if v < 6:
        coord = state.get("coordination", {})
        if not isinstance(coord, dict):
            coord = {}
        current = coord.get("assignment_from", ["pm", "cteam"])
        if "customer" in current:
            coord["assignment_from"] = [x for x in current if x != "customer"]
        coord.setdefault("assignment_type", ASSIGNMENT_TYPE)
        coord.setdefault("start_agents_on_assignment", True)
        coord.setdefault("pm_is_boss", True)
        state["coordination"] = coord
    # --- v < 7 upgrades (tickets) ---
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
        # Drop legacy md_rel if present.
        tickets_cfg = state.get("tickets") or {}
        if isinstance(tickets_cfg, dict) and "md_rel" in tickets_cfg:
            tickets_cfg.pop("md_rel", None)
            state["tickets"] = tickets_cfg

    state["version"] = STATE_VERSION
    compute_agent_abspaths(state)
    save_state(root, state)
    return state


def save_state(root: Path, state: Dict[str, Any]) -> None:
    atomic_write_text(root / STATE_FILENAME, json.dumps(state, indent=2, sort_keys=True) + "\n")


def load_state(root: Path) -> Dict[str, Any]:
    p = root / STATE_FILENAME
    if not p.exists():
        raise CTeamError(f"state file not found: {p}")
    state = json.loads(p.read_text(encoding="utf-8"))
    state["root_abs"] = str(root.resolve())
    state = upgrade_state_if_needed(root, state)
    compute_agent_abspaths(state)
    return state


# -----------------------------
# Templates / docs
# -----------------------------

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
        # Coordination protocol (PM-led)

        This workspace uses a PM-led protocol to avoid "everyone doing random work".

        ## Rules
        1) The **PM** owns coordination: goals, plan, tasks, sequencing, and merges.
        2) Non-PM agents **must not** start implementing work without an explicit message of:
           - `Type: {ASSIGNMENT_TYPE}`
        3) Default behavior for non-PM agents when unassigned:
           - do a quick scan if helpful
           - send a short note to the PM
           - then wait
        4) Agents communicate via mailboxes in `shared/mail/<agent>/`.

        ## How to assign work
        - Human or PM should use:
          - `python3 cteam.py <workdir> assign --to dev1 --task T001 --subject "..." "instructions..."`

        ## Router
        - A tmux window named `{ROUTER_WINDOW}` runs `cteam <workdir> watch`:
          - nudges agents when mail arrives
          - can auto-start Codex in an agent window when an assignment arrives
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

        Use this page for high-level goals/milestones only. All actionable work must live in tickets (`cteam tickets ...`).

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
        - List tickets: `python3 cteam.py . tickets list`
        - Show ticket:  `python3 cteam.py . tickets show --id T001`
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
        # Shared workspace — Clanker Team (cteam)

        Project: **{state['project_name']}**
        Root: `{state['root_abs']}`

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
        - tmux window `{ROUTER_WINDOW}` runs `python3 cteam.py . watch`
        """
    )


def render_team_roster(state: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Team roster (cteam)\n")
    lines.append(f"- Project: **{state['project_name']}**")
    lines.append(f"- Root: `{state['root_abs']}`")
    lines.append(f"- Created: {state['created_at']}")
    lines.append(f"- Mode: `{state.get('mode','new')}`")
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
        # STATUS — {agent['name']} ({agent['title']})

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


def render_agent_agents_md(state: Dict[str, Any], agent: Dict[str, Any]) -> str:
    role = agent["role"]
    mode = state.get("mode", "new")
    agent_cteam_rel = Path("..") / ".." / "cteam.py"
    repo_cteam_rel = Path("cteam.py")

    common = textwrap.dedent(
        f"""\
        # AGENTS.md — {agent['name']} ({agent['title']})

        You are a Codex agent in a multi-agent team managed by **Clanker Team (cteam)**.

        Project: **{state['project_name']}**
        Mode: `{mode}`
        Your identity/persona: {agent.get('persona','')}

        ## Evolving this file
        You may update **this AGENTS.md** as you learn better ways to work (tools, conventions, shortcuts),
        *as long as* you keep the PM-led coordination rules intact.

        ## Coordination (IMPORTANT)
        This project uses a **PM-led** protocol. Unless you are the PM:

        - Do NOT start implementing random work.
        - Wait for a message with:
          - `Type: {ASSIGNMENT_TYPE}`
        - If unassigned, you may do *lightweight reconnaissance* and send a short note to PM,
          but avoid code changes/commits.
        - All work is tracked in tickets (managed via `cteam tickets ...`). Only work on tickets assigned to you. Reference ticket IDs in STATUS updates and commits. Do not edit ticket files directly—use `cteam tickets ...` and `cteam assign`.

        ## Where to work
        - Your repo clone: `proj/` (work here)
        - Your inbox: `message.md` (check frequently)
        - Update: `STATUS.md`
        - Shared drive (non-repo artifacts): `shared-drive/` (links to shared/drive)

        Shared coordination:
        - `shared/GOALS.md`, `shared/PLAN.md` (high-level), tickets via `cteam tickets ...`, `shared/DECISIONS.md`
        - `shared/TEAM_ROSTER.md` (who is on the team)
        - `shared/drive/` for large/binary assets; keep code/config in git.

        ## Tooling
        - cteam CLI from your agent dir: `python3 {agent_cteam_rel} ...`
        - From your repo clone `proj/`: `python3 {repo_cteam_rel} ...`

        ## Team roles (who does what)
        - Project Manager (PM): owns scope/priorities and customer comms; plans work and delegates; avoids doing implementation unless absolutely necessary.
        - Architect: owns technical direction/design; records decisions; partners with PM on scope/impact.
        - Developers: implement assigned tasks; keep changes small/mergeable; do not self-assign from customer asks.
        - Tester/QA: owns verification; adds/executes tests; never offload testing to the customer.
        - Researcher: reduces uncertainty with actionable notes; no direct customer asks.
        - Customer: provides inputs/feedback only through PM; never asked to do engineering.

        ## Background runs / long tasks
        - If you start a long-running or background command, record it in `STATUS.md` with owner, command, working dir, start time, expected finish, and purpose.
        - Tell PM when it starts/finishes; if it stalls, update PM/STATUS.

        ## Messaging
        Preferred: use cteam CLI so delivery+nudge is reliable:

        - Send message:
          - `python3 cteam.py $CTEAM_ROOT msg --to pm --subject "..." "text..."`
        - Interactive chat:
          - `python3 cteam.py $CTEAM_ROOT chat --to pm`
        - Assignments (PM/human):
          - `python3 cteam.py $CTEAM_ROOT assign --to dev1 --task T001 --subject "..." "instructions"`

        If you must write manually, append to:
        - `shared/mail/<agent>/message.md`

        ## Git rules
        - Branch naming: `agent/{agent['name']}/<topic>`
        - Push early, push often; pull/rebase frequently to avoid conflicts.
        - Avoid rewriting shared history.
        - Keep large binaries out of git; put them in `shared/drive/` and reference them in notes/PRs.

        ## Role clarity
        - The customer never performs engineering tasks; they provide inputs, feedback, and clarifications.
        - Respect your lane: do your role’s work well; do not offload your duties to the customer.
        """
    )

    if role == "project_manager":
        block = textwrap.dedent(
            f"""\
            ## Role: Project Manager (the coordinator)
            Customer expectation: operate autonomously. Exhaust team options before contacting the customer. Only reach out after several concrete attempts (reading docs/code, small spikes, reproductions) have failed, and bundle any outreach with a concise summary of what was tried and the proposed next move.

            You are responsible for:
            - clarifying goals and constraints
            - producing/maintaining the plan and task breakdown
            - assigning work to other agents and rebalancing load
            - keeping everyone in sync and unblocked
            - ensuring the project completes with quality and customer intent intact

            Startup checklist:
            1) Read seed/ (if any) and repo README/docs.
            2) Capture assumptions + open questions in `shared/GOALS.md`.
            3) Draft `shared/PLAN.md` (goals/milestones only) and create tickets for all work.
            4) Create tickets (`cteam tickets create ...` or `cteam assign --title/--desc`) and assign via `cteam assign --ticket ...`.

            Execution cadence:
            - Keep non-PM agents focused on one assignment at a time; rebalance when capacity or priorities change.
            - Onboard new agents quickly and redistribute work to maintain momentum.
            - Require agents to keep `STATUS.md` current; collect short status updates regularly.
            - Track long-running/background work; note owner/command/timing in STATUS and nudge until complete or fixed.
            - If you must stop an agent’s input urgently, use `cteam nudge . --to <agent> --reason "..." --interrupt` (sends Escape first); follow with clear instructions.

            Customer channel policy:
            - Reply promptly when reading a customer message with at least an acknowledgement (e.g., “Received; filed Txxx, will report back”).
            - Minimize new asks; only ask questions after multiple self-serve attempts fail. When you do ask, include what was attempted and the proposed path forward.
            - Keep tickets as the single source of truth; migrate any stray PLAN/TASKS checklists into tickets and close duplicates there.

            Delegation and quality:
            - You are the planner/owner, not the implementer; only code as a last resort and keep changes minimal.
            - Lean on architect for design, developers for execution, tester for verification, researcher for uncertainty reduction.
            - Ensure deliveries match customer intent and quality before sending anything out.
            """
        )
    elif role == "architect":
        block = textwrap.dedent(
            """\
            ## Role: Architect
            - Produce/maintain architecture docs.
            - Record major decisions in shared/DECISIONS.md.
            - Coordinate with PM before making broad changes.
            - Keep the customer out of engineering tasks; they provide context/feedback only via PM.
            - Work only on tickets assigned to you; propose/shape tickets with PM for new design work.

            Default behavior if unassigned:
            - create a short "architecture map" note for PM
            - propose 1-3 concrete tasks
            - then wait
            """
        )
    elif role == "developer":
        block = textwrap.dedent(
            """\
            ## Role: Developer
            - Implement assigned tasks.
            - Keep changes small and mergeable.
            - Write tests when feasible.
            - Do not ask the customer to do engineering work; request clarifications via PM.
            - Work only on tickets assigned to you; include ticket IDs in commits/STATUS updates.

            Default behavior if unassigned:
            - scan repo quickly, propose a ticket to PM
            - do NOT start coding without assignment
            """
        )
    elif role == "tester":
        block = textwrap.dedent(
            """\
            ## Role: Tester / QA
            - Establish how to run tests.
            - Add smoke tests and regression tests.
            - Verify fixes.
            - You own quality; never push testing effort onto the customer.
            - Work only on tickets assigned to you; reference ticket IDs in STATUS updates and test notes.

            Default behavior if unassigned:
            - find how to run the project/tests, write a short note to PM
            - do NOT change code without assignment
            """
        )
    elif role == "researcher":
        block = textwrap.dedent(
            """\
            ## Role: Researcher
            - Inventory dependencies and risks.
            - Provide actionable notes to PM/Architect.
            - Use seed-extras and (optional) web search.
            - Do not ask the customer to do research; route questions through PM.

            Default behavior if unassigned:
            - write a dependency inventory note and send it to PM
            - then wait
            """
        )
    else:
        block = ""

    return common + "\n" + block


def initial_prompt_for_pm(state: Dict[str, Any]) -> str:
    mode = state.get("mode", "new")
    root_abs = state.get("root_abs", "")
    path_hint = f"cteam.py location: {root_abs}/cteam.py. "
    if mode == "import":
        return (
            "You are the Project Manager. First open message.md for any kickoff notes. "
            "Then immediately infer what this codebase is for. "
            "Open shared/GOALS.md and fill it with inferred goals, non-goals, and questions. "
            "Then write shared/PLAN.md (high-level only). Track all work in tickets via `cteam tickets ...`. "
            "Plan discovery work as tickets and assign them to the team to map the imported repo. "
            "Assign work with `cteam assign --ticket ...` (auto-create tickets with --title/--desc). "
            "Keep everyone coordinated; do not let others work unassigned. "
            "Start now. "
            f"{path_hint}"
        )
    return (
        "You are the Project Manager. First open message.md for any kickoff notes. "
        "Then read seed/ and write shared/GOALS.md + shared/PLAN.md (high-level only). "
        "Track work via `cteam tickets ...`. Assign tasks with `cteam assign --ticket ...` "
        "or auto-create tickets using --title/--desc. Keep everyone coordinated. Start now. "
        f"{path_hint}"
    )


def prompt_on_mail(agent_name: str) -> str:
    return (
        f"MAILBOX UPDATED for {agent_name}. Open message.md and follow the latest instructions. "
        "If it's an assignment, execute it and update STATUS.md. "
        "If unclear, ask PM a precise question."
    )


def standby_banner(agent: Dict[str, Any], state: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""\
        Clanker Team (cteam) — {state['project_name']}
        Agent window: {agent['name']} ({agent['title']})
        Mode: {state.get('mode','new')}

        Standby mode (PM-led coordination):
        - Waiting for PM/human assignment.
        - Your inbox: {DIR_MAIL}/{agent['name']}/message.md
        - Tickets: managed via `cteam tickets ...` (work only on tickets assigned to you)
        - If you are assigned work, this window will be repurposed to run Codex automatically.

        Tip: If you want to manually check mail:
          less -R {shlex.quote(str(Path(state['root_abs'])/DIR_MAIL/agent['name']/ 'message.md'))}
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
        atomic_write_text(root / DIR_SEED_EXTRAS / "README.md", render_seed_extras_readme())

    shared_files = {
        "README.md": render_shared_readme(state),
        "TEAM_ROSTER.md": render_team_roster(state),
        "PROTOCOL.md": render_protocol_md(),
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
            atomic_write_text(readme, f"# {state['project_name']}\n\n(Initialized by cteam)\n")
        docs = checkout / "docs"
        mkdirp(docs)
        if not (docs / "README.md").exists():
            atomic_write_text(docs / "README.md", "# Docs\n\n")
        git_config_local(checkout, "user.name", "cteam")
        git_config_local(checkout, "user.email", "cteam@local")
        run_cmd(["git", "-C", str(checkout), "add", "-A"])
        run_cmd(["git", "-C", str(checkout), "commit", "-m", "chore: initialize repository"], check=False)
        run_cmd(["git", "-C", str(checkout), "push", "-u", "origin", state["git"]["default_branch"]], check=False)

        try:
            git_append_info_exclude(checkout, [f"{DIR_SHARED}/", f"{DIR_SEED}/", f"{DIR_SEED_EXTRAS}/"])
        except Exception:
            pass


def create_git_scaffold_import(root: Path, state: Dict[str, Any], src: str) -> None:
    bare = root / DIR_PROJECT_BARE
    checkout = root / DIR_PROJECT_CHECKOUT

    if bare.exists() and any(bare.iterdir()):
        raise CTeamError(f"{DIR_PROJECT_BARE} already exists and is not empty in {root}")

    imported_as_git = False
    try:
        git_clone_bare(src, bare)
        imported_as_git = True
    except CTeamError:
        imported_as_git = False
        if _looks_like_git_url(src):
            raise

    if imported_as_git:
        if not checkout.exists():
            git_clone(bare, checkout)
        try:
            git_append_info_exclude(checkout, [f"{DIR_SHARED}/", f"{DIR_SEED}/", f"{DIR_SEED_EXTRAS}/"])
        except Exception:
            pass
        return

    src_path = Path(src).expanduser().resolve()
    if not src_path.exists() or not src_path.is_dir():
        raise CTeamError(f"--src is neither a git repo nor a readable directory: {src}")

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

    git_config_local(checkout, "user.name", "cteam-import")
    git_config_local(checkout, "user.email", "cteam+import@local")
    run_cmd(["git", "-C", str(checkout), "add", "-A"])
    try:
        run_cmd(["git", "-C", str(checkout), "commit", "-m", "Import existing project"], capture=True)
    except CTeamError:
        run_cmd(["git", "-C", str(checkout), "commit", "--allow-empty", "-m", "Import existing project"], capture=True)

    run_cmd(["git", "-C", str(checkout), "push", "origin", "HEAD"], capture=True)

    try:
        git_append_info_exclude(checkout, [f"{DIR_SHARED}/", f"{DIR_SEED}/", f"{DIR_SEED_EXTRAS}/"])
    except Exception:
        pass


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
            ["AGENTS.md", "STATUS.md", "message.md", "seed/", "seed-extras/", "shared/", "cteam.py"],
        )
    except Exception:
        pass

    safe_link_file(Path("..") / "AGENTS.md", repo_dir / "AGENTS.md")
    safe_link_file(Path("..") / "STATUS.md", repo_dir / "STATUS.md")
    safe_link_file(Path("..") / "message.md", repo_dir / "message.md")
    safe_link_dir(Path("..") / ".." / ".." / DIR_SHARED, repo_dir / "shared")
    safe_link_dir(Path("..") / ".." / ".." / DIR_SHARED_DRIVE, repo_dir / "shared-drive")
    safe_link_dir(Path("..") / ".." / DIR_SHARED_DRIVE, agent_dir / "shared-drive")
    if (agent_dir / "seed").exists():
        safe_link_dir(Path("..") / "seed", repo_dir / "seed")
    if (agent_dir / "seed-extras").exists():
        safe_link_dir(Path("..") / "seed-extras", repo_dir / "seed-extras")

    if (root / "cteam.py").exists():
        safe_link_file(Path("..") / ".." / "cteam.py", agent_dir / "cteam.py")
        safe_link_file(Path("..") / ".." / ".." / "cteam.py", repo_dir / "cteam.py")


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
        if agent_name in set(tmux_list_windows(session)):
            tmux(["kill-window", "-t", f"{session}:{agent_name}"], capture=True)
    except Exception:
        pass


def _archive_or_remove_agent_dirs(root: Path, agent: Dict[str, Any], *, purge: bool = False) -> None:
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


def _unassign_tickets_for_agent(root: Path, state: Dict[str, Any], agent_name: str) -> None:
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
                        "user": "cteam",
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
        if "MAILBOX UPDATED" in upper_r:
            normalized.add("MAILBOX UPDATED")
            continue
        normalized.add(r_clean)
    return normalized or {"NUDGE"}


def _mailbox_nudge_reason(ticket_id: Optional[str], msg_type: Optional[str]) -> str:
    reason = "MAILBOX UPDATED"
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
) -> None:
    ts = now_iso()
    if ticket_id:
        ticket_id = ticket_id.strip()
    entry = format_message(ts, sender, recipient, subject, body, msg_type=msg_type, task=task, ticket_id=ticket_id or None)
    if not entry.endswith("\n"):
        entry += "\n"

    agent_names = {a["name"] for a in state["agents"]}
    allowed = set(agent_names) | {"customer"}
    allowed_senders = allowed | {"cteam"}
    if recipient not in allowed:
        raise CTeamError(f"unknown recipient: {recipient}")
    if sender not in allowed_senders:
        raise CTeamError(f"unknown sender: {sender}")
    if sender == "customer" and recipient != "pm":
        raise CTeamError("customer can only message pm")
    if recipient == "customer" and sender not in {"pm", "cteam"}:
        raise CTeamError("only pm can message customer")
    if msg_type == ASSIGNMENT_TYPE and sender not in {"pm", "cteam"}:
        raise CTeamError("assignments must come from pm")
    if ticket_id:
        with ticket_lock(root, state):
            store = load_tickets(root, state)
            t = _find_ticket(store, ticket_id)
            if not t:
                raise CTeamError(f"unknown ticket: {ticket_id}")
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

    atomic_write_text(inbox_dir / f"{ts_for_filename(ts)}_{sender}.md", entry)

    append_text(msg_path, entry)

    append_text(root / DIR_SHARED / "MESSAGES.log.md", entry)

    if msg_type == ASSIGNMENT_TYPE:
        append_text(root / DIR_SHARED / "ASSIGNMENTS.log.md", entry)

    if sender == "pm" and recipient == "customer":
        cust_state = load_customer_state(root)
        cust_state["last_pm_reply_ts"] = ts
        save_customer_state(root, cust_state)

    if sender == "customer" and recipient == "pm":
        cust_state = load_customer_state(root)
        cust_state["last_customer_msg_ts"] = ts
        save_customer_state(root, cust_state)
        ack_body = (
            "Automated acknowledgement: we received your message and notified the PM. "
            "They will reply in this chat promptly. If action is needed, the PM will file the task "
            "and share status/results here."
        )
        try:
            write_message(
                root,
                state,
                sender="cteam",
                recipient="customer",
                subject="We received your message — PM notified",
                body=ack_body,
                msg_type="MESSAGE",
                task=None,
                nudge=False,
                start_if_needed=False,
            )
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
    if sender and sender != recipient and sender not in {"customer", "cteam"}:
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
            maybe_start_agent_on_message(root, state, recipient, sender=sender, msg_type=msg_type)
        nudge_reason = _mailbox_nudge_reason(ticket_id, msg_type)
        nudge_agent(root, state, recipient, reason=nudge_reason)


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
    tmux(["set-option", "-t", session, "remain-on-exit", "on"], capture=True, check=False)
    tmux(["set-option", "-t", session, "allow-rename", "off"], capture=True, check=False)


def router_window_command(root: Path) -> List[str]:
    shell = pick_shell()
    cmd = f"cd {shlex.quote(str(root))} && exec python3 cteam.py . watch"
    return shell + [cmd]

def customer_window_command(root: Path) -> List[str]:
    shell = pick_shell()
    cmd = f"cd {shlex.quote(str(root))} && exec python3 cteam.py . customer-chat"
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


def standby_window_command(root: Path, state: Dict[str, Any], agent: Dict[str, Any]) -> List[str]:
    shell = pick_shell()
    banner = standby_banner(agent, state).replace("'", "'\"'\"'")
    sh = os.environ.get("SHELL", "bash")
    cmd = f"cd {shlex.quote(str(Path(agent['dir_abs'])))} && printf '%s\n' '{banner}' && exec {shlex.quote(sh)}"
    return shell + [cmd]


def codex_window_command(root: Path, state: Dict[str, Any], agent: Dict[str, Any], *, boot: bool) -> List[str]:
    shell = pick_shell()
    repo_dir = Path(agent["repo_dir_abs"])
    base_args = build_codex_args_for_agent(state, agent, start_dir=repo_dir)

    exports = (
        f"export CTEAM_ROOT={shlex.quote(str(root))} "
        f"CTEAM_AGENT={shlex.quote(agent['name'])}; "
    )
    cmd = exports + "exec " + " ".join(shlex.quote(x) for x in base_args)
    cmd = "cd " + shlex.quote(str(repo_dir)) + " && " + cmd
    return shell + [cmd]


def is_codex_running(state: Dict[str, Any], window: str) -> bool:
    session = state["tmux"]["session"]
    cmd = tmux_pane_current_command(session, window)
    return "codex" in cmd.lower()


def start_codex_in_window(
    root: Path,
    state: Dict[str, Any],
    agent: Dict[str, Any],
    *,
    boot: bool,
) -> None:
    session = state["tmux"]["session"]
    w = agent["name"]
    repo_dir = Path(agent["repo_dir_abs"])
    if not repo_dir.exists():
        create_agent_dirs(root, state, agent)
        repo_dir = Path(agent["repo_dir_abs"])
    cmd_args = codex_window_command(root, state, agent, boot=boot)

    windows = set(tmux_list_windows(session))
    if w not in windows:
        tmux_new_window(session, w, repo_dir, command_args=cmd_args)
    else:
        tmux_respawn_window(session, w, repo_dir, command_args=cmd_args)

    if boot:
        if agent["name"] == "pm":
            first_prompt = initial_prompt_for_pm(state)
        else:
            if state.get("mode") == "import":
                first_prompt = (
                    f"You are {agent['title']} ({agent['name']}). Imported project; wait for ticketed assignments. "
                    f"Open AGENTS.md and message.md. Do NOT change code without a ticketed assignment (Type: {ASSIGNMENT_TYPE}). "
                    f"Send a short discovery note to PM if needed, then wait for a ticket. "
                    f"Tickets are managed via `cteam tickets ...`. cteam.py location: {state.get('root_abs','')}/cteam.py"
                )
            else:
                first_prompt = (
                    f"You are {agent['title']} ({agent['name']}). "
                    f"Open AGENTS.md and message.md. "
                    f"If you do not have an assignment (Type: {ASSIGNMENT_TYPE}), do NOT start coding; "
                    f"send a short recon note to PM, then wait. "
                    f"cteam.py location: {state.get('root_abs','')}/cteam.py"
                )
    else:
        first_prompt = prompt_on_mail(agent["name"])

    wait_for_pane_command(session, w, "codex", timeout=4.0)
    wait_for_pane_quiet(session, w, quiet_for=0.8, timeout=6.0)
    tmux_send_keys(session, w, ["C-u"])
    tmux_send_line_when_quiet(session, w, first_prompt)


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
    try:
        if agent_name not in set(tmux_list_windows(session)):
            ensure_agent_windows(root, state, launch_codex=True)
    except Exception:
        pass

    extra = ""
    if agent_name == "pm" and "IDLE" in reason.upper():
        extra = " Check who should be working, unblock them, and assign next steps. Confirm any background tasks are still running."
    if interrupt:
        msg = f"INTERRUPT — {reason}: open message.md now and check if you replied to everything. Update STATUS.md after acting. {extra}"
    else:
        msg = f"{reason}: open message.md and act. Also check if you replied to everything. Read and update STATUS.md. Make sure progress is being made. If you need anything from me, the customer, you can only reach me via the customer-pm chat, not here. {extra}"
    try:
        if interrupt:
            try:
                tmux_send_keys(session, agent_name, ["Escape"])
                time.sleep(0.1)
            except Exception:
                pass
        tmux_send_keys(session, agent_name, ["C-u"])
        ok = False
        if is_codex_running(state, agent_name):
            wait_for_pane_quiet(session, agent_name, quiet_for=0.8, timeout=4.0)
            ok = tmux_send_line_when_quiet(session, agent_name, msg)
        else:
            ok = tmux_send_line_when_quiet(session, agent_name, f"echo {shlex.quote(msg)}")
        if ok:
            _nudge_history[agent_name] = (time.time(), _normalize_reasons(_split_reasons(reason)))
        return ok
    except Exception:
        return False


class NudgeQueue:
    """Batch and deduplicate nudges before sending to tmux windows; defers until pane idle."""

    def __init__(self, root: Path, *, min_interval: float = 2.0, idle_for: float = 1.0) -> None:
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

    def _merge_back(self, agent_name: str, reasons: List[str], interrupt: bool) -> None:
        entry = self.pending.setdefault(agent_name, {"reasons": [], "interrupt": False})
        for r in reasons:
            if r not in entry["reasons"]:
                entry["reasons"].append(r)
        entry["interrupt"] = entry["interrupt"] or interrupt

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
        for agent_name in list(self.pending.keys()):
            entry = self.pending.pop(agent_name, {})
            reasons = entry.get("reasons") or []
            interrupt = bool(entry.get("interrupt", False))
            reason_text = " / ".join(reasons) if reasons else "NUDGE"

            if not self._is_idle(state, agent_name):
                self._merge_back(agent_name, reasons, interrupt)
                results.append((agent_name, False, reason_text, "busy"))
                continue

            last = self.last_sent.get(agent_name) or _nudge_history.get(agent_name)
            last_ts = None
            last_reasons: Set[str] = set()
            if last:
                try:
                    last_ts = float(last[0])
                    last_reasons = _normalize_reasons(set(last[1]))
                except Exception:
                    last_ts, last_reasons = None, set()

            current_reasons_raw = set(reasons) if reasons else _split_reasons(reason_text)
            current_reasons = _normalize_reasons(current_reasons_raw)
            if last_ts is not None and (now - last_ts) < self.min_interval:
                if current_reasons.issubset(last_reasons):
                    results.append((agent_name, False, reason_text, "duplicate"))
                    continue

            ok = nudge_agent(self.root, state, agent_name, reason=reason_text, interrupt=interrupt)
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

    allowed_from = set(coord.get("assignment_from", ["pm", "cteam"]))
    if sender not in allowed_from:
        return

    if msg_type != coord.get("assignment_type", ASSIGNMENT_TYPE):
        return

    agent = next((a for a in state["agents"] if a["name"] == recipient), None)
    if not agent:
        return

    ensure_tmux_session(root, state)
    if not is_codex_running(state, recipient):
        start_codex_in_window(root, state, agent, boot=False)


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
        raise CTeamError(
            f"telegram not configured: missing {cfg_path}. Run: python3 cteam.py {shlex.quote(str(root))} telegram-configure"
        )
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise CTeamError(f"invalid telegram config JSON: {cfg_path} ({e})") from e

    token = (cfg.get("bot_token") or "").strip()
    phone = (cfg.get("authorized_phone") or "").strip()
    if not token or not phone:
        raise CTeamError(f"telegram config incomplete (need bot_token + authorized_phone): {cfg_path}")
    cfg["authorized_phone_norm"] = _normalize_phone(phone)
    return cfg


def telegram_save_config(root: Path, state: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    cfg_path = telegram_config_path(root, state)
    cfg2 = dict(cfg)
    cfg2.pop("authorized_phone_norm", None)
    atomic_write_text(cfg_path, json.dumps(cfg2, indent=2, sort_keys=True) + "\n")
    _chmod_private(cfg_path)


def _telegram_api(token: str, method: str, payload: Dict[str, Any], *, timeout: float = 30.0) -> Any:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise CTeamError(f"telegram HTTP error calling {method}: {e.code} {e.reason}\n{detail}".strip()) from e
    except Exception as e:
        raise CTeamError(f"telegram error calling {method}: {e}") from e

    try:
        obj = json.loads(raw)
    except Exception as e:
        raise CTeamError(f"telegram non-JSON response from {method}: {raw[:400]}") from e
    if not obj.get("ok"):
        raise CTeamError(f"telegram API returned ok=false for {method}: {raw[:600]}")
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
        _telegram_api(token, "sendMessage", {"chat_id": chat_id, "text": c}, timeout=20.0)


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
            "text": "To link this bot to your cteam customer channel, please share your phone number.",
            "reply_markup": markup,
        },
        timeout=20.0,
    )


def _parse_entry(entry: str) -> Tuple[Optional[str], Optional[str], Optional[str], str, str, Optional[str], Optional[str]]:
    ts = None
    sender = None
    recipient = None
    subject = ""
    ticket_id: Optional[str] = None
    msg_type: Optional[str] = None
    lines = entry.splitlines()
    if lines:
        m = re.match(r"^##\s+(.+?)\s+—\s+From:\s*([^\s→]+)\s+→\s+To:\s*([^\s]+)\s*$", lines[0].strip())
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
    return any(name.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif"))


class TelegramBridge:
    """Bridges Telegram <-> cteam customer channel."""

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
                log_line(self.root, "[telegram] reset update_offset to 0 (chat not authorized yet; send /start and share contact)")
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
            return int(chat.get("id", 0)) == int(cfg["chat_id"]) and int(frm.get("id", 0)) == int(cfg["user_id"])
        return False

    def _rel_path(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except Exception:
            return str(p)

    def _save_telegram_file(self, cfg: Dict[str, Any], file_id: Optional[str], *, prefer_name: Optional[str] = None) -> Optional[Path]:
        token = cfg.get("bot_token")
        if not token or not file_id:
            return None
        try:
            info = _telegram_api(token, "getFile", {"file_id": file_id}, timeout=20.0) or {}
            file_path = info.get("file_path")
            if not file_path:
                raise CTeamError("missing file_path")
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
                _telegram_send_text(cfg, "✅ Linked! This chat is now authorized for the cteam customer channel.")
                log_line(self.root, f"[telegram] authorized chat_id={cfg['chat_id']} user_id={cfg['user_id']}")
            else:
                tmp = dict(cfg)
                tmp["chat_id"] = int(chat_id)
                _telegram_send_text(tmp, "⛔ Not authorized (phone number did not match).")
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
                saved = self._save_telegram_file(cfg, document.get("file_id"), prefer_name=document.get("file_name") or "image")
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
            entry = format_message(ts, "customer", "customer", "Telegram", body, msg_type="MESSAGE")
            cust_msg, cust_inbox, _ = mailbox_paths(self.root, "customer")
            mkdirp(cust_inbox)
            append_text(cust_msg, entry)
            atomic_write_text(cust_inbox / f"{ts_for_filename(ts)}_customer.md", entry)
        except Exception as e:
            log_line(self.root, f"[telegram] failed to mirror inbound to customer mailbox: {e}")

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
            out += (body or "(no body)")
            try:
                _telegram_send_text(cfg, out)
            except Exception as e:
                log_line(self.root, f"[telegram] send error: {e}")


# Router / watch loop
# -----------------------------

def parse_sender_and_type_from_message(text: str) -> Tuple[Optional[str], Optional[str]]:
    sender = None
    m = re.search(r"From:\s*([^\s→]+)", text)
    if m:
        sender = m.group(1).strip()
    m2 = re.search(r"\*\*Type:\*\*\s*([A-Z_]+)", text)
    msg_type = m2.group(1).strip() if m2 else None
    return sender, msg_type


def cmd_watch(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir)) or Path(args.workdir).expanduser().resolve()
    if not (root / STATE_FILENAME).exists():
        raise CTeamError("watch: could not find cteam.json in this directory or its parents")
    state = load_state(root)
    ensure_tmux_session(root, state)

    interval = max(0.5, float(args.interval))

    telegram_bridge: Optional[TelegramBridge] = None
    telegram_enabled = False

    seen_inbox: Dict[str, set[str]] = {}
    last_mail_sig: Dict[str, Tuple[int, int]] = {}
    nudge_queue = NudgeQueue(root)

    for a in state["agents"]:
        msg_path, inbox_dir, _ = mailbox_paths(root, a["name"])
        mkdirp(inbox_dir)
        seen_inbox[a["name"]] = set(p.name for p in inbox_dir.glob("*.md"))
        try:
            st = msg_path.stat()
            last_mail_sig[a["name"]] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            last_mail_sig[a["name"]] = (0, 0)

    log_line(root, f"[router] watching mailboxes under {root} (interval={interval}s)")

    now_ts = time.time()
    last_activity: Dict[str, float] = {a["name"]: now_ts for a in state["agents"]}
    while True:
        customer_nag_pending = False
        customer_state_for_nag: Optional[Dict[str, Any]] = None
        pm_idle_pending = False

        try:
            state = load_state(root)
        except Exception:
            pass

        # Start/stop Telegram bridge based on persisted state.
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
                last_activity[name] = time.time()

            if not new_files:
                continue

            last_txt = ""
            last_sender: Optional[str] = None
            last_type: Optional[str] = None
            last_ticket: Optional[str] = None
            last_body: str = ""
            for fp in new_files[-1:]:
                try:
                    last_txt = fp.read_text(encoding="utf-8", errors="replace")[:4000]
                except Exception:
                    last_txt = ""
                _, last_sender, _, _, last_body, last_ticket, last_type = _parse_entry(last_txt)

            if last_sender and last_type:
                maybe_start_agent_on_message(root, state, name, sender=last_sender, msg_type=last_type)

            reason = _mailbox_nudge_reason(last_ticket, last_type)
            nudge_queue.request(name, reason)

            if name == "pm" and (last_sender or "").lower() == "customer":
                if any(line.strip().lower() == "/tickets" for line in (last_body or "").splitlines()):
                    try:
                        store = load_tickets(root, state)
                        summary = _format_ticket_summary(store)
                        write_message(
                            root,
                            state,
                            sender="cteam",
                            recipient="customer",
                            subject="Ticket summary (open/blocked)",
                            body=summary,
                            msg_type="MESSAGE",
                            nudge=False,
                            start_if_needed=False,
                        )
                        try:
                            cust_state = load_customer_state(root)
                            cust_state["last_pm_reply_ts"] = now_iso()
                            save_customer_state(root, cust_state)
                        except Exception:
                            pass
                    except Exception:
                        pass

        # Customer wait loop: if last inbound from customer lacks a PM reply, keep nudging PM.
        try:
            cust_state = load_customer_state(root)
            last_in = iso_to_unix(str(cust_state.get("last_customer_msg_ts", "")))
            last_out = iso_to_unix(str(cust_state.get("last_pm_reply_ts", "")))
            last_nag = iso_to_unix(str(cust_state.get("last_pm_nag_ts", "")))
            if last_in and (not last_out or last_out < last_in):
                now_ts = time.time()
                should_nag = not last_nag or last_nag < last_in or (now_ts - last_nag) >= 60
                if should_nag:
                    nudge_queue.request("pm", reason="CUSTOMER WAITING — REPLY NOW")
                    customer_nag_pending = True
                    customer_state_for_nag = cust_state
        except Exception:
            pass

        # PM-level idle nudging: if all non-PM agents are idle >30s, ping PM.
        now = time.time()
        non_pm = [a["name"] for a in state["agents"] if a["name"] != "pm"]
        if non_pm:
            pm_idle = now - last_activity.get("pm", 0) >= 30
            team_idle = all(now - last_activity.get(n, 0) >= 30 for n in non_pm)
            if pm_idle and team_idle:
                if not state["tmux"].get("paused", False):
                    nudge_queue.request("pm", reason="TEAM IDLE — CHECK IN")
                    pm_idle_pending = True

        results = nudge_queue.flush(state)
        if results:
            for agent_name, ok, reason_text, skip_reason in results:
                if skip_reason == "duplicate":
                    log_line(root, f"[router] nudge skipped for {agent_name} (recent duplicate: {reason_text})")
                    continue
                if skip_reason == "busy":
                    log_line(root, f"[router] nudge deferred for {agent_name} (pane active)")
                    continue
                if ok:
                    log_line(root, f"[router] nudged {agent_name} ({reason_text})")
                else:
                    log_line(root, f"[router] could not nudge {agent_name} ({reason_text})")

                if ok and customer_nag_pending and agent_name == "pm" and "CUSTOMER WAITING" in reason_text:
                    try:
                        cust_state = customer_state_for_nag or load_customer_state(root)
                        cust_state["last_pm_nag_ts"] = now_iso()
                        save_customer_state(root, cust_state)
                    except Exception:
                        pass
                if ok and pm_idle_pending and agent_name == "pm" and "TEAM IDLE" in reason_text:
                    last_activity["pm"] = time.time()

        time.sleep(interval)


def cmd_customer_chat(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir)) or Path(args.workdir).expanduser().resolve()
    if not (root / STATE_FILENAME).exists():
        raise CTeamError("customer-chat: could not find cteam.json in this directory or its parents")
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
        atomic_write_text(chat_file, "# Customer channel\n\nUse this window to chat with the PM/team.\n\n")

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
    root = find_project_root(Path(args.workdir)) or Path(args.workdir).expanduser().resolve()
    if not (root / STATE_FILENAME).exists():
        raise CTeamError("chat: could not find cteam.json in this directory or its parents")
    state = load_state(root)

    history_path = root / DIR_RUNTIME / f"chat_{slugify(args.to)}.history"
    configure_readline(history_path)

    agent_names = {a["name"] for a in state["agents"]}
    allowed = set(agent_names) | {"customer"}
    recipient = args.to
    if recipient not in allowed:
        raise CTeamError(f"unknown recipient: {recipient}")

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
    root = find_project_root(Path(args.workdir)) or Path(args.workdir).expanduser().resolve()
    if not (root / STATE_FILENAME).exists():
        raise CTeamError("upload: could not find cteam.json in this directory or its parents")
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
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    state["tmux"]["paused"] = True
    save_state(root, state)

    session = state["tmux"]["session"]
    if tmux_has_session(session):
        # Move agent windows to standby shells.
        for agent in state["agents"]:
            try:
                try:
                    tmux_send_keys(session, agent["name"], ["Escape", "Escape", "Escape", "C-c"])
                except Exception:
                    pass
                cmd_args = standby_window_command(root, state, agent)
                tmux_respawn_window(session, agent["name"], Path(agent["dir_abs"]), command_args=cmd_args)
            except Exception:
                pass
        # Quiet router window.
        try:
            try:
                tmux_send_keys(session, ROUTER_WINDOW, ["Escape", "Escape", "Escape", "C-c"])
            except Exception:
                pass
            tmux_respawn_window(
                session,
                ROUTER_WINDOW,
                root,
                command_args=pick_shell() + ["printf 'cteam paused; router idle\\n'; exec $SHELL"],
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


def ensure_tmux(root: Path, state: Dict[str, Any], *, launch_codex: bool) -> None:
    ensure_tmux_session(root, state)
    ensure_router_window(root, state)
    ensure_customer_window(root, state)
    ensure_agent_windows(root, state, launch_codex=launch_codex)

    pm_agent = next((a for a in state["agents"] if a["name"] == "pm"), None)
    if pm_agent and launch_codex and not is_codex_running(state, "pm"):
        start_codex_in_window(root, state, pm_agent, boot=True)


def ensure_agent_windows(root: Path, state: Dict[str, Any], *, launch_codex: bool) -> None:
    session = state["tmux"]["session"]
    auto = set(autostart_agent_names(state))

    for agent in state["agents"]:
        w = agent["name"]
        agent_dir = Path(agent["dir_abs"])
        if launch_codex and w in auto:
            start_codex_in_window(root, state, agent, boot=True)
            continue

        windows = set(tmux_list_windows(session))
        if w not in windows:
            cmd_args = standby_window_command(root, state, agent)
            tmux_new_window(session, w, agent_dir, command_args=cmd_args)


# -----------------------------
# Commands
# -----------------------------

def _init_wizard(root: Path, state: Dict[str, Any], *, ask_telegram: bool = True) -> Dict[str, Any]:
    """Interactive init wizard: capture seed text and optionally configure Telegram."""
    history_path = root / DIR_RUNTIME / "init_wizard.history"
    configure_readline(history_path)

    seed_file = root / DIR_SEED / "SEED.md"
    if not seed_file.exists() or not seed_file.read_text(encoding="utf-8", errors="replace").strip():
        seed_txt = prompt_multiline("Paste customer/seed requirements (optional).", history_path=history_path)
        if seed_txt.strip():
            atomic_write_text(seed_file, "# Seed input\n\n" + seed_txt.strip() + "\n")

    extras_file = root / DIR_SEED_EXTRAS / "EXTRAS.md"
    if not extras_file.exists() or not extras_file.read_text(encoding="utf-8", errors="replace").strip():
        extras_txt = prompt_multiline("Paste seed-extras (links, research notes) (optional).", history_path=history_path)
        if extras_txt.strip():
            atomic_write_text(extras_file, "# Seed extras\n\n" + extras_txt.strip() + "\n")

    if ask_telegram and prompt_yes_no("Configure Telegram customer chat now?", default=False):
        token = prompt_line("Telegram bot token (from @BotFather)", default="")
        phone = prompt_line("Authorized phone number (digits; E.164 preferred, e.g. +15551234567)", default="")
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
            print("Telegram enabled. After startup, open your bot in Telegram and send /start, then share contact.")

    _readline_save_history(history_path)
    return state



def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.workdir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.force:
        raise CTeamError(f"directory not empty: {root} (use --force to reuse)")
    mkdirp(root)

    ensure_executable("git", hint="Install git")
    if not args.no_tmux:
        ensure_executable("tmux", hint="Install tmux")
    if not args.no_codex:
        if shutil.which(args.codex_cmd) is None:
            print(f"warning: codex executable not found: {args.codex_cmd} (windows will be shells)", file=sys.stderr)

    project_name = args.name or root.name
    state = build_state(
        root,
        project_name,
        devs=args.devs,
        mode="new",
        imported_from=None,
        codex_cmd=args.codex_cmd,
        codex_model=args.model,
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
        sender="cteam",
        recipient="pm",
        subject="Kickoff",
        body="Start by filling shared/GOALS.md and shared/PLAN.md (high-level). Track and assign all work via tickets (`cteam tickets ...`, `cteam assign ...`).",
        msg_type=ASSIGNMENT_TYPE,
        task="PM-KICKOFF",
        nudge=True,
        start_if_needed=True,
    )

    if args.no_tmux:
        print(f"Initialized workspace at {root}")
        print(f"Open tmux later: python3 cteam.py {shlex.quote(str(root))} open")
        return

    launch_codex = not args.no_codex and not state["tmux"].get("paused", False)
    ensure_tmux(root, state, launch_codex=launch_codex)
    print(f"tmux session: {state['tmux']['session']}")
    if not args.no_attach:
        tmux_attach(state["tmux"]["session"], window=args.window)


def cmd_import(args: argparse.Namespace) -> None:
    root = Path(args.workdir).expanduser().resolve()
    src = args.src

    if root.exists() and any(root.iterdir()) and not args.force:
        raise CTeamError(f"directory not empty: {root} (use --force to reuse)")
    mkdirp(root)

    ensure_executable("git", hint="Install git")
    if not args.no_tmux:
        ensure_executable("tmux", hint="Install tmux")
    if not args.no_codex:
        if shutil.which(args.codex_cmd) is None:
            print(f"warning: codex executable not found: {args.codex_cmd} (windows will be shells)", file=sys.stderr)

    project_name = args.name or (Path(src).name if src else root.name)
    state = build_state(
        root,
        project_name,
        devs=args.devs,
        mode="import",
        imported_from=src,
        codex_cmd=args.codex_cmd,
        codex_model=args.model,
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
    create_git_scaffold_import(root, state, src)
    ensure_agents_created(root, state)
    update_roster(root, state)
    save_state(root, state)

    state = load_state(root)
    try:
        write_message(
            root,
            state,
            sender="cteam",
            recipient="pm",
            subject="Import kickoff: infer goals + plan",
            body=(
                "This is an imported codebase.\n\n"
                "1) Inspect README/docs/code and fill shared/GOALS.md with inferred goals + open questions.\n"
                "2) Write shared/PLAN.md (high-level only).\n"
                "3) Create tickets for discovery tasks and assign them with `cteam tickets` + `cteam assign --ticket ...`.\n"
                "4) Keep agents coordinated; they will wait for ticketed assignments.\n"
            ),
            msg_type=ASSIGNMENT_TYPE,
            task="PM-IMPORT-KICKOFF",
            nudge=True,
            start_if_needed=True,
        )
    except Exception:
        pass

    if args.recon:
        for who, subj, task, body in [
            ("architect", "Recon: architecture map", "RECON-ARCH", "Create a short architecture map and send to PM. Avoid code changes."),
            ("tester", "Recon: how to run/tests", "RECON-TEST", "Figure out how to run the project and tests; write a short run/test note to PM."),
            ("researcher", "Recon: dependencies", "RECON-DEPS", "Inventory dependencies/versions; send a short risk/opportunity note to PM."),
        ]:
            if any(a["name"] == who for a in state["agents"]):
                write_message(
                    root,
                    state,
                    sender="cteam",
                    recipient=who,
                    subject=subj,
                    body=body,
                    msg_type=ASSIGNMENT_TYPE,
                    task=task,
                    nudge=True,
                    start_if_needed=True,
                )

    if args.no_tmux:
        print(f"Imported workspace at {root}")
        print(f"Open tmux later: python3 cteam.py {shlex.quote(str(root))} open")
        return

    launch_codex = not args.no_codex and not state["tmux"].get("paused", False)
    ensure_tmux(root, state, launch_codex=launch_codex)
    print(f"tmux session: {state['tmux']['session']}")
    if not args.no_attach:
        tmux_attach(state["tmux"]["session"], window=args.window)


def cmd_resume(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    if args.autostart:
        state["coordination"]["autostart"] = args.autostart
    if args.no_router:
        state["tmux"]["router"] = False
    if args.router:
        state["tmux"]["router"] = True
    save_state(root, state)

    new_tickets = create_root_structure(root, state)
    if not (root / DIR_PROJECT_BARE).exists() or not (root / DIR_PROJECT_BARE / "HEAD").exists():
        git_init_bare(root / DIR_PROJECT_BARE, branch=state["git"]["default_branch"])
    if not (root / DIR_PROJECT_CHECKOUT).exists():
        git_clone(root / DIR_PROJECT_BARE, root / DIR_PROJECT_CHECKOUT)

    ensure_agents_created(root, state)
    update_roster(root, state)
    state["tmux"]["paused"] = False
    save_state(root, state)

    if new_tickets and not (root / TICKET_MIGRATION_FLAG).exists():
        try:
            write_message(
                root,
                state,
                sender="cteam",
                recipient="pm",
                subject="Ticket system initialized",
                body=(
                    "A ticket database was created automatically (shared/TICKETS.json). "
                    "Please migrate any outstanding tasks/plan items into tickets and retire duplicate entries in legacy files. "
                    "Use `cteam tickets` to manage tickets and `cteam assign --ticket ...` for all assignments."
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

    launch_codex = not args.no_codex and not state["tmux"].get("paused", False)
    ensure_tmux(root, state, launch_codex=launch_codex)
    print(f"tmux session: {state['tmux']['session']}")
    if not args.no_attach:
        tmux_attach(state["tmux"]["session"], window=args.window)


def cmd_open(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    new_tickets = create_root_structure(root, state)
    if not (root / DIR_PROJECT_BARE).exists() or not (root / DIR_PROJECT_BARE / "HEAD").exists():
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
                sender="cteam",
                recipient="pm",
                subject="Ticket system initialized",
                body=(
                    "A ticket database was created automatically (shared/TICKETS.json). "
                    "Please migrate any outstanding tasks/plan items into tickets and retire duplicate entries in legacy files. "
                    "Use `cteam tickets` to manage tickets and `cteam assign --ticket ...` for all assignments."
                ),
                msg_type="MESSAGE",
                nudge=True,
                start_if_needed=True,
            )
            atomic_write_text(root / TICKET_MIGRATION_FLAG, now_iso())
        except Exception:
            pass
    if not args.no_tmux:
        launch_codex = not args.no_codex and not state["tmux"].get("paused", False)
        ensure_tmux(root, state, launch_codex=launch_codex)
        print(f"tmux session: {state['tmux']['session']}")
        if not args.no_attach:
            tmux_attach(state["tmux"]["session"], window=args.window)
    else:
        print(f"tmux disabled. Workspace at {root}")


def cmd_attach(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    ensure_executable("tmux")
    session = state["tmux"]["session"]
    if not tmux_has_session(session):
        raise CTeamError(f"tmux session not running: {session} (run: python3 cteam.py {root} open)")
    tmux_attach(session, window=args.window)


def cmd_kill(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)
    ensure_executable("tmux")
    tmux_kill_session(state["tmux"]["session"])
    print(f"killed tmux session: {state['tmux']['session']}")


def cmd_status(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    print(f"Project: {state['project_name']}")
    print(f"Root: {root}")
    print(f"Mode: {state.get('mode','new')}")
    print(f"tmux session: {state['tmux']['session']}")
    print(f"autostart: {state.get('coordination',{}).get('autostart')}")
    tg = state.get("telegram") or {}
    print(f"telegram: enabled={bool(tg.get('enabled', False))} config={tg.get('config_rel', TELEGRAM_CONFIG_REL)}")
    print("")

    for a in state["agents"]:
        a_dir = root / a["dir_rel"]
        status = a_dir / "STATUS.md"
        msg = a_dir / "message.md"
        s_mtime = _dt.datetime.fromtimestamp(status.stat().st_mtime) if status.exists() else None
        m_mtime = _dt.datetime.fromtimestamp(msg.stat().st_mtime) if msg.exists() else None
        print(f"== {a['name']} — {a['title']} ==")
        print(f"  status: {s_mtime.isoformat(timespec='seconds') if s_mtime else 'missing'}")
        print(f"  inbox : {m_mtime.isoformat(timespec='seconds') if m_mtime else 'missing'}")
        if status.exists():
            snippet = "\n".join(status.read_text(encoding="utf-8", errors="replace").splitlines()[:10])
            print("  --- STATUS snippet ---")
            print(textwrap.indent(snippet, "  "))
        print("")


def cmd_sync(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
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
                print(f"warning: fetch failed in integration checkout: {e}", file=sys.stderr)
        print("== Integration checkout (project/) ==")
        if args.pull:
            print("  pull:", git_try_pull_ff(checkout, branch=state["git"]["default_branch"]))
        print(textwrap.indent(git_status_short(checkout), "  "))
        print("")
    else:
        print("== Integration checkout (project/) ==\n  (missing)\n")

    if args.all:
        agent_map = {a["name"]: a for a in state["agents"]}
        targets = [p.strip() for p in args.agent.split(",")] if args.agent else list(agent_map.keys())
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
        raise CTeamError("could not find cteam.json in this directory or its parents")
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
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    ensure_shared_scaffold(root, state)
    install_self_into_root(root)
    ensure_agents_created(root, state)
    sync_cteam_into_agents(root, state)

    updated: List[str] = []
    for a in state["agents"]:
        a_dir = root / a["dir_rel"]
        atomic_write_text(a_dir / "AGENTS.md", render_agent_agents_md(state, a))
        repo_dir = root / a["repo_dir_rel"]
        try:
            safe_link_file(Path("..") / "AGENTS.md", repo_dir / "AGENTS.md")
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
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    cmd = args.ticket_cmd
    if not cmd:
        raise CTeamError("tickets: missing subcommand")

    if cmd == "list":
        store = load_tickets(root, state)
        tickets = store.get("tickets", [])
        if args.status:
            tickets = [t for t in tickets if t.get("status") == args.status]
        if args.assignee:
            tickets = [t for t in tickets if (t.get("assignee") or "").lower() == args.assignee.lower()]
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
        print(f"created ticket {ticket['id']} (assignee: {ticket.get('assignee') or 'unassigned'})")
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
                recipients=[ticket.get("assignee")],
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
        recipients = [ticket.get("assignee"), "pm"]
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
                recipients=[ticket.get("assignee")],
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
        recipients = [ticket.get("assignee"), "pm"]
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
                extra = f" blocked_on={h.get('blocked_on')}" if h_type == "blocked" else ""
                print(f"- {h_ts} {h_type}{from_to}{extra} — {h.get('user','')} {note}")
        return

    raise CTeamError(f"unknown tickets subcommand: {cmd}")


def cmd_msg(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    body = args.text
    if args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    if not body.strip():
        raise CTeamError("message body is empty (provide TEXT or --file)")

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
        if agent and not is_codex_running(state, args.to):
            start_codex_in_window(root, state, agent, boot=False)

    print(f"sent to {args.to}")


def cmd_broadcast(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    body = args.text
    if args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    if not body.strip():
        raise CTeamError("message body is empty (provide TEXT or --file)")

    sender = args.sender or "pm"
    subject = args.subject or "broadcast"
    for a in state["agents"]:
        write_message(
            root,
            state,
            sender=sender,
            recipient=a["name"],
            subject=subject,
            body=body,
            msg_type="MESSAGE",
            nudge=not args.no_nudge,
            start_if_needed=args.start_if_needed,
        )
    if args.start_if_needed:
        ensure_tmux_session(root, state)
        ensure_router_window(root, state)
        for a in state["agents"]:
            name = a["name"]
            if not is_codex_running(state, name):
                start_codex_in_window(root, state, a, boot=False)

    print("broadcast sent")


def cmd_assign(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    body = args.text
    if args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    if not body.strip():
        raise CTeamError("assignment body is empty (provide TEXT or --file)")

    sender = args.sender or "pm"
    ticket: Optional[Dict[str, Any]] = None
    created = False
    ticket_id = (args.ticket or "").strip() if hasattr(args, "ticket") else ""

    if ticket_id:
        store = load_tickets(root, state)
        ticket = _find_ticket(store, ticket_id)
        if not ticket:
            raise CTeamError(f"ticket not found: {ticket_id}")
    else:
        if not args.title or not args.desc:
            raise CTeamError("assignment requires --ticket or --title/--desc to auto-create a ticket")
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
    subject = args.subject or f"{ticket_id} — {ticket.get('title','')}"

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
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    targets: List[str]
    if args.to == "all":
        targets = [a["name"] for a in state["agents"]]
    else:
        targets = [p.strip() for p in args.to.split(",") if p.strip()]
    for t in targets:
        ok = nudge_agent(root, state, t, reason=args.reason or "NUDGE", interrupt=args.interrupt)
        print(f"{t}: {'nudged' if ok else 'could not nudge'}")


def cmd_restart(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
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
        start_codex_in_window(root, state, agent, boot=(name == "pm"))
        print(f"restarted codex in window: {name}")


def notify_pm_new_agent(root: Path, state: Dict[str, Any], agent: Dict[str, Any]) -> None:
    """
    Inform the PM that a new agent was added and remind them to balance workloads.
    """
    if not any(a["name"] == "pm" for a in state["agents"]):
        return
    role = agent.get("role", "agent").replace("_", " ")
    subject = f"New agent added: {agent.get('name','(unknown)')} ({agent.get('title','')})"
    lines = [
        f"A new {role} joined: **{agent.get('name','')}** — {agent.get('title','')}.",
        "Please balance work across the team: onboard them, delegate tasks, and adjust the plan if needed.",
        "Keep assignments explicit (`Type: ASSIGNMENT`) and update PLAN/tickets to reflect the new capacity.",
    ]
    body = "\n".join(lines)
    try:
        write_message(
            root,
            state,
            sender="cteam",
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


def cmd_add_agent(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    role = args.role
    if role == "project_manager" or (args.name and args.name.strip().lower() == "pm"):
        raise CTeamError("cannot add another project manager; workspace already has a PM")
    name = args.name
    if not name:
        existing = {a["name"] for a in state["agents"]}
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
        raise CTeamError(f"agent already exists: {name}")

    title = args.title or role.replace("_", " ").title()
    agent = {
        "name": name,
        "role": role,
        "title": title,
        "persona": args.persona or role_persona(role),
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

    if not args.no_tmux:
        ensure_tmux_session(root, state)
        cmd_args = standby_window_command(root, state, agent)
        tmux_new_window(state["tmux"]["session"], name, Path(agent["dir_abs"]), command_args=cmd_args)
        if args.start_codex:
            start_codex_in_window(root, state, agent, boot=False)

    notify_pm_new_agent(root, state, agent)
    print(f"added agent: {name} ({title}) role={role}")


def cmd_remove_agent(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    name = args.name.strip()
    if name == "pm":
        raise CTeamError("cannot remove the project manager")

    agent = next((a for a in state["agents"] if a["name"] == name), None)
    if not agent:
        raise CTeamError(f"agent not found: {name}")

    # Best-effort cleanup before state mutation.
    _kill_agent_window_if_present(state, name)
    _archive_or_remove_agent_dirs(root, agent, purge=args.purge)
    _unassign_tickets_for_agent(root, state, name)
    _nudge_history.pop(name, None)

    state["agents"] = [a for a in state["agents"] if a["name"] != name]
    compute_agent_abspaths(state)
    update_roster(root, state)
    save_state(root, state)

    print(f"removed agent: {name}")


def cmd_doc_walk(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
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
            ("architect", "Doc-walk: architecture doc", "DOC-ARCH", "Create docs/architecture.md and a decisions summary. Coordinate with PM."),
            ("tester", "Doc-walk: run/test guide", "DOC-TEST", "Create docs/runbook.md with how to run and test. Coordinate with PM."),
            ("researcher", "Doc-walk: dependencies inventory", "DOC-DEPS", "Create shared/research/deps.md with dependencies and risks. Coordinate with PM."),
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
    root = find_project_root(Path(args.workdir)) or Path(args.workdir).expanduser().resolve()
    if not (root / STATE_FILENAME).exists():
        raise CTeamError("telegram-configure: could not find cteam.json in this directory or its parents")
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
            phone = prompt_line("Authorized phone number (digits; E.164 preferred, e.g. +15551234567)", default="")

    token = token.strip()
    phone = phone.strip()
    if not token or not phone:
        raise CTeamError("telegram-configure: need --token and --phone (or run interactively)")

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
    print(f"   python3 cteam.py {shlex.quote(str(root))} telegram-enable")


def cmd_telegram_enable(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir)) or Path(args.workdir).expanduser().resolve()
    if not (root / STATE_FILENAME).exists():
        raise CTeamError("telegram-enable: could not find cteam.json in this directory or its parents")
    state = load_state(root)

    _ = telegram_load_config(root, state)  # validate

    state.setdefault("telegram", {})
    state["telegram"]["enabled"] = True

    if not state.get("tmux", {}).get("router", True):
        state["tmux"]["router"] = True

    save_state(root, state)

    print("Telegram integration enabled.")
    print("If the router is running (`cteam <workdir> watch`), it will start bridging immediately.")
    print("If not, open/start the workspace (tmux router runs the watcher):")
    print(f"  python3 cteam.py {shlex.quote(str(root))} open")


def cmd_telegram_disable(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir)) or Path(args.workdir).expanduser().resolve()
    if not (root / STATE_FILENAME).exists():
        raise CTeamError("telegram-disable: could not find cteam.json in this directory or its parents")
    state = load_state(root)

    state.setdefault("telegram", {})
    state["telegram"]["enabled"] = False
    save_state(root, state)
    print("Telegram integration disabled.")


# -----------------------------
# CLI
# -----------------------------

class CTeamHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Readable help with defaults and preserved newlines."""


class CTeamArgumentParser(argparse.ArgumentParser):
    """
    Custom parser that prints full help on usage errors for a friendlier UX.
    """

    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\ncteam error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = CTeamArgumentParser(
        prog="cteam",
        add_help=True,
        description=(
            "Clanker Team (cteam) orchestrates tmux + Codex agents around a shared git repo.\n"
            "Common flow: init/import a workspace, open tmux, assign work, chat with agents, and keep the customer updated."
        ),
        epilog=(
            "Popular: init/import/open/resume, msg/assign/broadcast/nudge, watch, add-agent/remove-agent, customer-chat.\n"
            "Git/ops: sync, status, seed-sync, update-workdir, restart, doc-walk, telegram-*. "
            "Run `cteam <workdir> <command> --help` for details. Full guide: README.md."
        ),
        formatter_class=CTeamHelpFormatter,
    )
    p.add_argument("workdir", help="Workspace directory or any path inside it.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_workspace(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--no-tmux", action="store_true", help="Do not start/manage tmux.")
        pp.add_argument("--no-codex", action="store_true", help="Do not launch Codex in windows.")
        pp.add_argument("--no-attach", action="store_true", help="Do not attach to tmux after starting.")
        pp.add_argument("--attach-late", action="store_true", help="Attach after tmux/windows/Codex are ready.")
        pp.add_argument("--window", help="Window name to attach/select (e.g., pm).")

    def add_codex_flags(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--codex-cmd", default=DEFAULT_CODEX_CMD, help="Codex executable name/path.")
        pp.add_argument("--model", default=None, help="Codex model (if supported).")
        pp.add_argument("--sandbox", default=DEFAULT_CODEX_SANDBOX, choices=["read-only","workspace-write","danger-full-access"])
        pp.add_argument("--ask-for-approval", default=DEFAULT_CODEX_APPROVAL, choices=["untrusted","on-failure","on-request","never"])
        pp.add_argument("--no-search", action="store_true", help="Disable Codex --search (if supported).")
        pp.add_argument("--full-auto", action="store_true", help="Use Codex --full-auto (if supported).")
        pp.add_argument("--yolo", action="store_true", help="Use Codex --yolo (if supported). DANGEROUS.")

    def add_coord_flags(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--autostart", default=DEFAULT_AUTOSTART, choices=["pm","pm+architect","all"],
                        help="Which agents start Codex immediately (PM-led coordination).")
        pp.add_argument("--no-router", action="store_true", help="Disable router/watch window.")

    p_init = sub.add_parser("init", help="Initialize a new cteam workspace (empty repo).")
    add_common_workspace(p_init)
    p_init.add_argument("--no-interactive", action="store_true", help="Disable interactive init wizard.")
    p_init.add_argument("--name", help="Project name (default: folder name).")
    p_init.add_argument("--devs", type=int, default=1, help="Number of developer agents.")
    p_init.add_argument("--force", action="store_true", help="Reuse non-empty directory.")
    add_codex_flags(p_init)
    add_coord_flags(p_init)
    p_init.set_defaults(func=cmd_init)

    p_imp = sub.add_parser("import", help="Import an existing git repo or directory into a new workspace.")
    add_common_workspace(p_imp)
    p_imp.add_argument("--src", required=True, help="Git repo URL/path OR local directory to import.")
    p_imp.add_argument("--name", help="Project name override.")
    p_imp.add_argument("--devs", type=int, default=1)
    p_imp.add_argument("--force", action="store_true")
    p_imp.add_argument("--recon", action="store_true",
                       help="Also send safe coordinated recon assignments (no code changes) to non-PM agents.")
    add_codex_flags(p_imp)
    add_coord_flags(p_imp)
    p_imp.set_defaults(func=cmd_import)

    p_resume = sub.add_parser("resume", help="Resume/manage an existing workspace (ensure dirs/windows exist).")
    add_common_workspace(p_resume)
    p_resume.add_argument("--autostart", choices=["pm","pm+architect","all"], help="Override autostart in state.")
    p_resume.add_argument("--router", action="store_true", help="Force-enable router.")
    p_resume.add_argument("--no-router", action="store_true", help="Disable router.")
    p_resume.set_defaults(func=cmd_resume)

    p_open = sub.add_parser("open", help="Resume and attach (open tmux in your terminal).")
    add_common_workspace(p_open)
    p_open.add_argument("--no-router", action="store_true")
    p_open.set_defaults(func=cmd_open)

    p_attach = sub.add_parser("attach", help="Attach to tmux session only.")
    add_common_workspace(p_attach)
    p_attach.set_defaults(func=cmd_attach)

    p_kill = sub.add_parser("kill", help="Kill the tmux session.")
    add_common_workspace(p_kill)
    p_kill.set_defaults(func=cmd_kill)

    p_watch = sub.add_parser("watch", help="Router: watch mailboxes and nudge/start agents in tmux.")
    p_watch.add_argument("--interval", type=float, default=1.5)
    p_watch.set_defaults(func=cmd_watch)

    p_pause = sub.add_parser("pause", help="Pause tmux: park agent windows and mark workspace paused.")
    p_pause.set_defaults(func=cmd_pause)

    p_chat = sub.add_parser("chat", help="Interactive chat with an agent or customer.")
    p_chat.add_argument("--to", required=True, help="Recipient agent name or 'customer'.")
    p_chat.add_argument("--sender", help="Override sender name (default: human).")
    p_chat.add_argument("--subject", help="Optional subject.")
    p_chat.add_argument("--no-nudge", action="store_true")
    p_chat.add_argument("--start-if-needed", action="store_true")
    p_chat.set_defaults(func=cmd_chat)

    p_upload = sub.add_parser("upload", help="Copy files/dirs into shared/drive for sharing with the team.")
    p_upload.add_argument("paths", nargs="+", help="Files or directories to copy into shared/drive.")
    p_upload.add_argument("--dest", help="Optional destination name/path under shared/drive (default: use source name).")
    p_upload.add_argument("--from", dest="sender", help="Sender name for PM notification (default: human).")
    p_upload.set_defaults(func=cmd_upload)

    p_cust = sub.add_parser("customer-chat", help="Run an interactive customer chat window (tmux-friendly).")
    p_cust.set_defaults(func=cmd_customer_chat)

    p_status = sub.add_parser("status", help="Show workspace + agent status.")
    p_status.set_defaults(func=cmd_status)

    p_sync = sub.add_parser("sync", help="Fetch/pull and show git statuses.")
    p_sync.add_argument("--all", action="store_true", help="Also show agent repo statuses.")
    p_sync.add_argument("--agent", help="Limit to specific agent(s) with --all.")
    p_sync.add_argument("--fetch", action="store_true")
    p_sync.add_argument("--pull", action="store_true")
    p_sync.add_argument("--no-show-branches", action="store_false", dest="show_branches")
    p_sync.set_defaults(show_branches=True)
    p_sync.set_defaults(func=cmd_sync)

    p_seed = sub.add_parser("seed-sync", help="Copy seed/ and seed-extras/ into agent workdirs.")
    p_seed.add_argument("--clean", action="store_true")
    p_seed.set_defaults(func=cmd_seed_sync)

    p_update = sub.add_parser("update-workdir", help="Refresh agent AGENTS.md files from current cteam templates.")
    p_update.set_defaults(func=cmd_update_workdir)

    p_tickets = sub.add_parser("tickets", help="Ticket management (list/create/assign/block/reopen/close/show).")
    tickets_sub = p_tickets.add_subparsers(dest="ticket_cmd")

    p_t_list = tickets_sub.add_parser("list", help="List tickets.")
    p_t_list.add_argument("--status", choices=[TICKET_STATUS_OPEN, TICKET_STATUS_BLOCKED, TICKET_STATUS_CLOSED])
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
    p_t_create.add_argument("--assign-note", help="Optional note for initial assignment")
    p_t_create.set_defaults(func=cmd_tickets)

    p_t_assign = tickets_sub.add_parser("assign", help="Assign an existing ticket.")
    p_t_assign.add_argument("--id", required=True)
    p_t_assign.add_argument("--assignee", required=True)
    p_t_assign.add_argument("--user", help="User performing the assignment (default: pm)")
    p_t_assign.add_argument("--note")
    p_t_assign.add_argument("--notify", action="store_true", help="Also notify the assignee.")
    p_t_assign.set_defaults(func=cmd_tickets)

    p_t_block = tickets_sub.add_parser("block", help="Mark a ticket as blocked.")
    p_t_block.add_argument("--id", required=True)
    p_t_block.add_argument("--on", required=True, help="Ticket ID or external issue causing the block.")
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
    p_msg.add_argument("--file")
    p_msg.add_argument("--no-nudge", action="store_true")
    p_msg.add_argument("--start-if-needed", action="store_true",
                       help="If recipient is not running Codex, start it in their tmux window.")
    p_msg.add_argument("--no-follow", action="store_true", help="Do not select recipient window after sending.")
    p_msg.add_argument("--ticket", help="Ticket ID to link this message.")
    p_msg.add_argument("text", nargs="?", default="")
    p_msg.set_defaults(func=cmd_msg)

    p_b = sub.add_parser("broadcast", help="Broadcast a message to all agents.")
    p_b.add_argument("--from", dest="sender")
    p_b.add_argument("--subject")
    p_b.add_argument("--file")
    p_b.add_argument("--no-nudge", action="store_true")
    p_b.add_argument("--start-if-needed", action="store_true", help="Start Codex in agent windows if needed.")
    p_b.add_argument("text", nargs="?", default="")
    p_b.set_defaults(func=cmd_broadcast)

    p_assign = sub.add_parser("assign", help="Send an ASSIGNMENT to an agent (starts them if needed).")
    p_assign.add_argument("--to", required=True)
    p_assign.add_argument("--task", help="Task ID (e.g., T001).")
    p_assign.add_argument("--from", dest="sender")
    p_assign.add_argument("--subject")
    p_assign.add_argument("--file")
    p_assign.add_argument("--no-nudge", action="store_true")
    p_assign.add_argument("--ticket", help="Existing ticket ID; if omitted, requires --title and --desc.")
    p_assign.add_argument("--title", help="Title when auto-creating a ticket.")
    p_assign.add_argument("--desc", help="Description when auto-creating a ticket.")
    p_assign.add_argument("--assign-note", help="Note to record on assignment.")
    p_assign.add_argument("--tags", help="Comma-separated tags when auto-creating a ticket.")
    p_assign.add_argument("text", nargs="?", default="")
    p_assign.add_argument("--no-follow", action="store_true", help="Do not select recipient window after sending.")
    p_assign.set_defaults(func=cmd_assign)

    p_nudge = sub.add_parser("nudge", help="Send a manual nudge to an agent window.")
    p_nudge.add_argument("--to", default="pm", help="agent(s) comma-separated or 'all'")
    p_nudge.add_argument("--reason", default="NUDGE")
    p_nudge.add_argument("--no-follow", action="store_true", help="Do not select the target window.")
    p_nudge.add_argument("--interrupt", action="store_true", help="Send an interrupting nudge (sends Escape before the message).")
    p_nudge.set_defaults(func=cmd_nudge)

    p_restart = sub.add_parser("restart", help="Restart Codex in agent tmux windows (respawn).")
    p_restart.add_argument("--window", help="agent window name(s) comma-separated or 'all'")
    p_restart.add_argument("--hard", action="store_true", help="send Ctrl-C before respawn (best-effort)")
    p_restart.set_defaults(func=cmd_restart)

    p_add = sub.add_parser("add-agent", help="Add a new agent to an existing workspace (one PM only).")
    p_add.add_argument(
        "--role",
        default="developer",
        choices=["developer","tester","researcher","architect"],
    )
    p_add.add_argument("--name")
    p_add.add_argument("--title")
    p_add.add_argument("--persona")
    p_add.add_argument("--no-tmux", action="store_true")
    p_add.add_argument("--start-codex", action="store_true", help="Start codex immediately for the new agent.")
    p_add.set_defaults(func=cmd_add_agent)

    p_rm = sub.add_parser("remove-agent", help="Remove an agent from the workspace (non-PM only).")
    p_rm.add_argument("--name", required=True, help="Agent name to remove (not pm).")
    p_rm.add_argument("--purge", action="store_true", help="Delete agent dirs/mail instead of archiving under _removed/.")
    p_rm.set_defaults(func=cmd_remove_agent)

    p_doc = sub.add_parser("doc-walk", help="Kick off a documentation sprint over the repo (PM-led).")
    p_doc.add_argument("--from", dest="sender")
    p_doc.add_argument("--subject")
    p_doc.add_argument("--task")
    p_doc.add_argument("--auto", action="store_true", help="Also assign initial doc tasks to architect/tester/researcher.")
    p_doc.set_defaults(func=cmd_doc_walk)


    p_tg_cfg = sub.add_parser("telegram-configure", help="Configure Telegram bot credentials for customer chat.")
    p_tg_cfg.add_argument("--token", help="Bot token from @BotFather (stored locally).")
    p_tg_cfg.add_argument("--phone", help="Authorized phone number (digits; E.164 preferred).")
    p_tg_cfg.add_argument("--no-interactive", action="store_true", help="Do not prompt (require flags).")
    p_tg_cfg.set_defaults(func=cmd_telegram_configure)

    p_tg_en = sub.add_parser("telegram-enable", help="Enable Telegram customer chat bridge (requires prior configure).")
    p_tg_en.set_defaults(func=cmd_telegram_enable)

    p_tg_dis = sub.add_parser("telegram-disable", help="Disable Telegram customer chat bridge.")
    p_tg_dis.set_defaults(func=cmd_telegram_disable)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
        return 0
    except CTeamError as e:
        print(f"cteam error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
