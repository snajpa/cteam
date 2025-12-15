#!/usr/bin/env python3
"""
cteam — single-file multi-agent manager for tmux + OpenAI Codex CLI + git repos.

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
- A router/watch loop (`cteam watch`) runs inside tmux in a `router` window:
  - watches inbox directories and mailbox changes
  - nudges recipients
  - can auto-start Codex for agents when assignments arrive

Major commands
- init, import, resume, open, attach, kill
- msg, broadcast, assign, nudge
- watch (router)
- status, sync, seed-sync
- add-agent, restart
- doc-walk (kick off documentation sprint)

External deps
- tmux, git, codex (Codex CLI). Everything else is Python stdlib.

Security note
- The defaults minimize Codex approval prompts by using:
    --ask-for-approval never
  and a permissive sandbox default:
    --sandbox danger-full-access
  Adjust via CLI flags if you want more safety.

"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Constants / layout
# -----------------------------

STATE_FILENAME = "cteam.json"
STATE_VERSION = 4

DIR_PROJECT_BARE = "project.git"
DIR_PROJECT_CHECKOUT = "project"
DIR_SEED = "seed"
DIR_SEED_EXTRAS = "seed-extras"
DIR_SHARED = "shared"
DIR_AGENTS = "agents"
DIR_LOGS = "logs"
DIR_MAIL = f"{DIR_SHARED}/mail"
DIR_RUNTIME = f"{DIR_SHARED}/runtime"

ROUTER_WINDOW = "router"

DEFAULT_GIT_BRANCH = "main"

DEFAULT_CODEX_CMD = "codex"
DEFAULT_CODEX_SANDBOX = "danger-full-access"   # read-only | workspace-write | danger-full-access
DEFAULT_CODEX_APPROVAL = "never"               # untrusted | on-failure | on-request | never
DEFAULT_CODEX_SEARCH = True

DEFAULT_AUTOSTART = "all"  # pm | pm+architect | all

# For robust "assignment" detection
ASSIGNMENT_TYPE = "ASSIGNMENT"


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


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "project"


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
            "assignment_from": ["pm", "human", "cteam"],   # who can "start" an agent by messaging them
        },
        "tmux": {
            "session": session,
            "router": bool(router),
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
        state["version"] = STATE_VERSION
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
        coord.setdefault("assignment_from", ["pm", "human", "cteam"])
        state["coordination"] = coord

        tm = state.get("tmux", {})
        tm.setdefault("session", f"cteam_{slugify(state.get('project_name', root.name))}")
        tm.setdefault("router", True)
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
          - `python3 cteam.py assign . --to dev1 --task T001 --subject "..." "instructions..."`

        ## Router
        - A tmux window named `{ROUTER_WINDOW}` runs `cteam watch`:
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

        Suggested structure:
        - Goals (link to GOALS.md)
        - Milestones
        - Work breakdown (task IDs)
        - Risks & mitigations
        - Definition of Done
        """
    )


def render_tasks_template() -> str:
    return textwrap.dedent(
        """\
        # TASKS

        Owned by: **PM**

        Format suggestion:
        - [ ] T001 — ...
          - Owner: dev1
          - Depends on: ...
          - Acceptance criteria: ...
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
        # Shared workspace (cteam)

        Project: **{state['project_name']}**
        Root: `{state['root_abs']}`

        This directory is outside the git repo; it's for coordination artifacts.

        Key files:
        - GOALS.md, PLAN.md, TASKS.md
        - DECISIONS.md, TIMELINE.md
        - PROTOCOL.md

        Mail:
        - shared/mail/<agent>/message.md (append-only mailbox)

        Router:
        - tmux window `{ROUTER_WINDOW}` runs `python3 cteam.py watch .`
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

    common = textwrap.dedent(
        f"""\
        # AGENTS.md — {agent['name']} ({agent['title']})

        You are a Codex agent in a multi-agent team managed by **cteam**.

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

        ## Where to work
        - Your repo clone: `proj/` (work here)
        - Your inbox: `message.md` (check frequently)
        - Update: `STATUS.md`

        Shared coordination:
        - `shared/GOALS.md`, `shared/PLAN.md`, `shared/TASKS.md`, `shared/DECISIONS.md`
        - `shared/TEAM_ROSTER.md` (who is on the team)

        ## Messaging
        Preferred: use cteam CLI so delivery+nudge is reliable:

        - Send message:
          - `python3 cteam.py msg $CTEAM_ROOT --to pm --subject "..." "text..."`
        - Assignments (PM/human):
          - `python3 cteam.py assign $CTEAM_ROOT --to dev1 --task T001 --subject "..." "instructions"`

        If you must write manually, append to:
        - `shared/mail/<agent>/message.md`

        ## Git rules
        - Branch naming: `agent/{agent['name']}/<topic>`
        - Push early, push often.
        - Avoid rewriting shared history.
        """
    )

    if role == "project_manager":
        block = textwrap.dedent(
            f"""\
            ## Role: Project Manager (the coordinator)
            You are responsible for:
            - inferring/clarifying goals
            - producing the plan and task breakdown
            - assigning work to other agents
            - keeping everyone in sync
            - ensuring the project is completed

            On startup:
            1) Read seed/ (if any) and repo README/docs.
            2) Fill `shared/GOALS.md` with assumptions + questions.
            3) Write `shared/PLAN.md` and `shared/TASKS.md`.
            4) Start assigning work using `cteam assign`.

            Important:
            - Keep non-PM agents focused on one assignment at a time.
            - Ask for short status updates; require each agent to keep `STATUS.md` updated.
            """
        )
    elif role == "architect":
        block = textwrap.dedent(
            """\
            ## Role: Architect
            - Produce/maintain architecture docs.
            - Record major decisions in shared/DECISIONS.md.
            - Coordinate with PM before making broad changes.

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

            Default behavior if unassigned:
            - scan repo quickly, propose a task to PM
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
    if mode == "import":
        return (
            "You are the Project Manager. First open message.md for any kickoff notes. "
            "Then immediately infer what this codebase is for. "
            "Open shared/GOALS.md and fill it with inferred goals, non-goals, and questions. "
            "Then write shared/PLAN.md + shared/TASKS.md, and assign first tasks using `cteam assign`. "
            "Keep everyone coordinated; do not let others work unassigned. "
            "Start now."
        )
    return (
        "You are the Project Manager. First open message.md for any kickoff notes. "
        "Then read seed/ and write shared/GOALS.md + shared/PLAN.md + shared/TASKS.md. "
        "Then assign tasks to the team using `cteam assign`. Keep everyone coordinated. Start now."
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
        cteam — {state['project_name']}
        Agent window: {agent['name']} ({agent['title']})
        Mode: {state.get('mode','new')}

        Standby mode (PM-led coordination):
        - Waiting for PM/human assignment.
        - Your inbox: {DIR_MAIL}/{agent['name']}/message.md
        - If you are assigned work, this window will be repurposed to run Codex automatically.

        Tip: If you want to manually check mail:
          less -R {shlex.quote(str(Path(state['root_abs'])/DIR_MAIL/agent['name']/ 'message.md'))}
        """
    ).strip()


# -----------------------------
# Workspace creation
# -----------------------------

def ensure_shared_scaffold(root: Path, state: Dict[str, Any]) -> None:
    mkdirp(root / DIR_LOGS)
    mkdirp(root / DIR_SEED)
    mkdirp(root / DIR_SEED_EXTRAS)
    mkdirp(root / DIR_SHARED)
    mkdirp(root / DIR_AGENTS)
    mkdirp(root / DIR_RUNTIME)

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
    if (agent_dir / "seed").exists():
        safe_link_dir(Path("..") / "seed", repo_dir / "seed")
    if (agent_dir / "seed-extras").exists():
        safe_link_dir(Path("..") / "seed-extras", repo_dir / "seed-extras")

    if (root / "cteam.py").exists():
        safe_link_file(Path("..") / ".." / ".." / "cteam.py", repo_dir / "cteam.py")


def ensure_agents_created(root: Path, state: Dict[str, Any]) -> None:
    for a in state["agents"]:
        create_agent_dirs(root, state, a)


def update_roster(root: Path, state: Dict[str, Any]) -> None:
    atomic_write_text(root / DIR_SHARED / "TEAM_ROSTER.md", render_team_roster(state))


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
) -> str:
    subject_line = subject.strip() if subject.strip() else "(no subject)"
    task_line = f"**Task:** {task}\n" if task else ""
    return textwrap.dedent(
        f"""\
        ## {ts} — From: {sender} → To: {recipient}
        **Type:** {msg_type}
        **Subject:** {subject_line}
        {task_line}
        {body.rstrip()}

        ---
        """
    )


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
) -> None:
    ts = now_iso()
    entry = format_message(ts, sender, recipient, subject, body, msg_type=msg_type, task=task)
    if not entry.endswith("\n"):
        entry += "\n"

    agent_names = {a["name"] for a in state["agents"]}
    if recipient not in agent_names:
        raise CTeamError(f"unknown recipient agent: {recipient}")

    msg_path, inbox_dir, _ = mailbox_paths(root, recipient)
    mkdirp(inbox_dir)

    atomic_write_text(inbox_dir / f"{ts_for_filename(ts)}_{sender}.md", entry)

    append_text(msg_path, entry)

    append_text(root / DIR_SHARED / "MESSAGES.log.md", entry)

    if msg_type == ASSIGNMENT_TYPE:
        append_text(root / DIR_SHARED / "ASSIGNMENTS.log.md", entry)

    # Save a copy in sender outbox (if sender is a known agent) for local context.
    if sender in agent_names:
        _, _, outbox_dir = mailbox_paths(root, sender)
        mkdirp(outbox_dir)
        atomic_write_text(outbox_dir / f"{ts_for_filename(ts)}_{recipient}.md", entry)

    rec = next(a for a in state["agents"] if a["name"] == recipient)
    agent_dir_msg = root / rec["dir_rel"] / "message.md"
    repo_msg = root / rec["repo_dir_rel"] / "message.md"
    if agent_dir_msg.exists() and not samefile(agent_dir_msg, msg_path):
        append_text(agent_dir_msg, entry)
    if repo_msg.exists() and not samefile(repo_msg, msg_path):
        append_text(repo_msg, entry)

    if not nudge:
        return

    if start_if_needed:
        maybe_start_agent_on_message(root, state, recipient, sender=sender, msg_type=msg_type)

    nudge_agent(root, state, recipient, reason="MAILBOX UPDATED")


# -----------------------------
# Starting / nudging agents in tmux
# -----------------------------

def ensure_tmux_session(root: Path, state: Dict[str, Any]) -> None:
    ensure_executable("tmux", hint="Install tmux (apt/brew).")
    session = state["tmux"]["session"]
    if not tmux_has_session(session):
        tmux_new_session(session, root)
    tmux(["set-option", "-t", session, "remain-on-exit", "on"], capture=True, check=False)
    tmux(["set-option", "-t", session, "allow-rename", "off"], capture=True, check=False)


def router_window_command(root: Path) -> List[str]:
    shell = pick_shell()
    cmd = f"cd {shlex.quote(str(root))} && exec python3 cteam.py watch ."
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
            first_prompt = (
                f"You are {agent['title']} ({agent['name']}). "
                f"Open AGENTS.md and message.md. "
                f"If you do not have an assignment (Type: {ASSIGNMENT_TYPE}), do NOT start coding; "
                f"send a short recon note to PM, then wait."
            )
    else:
        first_prompt = prompt_on_mail(agent["name"])

    wait_for_pane_command(session, w, "codex", timeout=4.0)
    wait_for_pane_quiet(session, w, quiet_for=0.8, timeout=6.0)
    tmux_send_keys(session, w, ["C-u"])
    tmux_send_line_when_quiet(session, w, first_prompt)


def nudge_agent(root: Path, state: Dict[str, Any], agent_name: str, *, reason: str = "MAILBOX UPDATED") -> bool:
    session = state["tmux"]["session"]
    if not tmux_has_session(session):
        return False
    windows = set(tmux_list_windows(session))
    if agent_name not in windows:
        return False

    msg = f"{reason}: open message.md and act. If assigned, proceed; update STATUS.md."
    tmux_send_keys(session, agent_name, ["C-u"])
    if is_codex_running(state, agent_name):
        wait_for_pane_quiet(session, agent_name, quiet_for=0.8, timeout=4.0)
        return tmux_send_line_when_quiet(session, agent_name, msg)
    else:
        return tmux_send_line_when_quiet(session, agent_name, f"echo {shlex.quote(msg)}")


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

    allowed_from = set(coord.get("assignment_from", ["pm", "human", "cteam"]))
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

    seen_inbox: Dict[str, set[str]] = {}
    last_mail_sig: Dict[str, Tuple[int, int]] = {}

    for a in state["agents"]:
        msg_path, inbox_dir, _ = mailbox_paths(root, a["name"])
        mkdirp(inbox_dir)
        seen_inbox[a["name"]] = set(p.name for p in inbox_dir.glob("*.md"))
        try:
            st = msg_path.stat()
            last_mail_sig[a["name"]] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            last_mail_sig[a["name"]] = (0, 0)

    print(f"[router] watching mailboxes under {root} (interval={interval}s)")
    sys.stdout.flush()

    while True:
        try:
            state = load_state(root)
        except Exception:
            pass

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

            if not new_files:
                continue

            last_txt = ""
            last_sender: Optional[str] = None
            last_type: Optional[str] = None
            for fp in new_files[-1:]:
                try:
                    last_txt = fp.read_text(encoding="utf-8", errors="replace")[:4000]
                except Exception:
                    last_txt = ""
                last_sender, last_type = parse_sender_and_type_from_message(last_txt)

            if last_sender and last_type:
                maybe_start_agent_on_message(root, state, name, sender=last_sender, msg_type=last_type)

            ok = nudge_agent(root, state, name, reason="MAILBOX UPDATED")
            if ok:
                print(f"[router] nudged {name}")
            else:
                print(f"[router] mailbox updated for {name}, but could not nudge (window missing?)")
            sys.stdout.flush()

        time.sleep(interval)


# -----------------------------
# High-level operations: init/import/resume/open
# -----------------------------

def create_root_structure(root: Path, state: Dict[str, Any]) -> None:
    mkdirp(root)
    install_self_into_root(root)
    ensure_shared_scaffold(root, state)
    update_roster(root, state)
    save_state(root, state)


def ensure_tmux(root: Path, state: Dict[str, Any], *, launch_codex: bool) -> None:
    ensure_tmux_session(root, state)
    ensure_router_window(root, state)
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

    create_root_structure(root, state)
    create_git_scaffold_new(root, state)
    ensure_agents_created(root, state)
    update_roster(root, state)
    save_state(root, state)

    state = load_state(root)
    write_message(
        root,
        state,
        sender="cteam",
        recipient="pm",
        subject="Kickoff",
        body="Start by filling shared/GOALS.md, shared/PLAN.md, shared/TASKS.md and then assign tasks.",
        msg_type=ASSIGNMENT_TYPE,
        task="PM-KICKOFF",
        nudge=True,
        start_if_needed=True,
    )

    if args.no_tmux:
        print(f"Initialized workspace at {root}")
        print(f"Open tmux later: python3 cteam.py open {shlex.quote(str(root))}")
        return

    ensure_tmux(root, state, launch_codex=not args.no_codex)
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

    create_root_structure(root, state)
    create_git_scaffold_import(root, state, src)
    ensure_agents_created(root, state)
    update_roster(root, state)
    save_state(root, state)

    state = load_state(root)
    write_message(
        root,
        state,
        sender="cteam",
        recipient="pm",
        subject="Import kickoff: infer goals + plan",
        body=(
            "This is an imported codebase.\n\n"
            "1) Inspect README/docs/code and fill shared/GOALS.md with inferred goals + open questions.\n"
            "2) Write shared/PLAN.md + shared/TASKS.md.\n"
            "3) Assign first tasks to architect/devs/tester/researcher using `cteam assign`.\n"
        ),
        msg_type=ASSIGNMENT_TYPE,
        task="PM-IMPORT-KICKOFF",
        nudge=True,
        start_if_needed=True,
    )

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
        print(f"Open tmux later: python3 cteam.py open {shlex.quote(str(root))}")
        return

    ensure_tmux(root, state, launch_codex=not args.no_codex)
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

    create_root_structure(root, state)
    if not (root / DIR_PROJECT_BARE).exists() or not (root / DIR_PROJECT_BARE / "HEAD").exists():
        git_init_bare(root / DIR_PROJECT_BARE, branch=state["git"]["default_branch"])
    if not (root / DIR_PROJECT_CHECKOUT).exists():
        git_clone(root / DIR_PROJECT_BARE, root / DIR_PROJECT_CHECKOUT)

    ensure_agents_created(root, state)
    update_roster(root, state)
    save_state(root, state)

    if args.no_tmux:
        print(f"Workspace ready at {root} (tmux disabled)")
        return

    ensure_tmux(root, state, launch_codex=not args.no_codex)
    print(f"tmux session: {state['tmux']['session']}")
    if not args.no_attach:
        tmux_attach(state["tmux"]["session"], window=args.window)


def cmd_open(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    create_root_structure(root, state)
    if not (root / DIR_PROJECT_BARE).exists() or not (root / DIR_PROJECT_BARE / "HEAD").exists():
        git_init_bare(root / DIR_PROJECT_BARE, branch=state["git"]["default_branch"])
    if not (root / DIR_PROJECT_CHECKOUT).exists():
        git_clone(root / DIR_PROJECT_BARE, root / DIR_PROJECT_CHECKOUT)

    ensure_agents_created(root, state)
    update_roster(root, state)
    save_state(root, state)
    if not args.no_tmux:
        ensure_tmux(root, state, launch_codex=not args.no_codex)
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
        raise CTeamError(f"tmux session not running: {session} (run: python3 cteam.py open {root})")
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
        sender=args.sender or "human",
        recipient=args.to,
        subject=args.subject or "",
        body=body,
        msg_type="MESSAGE",
        task=None,
        nudge=not args.no_nudge,
        start_if_needed=args.start_if_needed,
    )
    if args.start_if_needed:
        ensure_tmux_session(root, state)
        ensure_router_window(root, state)
        agent = next((a for a in state["agents"] if a["name"] == args.to), None)
        if agent and not is_codex_running(state, args.to):
            start_codex_in_window(root, state, agent, boot=False)

    if not args.no_follow:
        session = state["tmux"]["session"]
        if tmux_has_session(session):
            tmux_select_window(session, args.to)

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

    sender = args.sender or "human"
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

    write_message(
        root,
        state,
        sender=args.sender or "pm",
        recipient=args.to,
        subject=args.subject or "",
        body=body,
        msg_type=ASSIGNMENT_TYPE,
        task=args.task,
        nudge=not args.no_nudge,
        start_if_needed=True,
    )
    if not args.no_follow:
        session = state["tmux"]["session"]
        if tmux_has_session(session):
            tmux_select_window(session, args.to)
    print(f"assigned to {args.to}")


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
        ok = nudge_agent(root, state, t, reason=args.reason or "NUDGE")
        print(f"{t}: {'nudged' if ok else 'could not nudge'}")
    if not args.no_follow and len(targets) == 1:
        session = state["tmux"]["session"]
        if tmux_has_session(session):
            tmux_select_window(session, targets[0])


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


def cmd_add_agent(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    role = args.role
    name = args.name
    if not name:
        existing = {a["name"] for a in state["agents"]}
        if role == "developer":
            i = 1
            while f"dev{i}" in existing:
                i += 1
            name = f"dev{i}"
        else:
            base = {"project_manager": "pm"}.get(role, role)
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

    print(f"added agent: {name} ({title}) role={role}")


def cmd_doc_walk(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.workdir))
    if not root:
        raise CTeamError("could not find cteam.json in this directory or its parents")
    state = load_state(root)

    write_message(
        root,
        state,
        sender=args.sender or "human",
        recipient="pm",
        subject=args.subject or "Doc-walk kickoff",
        body=(
            "Please run a documentation sprint over the existing code:\n"
            "- Fill shared/GOALS.md (infer goals)\n"
            "- Build a doc plan in shared/PLAN.md + shared/TASKS.md\n"
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
                    sender=args.sender or "human",
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
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cteam", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_workspace(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("workdir", help="Workspace directory or any path inside it.")
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
    p_watch.add_argument("workdir")
    p_watch.add_argument("--interval", type=float, default=1.5)
    p_watch.set_defaults(func=cmd_watch)

    p_status = sub.add_parser("status", help="Show workspace + agent status.")
    p_status.add_argument("workdir")
    p_status.set_defaults(func=cmd_status)

    p_sync = sub.add_parser("sync", help="Fetch/pull and show git statuses.")
    p_sync.add_argument("workdir")
    p_sync.add_argument("--all", action="store_true", help="Also show agent repo statuses.")
    p_sync.add_argument("--agent", help="Limit to specific agent(s) with --all.")
    p_sync.add_argument("--fetch", action="store_true")
    p_sync.add_argument("--pull", action="store_true")
    p_sync.add_argument("--no-show-branches", action="store_false", dest="show_branches")
    p_sync.set_defaults(show_branches=True)
    p_sync.set_defaults(func=cmd_sync)

    p_seed = sub.add_parser("seed-sync", help="Copy seed/ and seed-extras/ into agent workdirs.")
    p_seed.add_argument("workdir")
    p_seed.add_argument("--clean", action="store_true")
    p_seed.set_defaults(func=cmd_seed_sync)

    p_msg = sub.add_parser("msg", help="Send a message to one agent.")
    p_msg.add_argument("workdir")
    p_msg.add_argument("--to", required=True)
    p_msg.add_argument("--from", dest="sender")
    p_msg.add_argument("--subject")
    p_msg.add_argument("--file")
    p_msg.add_argument("--no-nudge", action="store_true")
    p_msg.add_argument("--start-if-needed", action="store_true",
                       help="If recipient is not running Codex, start it in their tmux window.")
    p_msg.add_argument("--no-follow", action="store_true", help="Do not select recipient window after sending.")
    p_msg.add_argument("text", nargs="?", default="")
    p_msg.set_defaults(func=cmd_msg)

    p_b = sub.add_parser("broadcast", help="Broadcast a message to all agents.")
    p_b.add_argument("workdir")
    p_b.add_argument("--from", dest="sender")
    p_b.add_argument("--subject")
    p_b.add_argument("--file")
    p_b.add_argument("--no-nudge", action="store_true")
    p_b.add_argument("--start-if-needed", action="store_true", help="Start Codex in agent windows if needed.")
    p_b.add_argument("text", nargs="?", default="")
    p_b.set_defaults(func=cmd_broadcast)

    p_assign = sub.add_parser("assign", help="Send an ASSIGNMENT to an agent (starts them if needed).")
    p_assign.add_argument("workdir")
    p_assign.add_argument("--to", required=True)
    p_assign.add_argument("--task", help="Task ID (e.g., T001).")
    p_assign.add_argument("--from", dest="sender")
    p_assign.add_argument("--subject")
    p_assign.add_argument("--file")
    p_assign.add_argument("--no-nudge", action="store_true")
    p_assign.add_argument("text", nargs="?", default="")
    p_assign.add_argument("--no-follow", action="store_true", help="Do not select recipient window after sending.")
    p_assign.set_defaults(func=cmd_assign)

    p_nudge = sub.add_parser("nudge", help="Send a manual nudge to an agent window.")
    p_nudge.add_argument("workdir")
    p_nudge.add_argument("--to", default="pm", help="agent(s) comma-separated or 'all'")
    p_nudge.add_argument("--reason", default="NUDGE")
    p_nudge.add_argument("--no-follow", action="store_true", help="Do not select the target window.")
    p_nudge.set_defaults(func=cmd_nudge)

    p_restart = sub.add_parser("restart", help="Restart Codex in agent tmux windows (respawn).")
    p_restart.add_argument("workdir")
    p_restart.add_argument("--window", help="agent window name(s) comma-separated or 'all'")
    p_restart.add_argument("--hard", action="store_true", help="send Ctrl-C before respawn (best-effort)")
    p_restart.set_defaults(func=cmd_restart)

    p_add = sub.add_parser("add-agent", help="Add a new agent to an existing workspace.")
    p_add.add_argument("workdir")
    p_add.add_argument("--role", default="developer", choices=["developer","tester","researcher","architect","project_manager"])
    p_add.add_argument("--name")
    p_add.add_argument("--title")
    p_add.add_argument("--persona")
    p_add.add_argument("--no-tmux", action="store_true")
    p_add.add_argument("--start-codex", action="store_true", help="Start codex immediately for the new agent.")
    p_add.set_defaults(func=cmd_add_agent)

    p_doc = sub.add_parser("doc-walk", help="Kick off a documentation sprint over the repo (PM-led).")
    p_doc.add_argument("workdir")
    p_doc.add_argument("--from", dest="sender")
    p_doc.add_argument("--subject")
    p_doc.add_argument("--task")
    p_doc.add_argument("--auto", action="store_true", help="Also assign initial doc tasks to architect/tester/researcher.")
    p_doc.set_defaults(func=cmd_doc_walk)

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
