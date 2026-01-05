#!/usr/bin/env python3
"""oteam_openhands_trainyard.py

A production-grade single-file daemon that ports the core *OTeam* idea onto the
OpenHands Software Agent SDK.

Design intent (per your requirements):
- One daemon instance == one project (one repo/workdir).
- Continuous loop: seed -> plan -> execute -> verify -> idle -> (seed changes) -> migrate -> repeat.
- Works for *very long* projects (weeks) where neither SEED nor the whole project can fit in any LLM context.
- Parallelization happens at the level of *independent work packets* (DAG nodes), not as a serialized
  “implement/test/review” pipeline. The system can fan out (multiple parallel lanes) and then
  fan back in (join nodes) like tracks in a train station.
- Each DAG node executes inside its own git-backed workspace (git worktree), with a shared directory
  available inside the workspace. The central/trunk workspace is also git-backed. Before fanning out,
  the trunk is checkpoint-committed.
- No interactive TUI; instead: structured JSONL logs you can `tail -f`.
- Human interaction is via:
    (a) Customer inbound/outbound HTTP endpoints (customer chat),
    (b) Telegram bot for operator control and status.

Important caveats:
- This file assumes you have OpenHands SDK installed in the environment where you run it.
  (package: `openhands-sdk` / `software-agent-sdk` depending on installation method)
- This daemon is intentionally generic: it can manage software projects, research projects,
  documentation, datasets, etc. Git is used as the universal, content-addressed workspace log.

Usage:
    python oteam_openhands_trainyard.py --project /path/to/repo

Environment variables (LLM):
    LLM_API_KEY      (required)
    LLM_MODEL        (default: anthropic/claude-sonnet-4-5-20250929)
    LLM_BASE_URL     (optional; for LiteLLM-compatible gateways)

Environment variables (Telegram):
    TELEGRAM_BOT_TOKEN (optional)
    TELEGRAM_ADMIN_CHAT_ID (optional; restrict operator commands)

"""


from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import http.server
import io
import json
import logging
import os
import queue
import re
import shutil
import signal
import socketserver
import sqlite3
import subprocess
import threading
import time
import traceback
import types
import typing as t
import urllib.parse
import urllib.request
import uuid


# Optional dependencies (we keep graceful fallbacks where possible).
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from pydantic import SecretStr  # type: ignore
except Exception:  # pragma: no cover
    SecretStr = None  # type: ignore



# OpenHands SDK imports (required at runtime for actual agent execution).
# We import lazily where possible so that basic commands (like --help) still work without it.
OH_AVAILABLE = False
try:  # pragma: no cover
    from openhands.sdk import (
        LLM,
        Agent,
        AgentContext,
        Conversation,
        Event,
        LLMConvertibleEvent,
        get_logger as oh_get_logger,
    )
    from openhands.sdk.context import Skill  # type: ignore
    from openhands.sdk.context.condenser import LLMSummarizingCondenser  # type: ignore
    from openhands.sdk.tool import Tool  # type: ignore
    from openhands.sdk.llm import content_to_str  # type: ignore

    from openhands.tools.file_editor import FileEditorTool  # type: ignore
    from openhands.tools.terminal import TerminalTool  # type: ignore
    try:
        from openhands.tools.task_tracker import TaskTrackerTool  # type: ignore
    except Exception:
        TaskTrackerTool = None  # type: ignore

    OH_AVAILABLE = True
except Exception:  # pragma: no cover
    OH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utcnow() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def short_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:12]


def stable_json(obj: t.Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if max_bytes is None:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        else:
            remaining = max_bytes
            while remaining > 0:
                b = f.read(min(1024 * 1024, remaining))
                if not b:
                    break
                h.update(b)
                remaining -= len(b)
    return h.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(
    cmd: list[str],
    cwd: str | None = None,
    timeout_s: int | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a command and return (rc, stdout, stderr)."""
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = p.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out, err + "\n[timeout]"
    return p.returncode, out, err


_JSON_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def extract_first_json_object(text: str) -> dict[str, t.Any] | None:
    """Best-effort extraction of the first JSON object from a blob of text."""
    text = text.strip()
    if not text:
        return None
    # Fast path: pure JSON.
    if text.startswith("{") and text.endswith("}"):
        try:
            return t.cast(dict[str, t.Any], json.loads(text))
        except Exception:
            pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return t.cast(dict[str, t.Any], json.loads(blob))
    except Exception:
        # Last attempt: strip code fences and retry.
        blob = blob.strip().strip("`")

        try:
            return t.cast(dict[str, t.Any], json.loads(blob))
        except Exception:
            return None


def clamp(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def now_ts() -> float:
    return time.time()



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    project_root: str
    state_dir: str
    db_path: str
    log_dir: str
    jsonl_log_path: str
    conversations_dir: str
    workdir_root: str
    trunk_workdir: str
    lanes_root: str
    prompts_dir: str

    trunk_branch: str = "oteam/trunk"
    http_host: str = "0.0.0.0"
    http_port: int = 8088

    telegram_bot_token: str | None = None
    telegram_admin_chat_id: str | None = None  # if set, restrict operator commands to this chat id

    node_workers: int = 3
    max_parallel_lanes: int = 6

    scheduler_tick_s: float = 2.0
    conductor_tick_s: float = 10.0
    repo_index_tick_s: float = 60.0

    repo_index_max_file_bytes: int = 200_000
    repo_index_max_files_per_tick: int = 200

    llm_model: str = "anthropic/claude-sonnet-4-5-20250929"
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    condenser_max_events: int = 40
    condenser_keep_first: int = 4

    # Stationmaster actor-critic control
    stationmaster_actor_critic: bool = True
    stationmaster_ac_max_retries: int = 1

    # Safety/ops
    git_user_name: str = "oteam-bot"
    git_user_email: str = "oteam-bot@localhost"

    def ensure_layout(self) -> None:
        for p in [
            self.state_dir,
            self.log_dir,
            self.conversations_dir,
            self.workdir_root,
            self.trunk_workdir,
            self.lanes_root,
            self.prompts_dir,
        ]:
            ensure_dir(p)

    @staticmethod
    def from_project(project_root: str, state_dir_name: str = ".oteam") -> "Config":
        project_root = os.path.abspath(project_root)
        state_dir = os.path.join(project_root, state_dir_name)
        db_path = os.path.join(state_dir, "state.sqlite3")
        log_dir = os.path.join(state_dir, "logs")
        jsonl_log_path = os.path.join(log_dir, "events.jsonl")
        conversations_dir = os.path.join(state_dir, "conversations")
        workdir_root = os.path.join(state_dir, "workdir")
        trunk_workdir = os.path.join(workdir_root, "trunk")
        lanes_root = os.path.join(workdir_root, "lanes")
        prompts_dir = os.path.join(state_dir, "prompts")

        cfg = Config(
            project_root=project_root,
            state_dir=state_dir,
            db_path=db_path,
            log_dir=log_dir,
            jsonl_log_path=jsonl_log_path,
            conversations_dir=conversations_dir,
            workdir_root=workdir_root,
            trunk_workdir=trunk_workdir,
            lanes_root=lanes_root,
            prompts_dir=prompts_dir,
        )

        # Environment overrides
        cfg.http_host = os.getenv("OTEAM_HTTP_HOST", cfg.http_host)
        cfg.http_port = int(os.getenv("OTEAM_HTTP_PORT", str(cfg.http_port)))
        cfg.trunk_branch = os.getenv("OTEAM_TRUNK_BRANCH", cfg.trunk_branch)

        cfg.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("OTEAM_TELEGRAM_BOT_TOKEN")
        cfg.telegram_admin_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID") or os.getenv("OTEAM_TELEGRAM_ADMIN_CHAT_ID")

        cfg.node_workers = int(os.getenv("OTEAM_NODE_WORKERS", str(cfg.node_workers)))
        cfg.max_parallel_lanes = int(os.getenv("OTEAM_MAX_PARALLEL_LANES", str(cfg.max_parallel_lanes)))

        cfg.scheduler_tick_s = float(os.getenv("OTEAM_SCHEDULER_TICK_S", str(cfg.scheduler_tick_s)))
        cfg.conductor_tick_s = float(os.getenv("OTEAM_CONDUCTOR_TICK_S", str(cfg.conductor_tick_s)))
        cfg.repo_index_tick_s = float(os.getenv("OTEAM_REPO_INDEX_TICK_S", str(cfg.repo_index_tick_s)))

        cfg.llm_model = os.getenv("LLM_MODEL", cfg.llm_model)
        cfg.llm_base_url = os.getenv("LLM_BASE_URL", cfg.llm_base_url or "") or None
        cfg.llm_api_key = os.getenv("LLM_API_KEY", cfg.llm_api_key or "") or None

        cfg.condenser_max_events = int(os.getenv("OTEAM_CONDENSER_MAX_EVENTS", str(cfg.condenser_max_events)))
        cfg.condenser_keep_first = int(os.getenv("OTEAM_CONDENSER_KEEP_FIRST", str(cfg.condenser_keep_first)))

        ac = os.getenv("OTEAM_STATIONMASTER_ACTOR_CRITIC", "")
        if ac.strip() != "":
            cfg.stationmaster_actor_critic = ac.strip().lower() in ("1", "true", "yes", "y", "on")
        cfg.stationmaster_ac_max_retries = int(os.getenv("OTEAM_STATIONMASTER_AC_MAX_RETRIES", str(cfg.stationmaster_ac_max_retries)))

        cfg.git_user_name = os.getenv("OTEAM_GIT_USER_NAME", cfg.git_user_name)
        cfg.git_user_email = os.getenv("OTEAM_GIT_USER_EMAIL", cfg.git_user_email)

        return cfg



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class JsonlEventLog:
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path))
        self._lock = threading.Lock()

    def emit(self, event_type: str, **fields: t.Any) -> None:
        rec = {"ts": utcnow(), "type": event_type}
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def setup_logging(cfg: Config) -> tuple[logging.Logger, JsonlEventLog]:
    ensure_dir(cfg.log_dir)
    logger = logging.getLogger("oteam")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(cfg.log_dir, "oteam.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    jlog = JsonlEventLog(cfg.jsonl_log_path)
    return logger, jlog



# ---------------------------------------------------------------------------
# Persistent State (SQLite)
# ---------------------------------------------------------------------------

class StateDB:
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path))
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    sender TEXT,
                    text TEXT NOT NULL,
                    raw_json TEXT
                );

                CREATE TABLE IF NOT EXISTS outbox (
                    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING'
                );

                CREATE TABLE IF NOT EXISTS seed_draft_items (
                    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'NEW',
                    tags_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS seed_versions (
                    seed_version_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    commit_hash TEXT,
                    seed_path TEXT,
                    summary TEXT,
                    seed_text_b64 TEXT
                );

                CREATE TABLE IF NOT EXISTS requirement_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    seed_version_id TEXT NOT NULL,
                    heading TEXT,
                    content TEXT NOT NULL,
                    token_est INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(seed_version_id) REFERENCES seed_versions(seed_version_id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS requirement_fts
                USING fts5(chunk_id, seed_version_id, heading, content);

                CREATE TABLE IF NOT EXISTS epochs (
                    epoch_id TEXT PRIMARY KEY,
                    seed_version_id TEXT NOT NULL,
                    ts_start TEXT NOT NULL,
                    ts_end TEXT,
                    status TEXT NOT NULL,
                    notes TEXT,
                    base_commit TEXT,
                    FOREIGN KEY(seed_version_id) REFERENCES seed_versions(seed_version_id)
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    epoch_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    deliverables_json TEXT NOT NULL DEFAULT '[]',
                    req_links_json TEXT NOT NULL DEFAULT '[]',
                    scope_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    base_commit TEXT,
                    result_commit TEXT,
                    workspace_path TEXT,
                    conversation_id TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    summary TEXT,
                    output_json TEXT,
                    FOREIGN KEY(epoch_id) REFERENCES epochs(epoch_id)
                );

                CREATE TABLE IF NOT EXISTS node_deps (
                    node_id TEXT NOT NULL,
                    dep_node_id TEXT NOT NULL,
                    PRIMARY KEY (node_id, dep_node_id),
                    FOREIGN KEY(node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
                    FOREIGN KEY(dep_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_by TEXT,
                    title TEXT,
                    details TEXT,
                    data_json TEXT,
                    decision_ts TEXT,
                    decision_by TEXT,
                    decision_reason TEXT
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    kind TEXT,
                    path TEXT,
                    sha256 TEXT,
                    meta_json TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS repo_fts
                USING fts5(path, content);

                CREATE TABLE IF NOT EXISTS repo_index_meta (
                    path TEXT PRIMARY KEY,
                    mtime REAL,
                    size INTEGER,
                    sha256 TEXT
                );
                """
            )
            self._conn.commit()

    # ---- low-level helpers

    def execute(self, sql: str, params: tuple[t.Any, ...] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def executemany(self, sql: str, seq: list[tuple[t.Any, ...]]) -> None:
        with self._lock:
            self._conn.executemany(sql, seq)
            self._conn.commit()

    def query(self, sql: str, params: tuple[t.Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchall()

    def query_one(self, sql: str, params: tuple[t.Any, ...] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---- meta

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        row = self.query_one("SELECT value FROM meta WHERE key=?", (key,))
        return row["value"] if row else default

    def meta_set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    # ---- messages / outbox

    def add_message(self, channel: str, sender: str | None, text: str, raw: dict[str, t.Any] | None = None) -> int:
        raw_json = json.dumps(raw, ensure_ascii=False) if raw is not None else None
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages(ts,channel,sender,text,raw_json) VALUES(?,?,?,?,?)",
                (utcnow(), channel, sender, text, raw_json),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_messages(self, channel: str | None = None, since_id: int | None = None, limit: int = 50) -> list[sqlite3.Row]:
        q = "SELECT * FROM messages"
        params: list[t.Any] = []
        where: list[str] = []
        if channel is not None:
            where.append("channel=?")
            params.append(channel)
        if since_id is not None:
            where.append("msg_id>?")
            params.append(since_id)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY msg_id DESC LIMIT ?"
        params.append(limit)
        rows = self.query(q, tuple(params))
        return list(reversed(rows))

    def enqueue_outbox(self, customer_id: str, text: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO outbox(ts,customer_id,text,status) VALUES(?,?,?, 'PENDING')",
                (utcnow(), customer_id, text),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def dequeue_outbox(self, customer_id: str, limit: int = 20, ack: bool = True) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM outbox WHERE customer_id=? AND status='PENDING' ORDER BY outbox_id ASC LIMIT ?",
                (customer_id, limit),
            ).fetchall()
            if ack and rows:
                ids = [r["outbox_id"] for r in rows]
                self._conn.executemany(
                    "UPDATE outbox SET status='SENT' WHERE outbox_id=?",
                    [(i,) for i in ids],
                )
                self._conn.commit()
            return rows

    # ---- seed draft + versions

    def add_seed_draft_item(self, source: str, text: str, tags: list[str] | None = None) -> int:
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO seed_draft_items(ts,source,text,status,tags_json) VALUES(?,?,?,?,?)",
                (utcnow(), source, text, "NEW", tags_json),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_seed_draft_items(self, since_id: int | None = None, limit: int = 200) -> list[sqlite3.Row]:
        if since_id is None:
            return self.query(
                "SELECT * FROM seed_draft_items ORDER BY item_id DESC LIMIT ?",
                (limit,),
            )[::-1]
        return self.query(
            "SELECT * FROM seed_draft_items WHERE item_id>? ORDER BY item_id ASC LIMIT ?",
            (since_id, limit),
        )

    def mark_seed_draft_item(self, item_id: int, status: str) -> None:
        self.execute("UPDATE seed_draft_items SET status=? WHERE item_id=?", (status, item_id))

    def insert_seed_version(self, seed_text: str, commit_hash: str | None, seed_path: str, summary: str | None = None) -> str:
        seed_version_id = sha256_bytes(seed_text.encode("utf-8"))[:24]
        b64 = base64.b64encode(seed_text.encode("utf-8")).decode("ascii")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO seed_versions(seed_version_id,ts,commit_hash,seed_path,summary,seed_text_b64) VALUES(?,?,?,?,?,?)",
                (seed_version_id, utcnow(), commit_hash, seed_path, summary, b64),
            )
            self._conn.commit()
        return seed_version_id

    def get_seed_version(self, seed_version_id: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM seed_versions WHERE seed_version_id=?", (seed_version_id,))

    def get_latest_seed_version(self) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM seed_versions ORDER BY ts DESC LIMIT 1")

    def set_active_seed_version(self, seed_version_id: str) -> None:
        self.meta_set("active_seed_version_id", seed_version_id)

    def get_active_seed_version_id(self) -> str | None:
        return self.meta_get("active_seed_version_id")

    def replace_requirement_chunks(self, seed_version_id: str, chunks: list[dict[str, t.Any]]) -> None:
        """Chunks: [{chunk_id, heading, content, token_est}]"""
        with self._lock:
            self._conn.execute("DELETE FROM requirement_chunks WHERE seed_version_id=?", (seed_version_id,))
            self._conn.execute("DELETE FROM requirement_fts WHERE seed_version_id=?", (seed_version_id,))
            self._conn.executemany(
                "INSERT INTO requirement_chunks(chunk_id,seed_version_id,heading,content,token_est) VALUES(?,?,?,?,?)",
                [
                    (
                        c["chunk_id"],
                        seed_version_id,
                        c.get("heading"),
                        c["content"],
                        int(c.get("token_est") or 0),
                    )
                    for c in chunks
                ],
            )
            self._conn.executemany(
                "INSERT INTO requirement_fts(chunk_id,seed_version_id,heading,content) VALUES(?,?,?,?)",
                [
                    (
                        c["chunk_id"],
                        seed_version_id,
                        c.get("heading"),
                        c["content"],
                    )
                    for c in chunks
                ],
            )
            self._conn.commit()

    def search_requirements(self, query: str, seed_version_id: str | None = None, limit: int = 8) -> list[dict[str, t.Any]]:
        if seed_version_id:
            rows = self.query(
                "SELECT chunk_id, heading, snippet(requirement_fts, 3, '[', ']', '…', 12) AS snip "
                "FROM requirement_fts WHERE requirement_fts MATCH ? AND seed_version_id=? LIMIT ?",
                (query, seed_version_id, limit),
            )
        else:
            rows = self.query(
                "SELECT chunk_id, heading, snippet(requirement_fts, 3, '[', ']', '…', 12) AS snip "
                "FROM requirement_fts WHERE requirement_fts MATCH ? LIMIT ?",
                (query, limit),
            )
        return [
            {"chunk_id": r["chunk_id"], "heading": r["heading"], "snippet": r["snip"]}
            for r in rows
        ]

    def get_requirement_chunk(self, chunk_id: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM requirement_chunks WHERE chunk_id=?", (chunk_id,))

    # ---- epochs (seed runs)

    def create_epoch(self, seed_version_id: str, base_commit: str | None) -> str:
        epoch_id = f"E-{_dt.datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO epochs(epoch_id,seed_version_id,ts_start,status,notes,base_commit) VALUES(?,?,?,?,?,?)",
                (epoch_id, seed_version_id, utcnow(), "PLANNING", None, base_commit),
            )
            self._conn.commit()
        self.meta_set("active_epoch_id", epoch_id)
        return epoch_id

    def get_active_epoch_id(self) -> str | None:
        return self.meta_get("active_epoch_id")

    def get_epoch(self, epoch_id: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM epochs WHERE epoch_id=?", (epoch_id,))

    def set_epoch_status(self, epoch_id: str, status: str, notes: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE epochs SET status=?, notes=?, ts_end=CASE WHEN ? IN ('DONE','SUPERSEDED') THEN ? ELSE ts_end END WHERE epoch_id=?",
                (status, notes, status, utcnow(), epoch_id),
            )
            self._conn.commit()

    # ---- nodes (DAG)

    def upsert_node(self, node: dict[str, t.Any]) -> None:
        # required keys
        now = utcnow()
        node_id = node["node_id"]
        epoch_id = node["epoch_id"]
        with self._lock:
            existing = self._conn.execute("SELECT node_id FROM nodes WHERE node_id=?", (node_id,)).fetchone()
            if existing:
                self._conn.execute(
                    """
                    UPDATE nodes SET
                        kind=?, title=?, objective=?,
                        deliverables_json=?, req_links_json=?, scope_json=?,
                        status=?, priority=?,
                        updated_at=?,
                        base_commit=COALESCE(?, base_commit),
                        result_commit=COALESCE(?, result_commit),
                        workspace_path=COALESCE(?, workspace_path),
                        conversation_id=COALESCE(?, conversation_id),
                        attempt_count=COALESCE(?, attempt_count),
                        last_error=?,
                        summary=?,
                        output_json=?
                    WHERE node_id=?
                    """,
                    (
                        node["kind"],
                        node["title"],
                        node["objective"],
                        json.dumps(node.get("deliverables") or [], ensure_ascii=False),
                        json.dumps(node.get("req_links") or [], ensure_ascii=False),
                        json.dumps(node.get("scope_hints") or [], ensure_ascii=False),
                        node.get("status") or "PENDING",
                        int(node.get("priority") or 0),
                        now,
                        node.get("base_commit"),
                        node.get("result_commit"),
                        node.get("workspace_path"),
                        node.get("conversation_id"),
                        int(node.get("attempt_count") or 0),
                        node.get("last_error"),
                        node.get("summary"),
                        json.dumps(node.get("output") or {}, ensure_ascii=False),
                        node_id,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO nodes(
                        node_id, epoch_id, kind, title, objective,
                        deliverables_json, req_links_json, scope_json,
                        status, priority, created_at, updated_at,
                        base_commit, result_commit, workspace_path, conversation_id,
                        attempt_count, last_error, summary, output_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        node_id,
                        epoch_id,
                        node["kind"],
                        node["title"],
                        node["objective"],
                        json.dumps(node.get("deliverables") or [], ensure_ascii=False),
                        json.dumps(node.get("req_links") or [], ensure_ascii=False),
                        json.dumps(node.get("scope_hints") or [], ensure_ascii=False),
                        node.get("status") or "PENDING",
                        int(node.get("priority") or 0),
                        now,
                        now,
                        node.get("base_commit"),
                        node.get("result_commit"),
                        node.get("workspace_path"),
                        node.get("conversation_id"),
                        int(node.get("attempt_count") or 0),
                        node.get("last_error"),
                        node.get("summary"),
                        json.dumps(node.get("output") or {}, ensure_ascii=False),
                    ),
                )
            self._conn.commit()

    def set_node_status(self, node_id: str, status: str, last_error: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE nodes SET status=?, last_error=?, updated_at=? WHERE node_id=?",
                (status, last_error, utcnow(), node_id),
            )
            self._conn.commit()

    def bump_node_attempt(self, node_id: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT attempt_count FROM nodes WHERE node_id=?", (node_id,)).fetchone()
            n = int(row["attempt_count"]) if row else 0
            n += 1
            self._conn.execute(
                "UPDATE nodes SET attempt_count=?, updated_at=? WHERE node_id=?",
                (n, utcnow(), node_id),
            )
            self._conn.commit()
            return n

    def list_nodes(self, epoch_id: str, status: str | None = None, limit: int = 200) -> list[dict[str, t.Any]]:
        q = "SELECT * FROM nodes WHERE epoch_id=?"
        params: list[t.Any] = [epoch_id]
        if status:
            q += " AND status=?"
            params.append(status)
        q += " ORDER BY priority DESC, created_at ASC LIMIT ?"
        params.append(limit)
        rows = self.query(q, tuple(params))
        out: list[dict[str, t.Any]] = []
        for r in rows:
            out.append(
                {
                    "node_id": r["node_id"],
                    "epoch_id": r["epoch_id"],
                    "kind": r["kind"],
                    "title": r["title"],
                    "objective": r["objective"],
                    "deliverables": json.loads(r["deliverables_json"] or "[]"),
                    "req_links": json.loads(r["req_links_json"] or "[]"),
                    "scope_hints": json.loads(r["scope_json"] or "[]"),
                    "status": r["status"],
                    "priority": int(r["priority"] or 0),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "base_commit": r["base_commit"],
                    "result_commit": r["result_commit"],
                    "workspace_path": r["workspace_path"],
                    "conversation_id": r["conversation_id"],
                    "attempt_count": int(r["attempt_count"] or 0),
                    "last_error": r["last_error"],
                    "summary": r["summary"],
                    "output": json.loads(r["output_json"] or "{}"),
                }
            )
        return out

    def get_node(self, node_id: str) -> dict[str, t.Any] | None:
        r = self.query_one("SELECT * FROM nodes WHERE node_id=?", (node_id,))
        if not r:
            return None
        return {
            "node_id": r["node_id"],
            "epoch_id": r["epoch_id"],
            "kind": r["kind"],
            "title": r["title"],
            "objective": r["objective"],
            "deliverables": json.loads(r["deliverables_json"] or "[]"),
            "req_links": json.loads(r["req_links_json"] or "[]"),
            "scope_hints": json.loads(r["scope_json"] or "[]"),
            "status": r["status"],
            "priority": int(r["priority"] or 0),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "base_commit": r["base_commit"],
            "result_commit": r["result_commit"],
            "workspace_path": r["workspace_path"],
            "conversation_id": r["conversation_id"],
            "attempt_count": int(r["attempt_count"] or 0),
            "last_error": r["last_error"],
            "summary": r["summary"],
            "output": json.loads(r["output_json"] or "{}"),
        }

    def set_node_output(self, node_id: str, summary: str | None, output: dict[str, t.Any] | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE nodes SET summary=?, output_json=?, updated_at=? WHERE node_id=?",
                (summary, json.dumps(output or {}, ensure_ascii=False), utcnow(), node_id),
            )
            self._conn.commit()

    def add_dep(self, node_id: str, dep_node_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO node_deps(node_id,dep_node_id) VALUES(?,?)",
                (node_id, dep_node_id),
            )
            self._conn.commit()

    def list_deps(self, node_id: str) -> list[str]:
        rows = self.query("SELECT dep_node_id FROM node_deps WHERE node_id=?", (node_id,))
        return [r["dep_node_id"] for r in rows]

    def list_dependents(self, dep_node_id: str) -> list[str]:
        rows = self.query("SELECT node_id FROM node_deps WHERE dep_node_id=?", (dep_node_id,))
        return [r["node_id"] for r in rows]

    # ---- approvals

    def create_approval(self, title: str, details: str, data: dict[str, t.Any], requested_by: str | None = None) -> str:
        approval_id = "A-" + uuid.uuid4().hex[:10]
        with self._lock:
            self._conn.execute(
                "INSERT INTO approvals(approval_id,ts,status,requested_by,title,details,data_json) VALUES(?,?,?,?,?,?,?)",
                (approval_id, utcnow(), "PENDING", requested_by, title, details, json.dumps(data, ensure_ascii=False)),
            )
            self._conn.commit()
        return approval_id

    def list_approvals(self, status: str | None = None, limit: int = 50) -> list[dict[str, t.Any]]:
        q = "SELECT * FROM approvals"
        params: list[t.Any] = []
        if status:
            q += " WHERE status=?"
            params.append(status)
        q += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self.query(q, tuple(params))
        out: list[dict[str, t.Any]] = []
        for r in rows:
            out.append(
                {
                    "approval_id": r["approval_id"],
                    "ts": r["ts"],
                    "status": r["status"],
                    "requested_by": r["requested_by"],
                    "title": r["title"],
                    "details": r["details"],
                    "data": json.loads(r["data_json"] or "{}"),
                    "decision_ts": r["decision_ts"],
                    "decision_by": r["decision_by"],
                    "decision_reason": r["decision_reason"],
                }
            )
        return out

    def decide_approval(self, approval_id: str, decision: str, by: str, reason: str | None = None) -> None:
        assert decision in ("APPROVED", "REJECTED")
        with self._lock:
            self._conn.execute(
                "UPDATE approvals SET status=?, decision_ts=?, decision_by=?, decision_reason=? WHERE approval_id=?",
                (decision, utcnow(), by, reason, approval_id),
            )
            self._conn.commit()

    # ---- repo search index (very lightweight)

    def repo_index_upsert(self, path: str, content: str, mtime: float, size: int, sha256: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO repo_fts(path, content) VALUES(?,?)", (path, content))
            self._conn.execute(
                "INSERT OR REPLACE INTO repo_index_meta(path,mtime,size,sha256) VALUES(?,?,?,?)",
                (path, mtime, size, sha256),
            )
            self._conn.commit()

    def repo_index_needs_update(self, path: str, mtime: float, size: int) -> bool:
        r = self.query_one("SELECT mtime,size FROM repo_index_meta WHERE path=?", (path,))
        if not r:
            return True
        return float(r["mtime"] or 0) != float(mtime) or int(r["size"] or 0) != int(size)

    def repo_search(self, query: str, limit: int = 8) -> list[dict[str, t.Any]]:
        rows = self.query(
            "SELECT path, snippet(repo_fts, 1, '[', ']', '…', 12) AS snip "
            "FROM repo_fts WHERE repo_fts MATCH ? LIMIT ?",
            (query, limit),
        )
        return [{"path": r["path"], "snippet": r["snip"]} for r in rows]



# ---------------------------------------------------------------------------
# Git-backed workdir manager
# ---------------------------------------------------------------------------

class GitWorkdirError(RuntimeError):
    pass


class GitWorkdirManager:
    """Manages:
    - a *trunk* workspace (git worktree on cfg.trunk_branch),
    - per-node *lane* workspaces (git worktrees in detached HEAD),
    - integration of lane commits back into trunk.
    """

    def __init__(self, cfg: Config, logger: logging.Logger, jlog: JsonlEventLog):
        self.cfg = cfg
        self.logger = logger
        self.jlog = jlog
        self._git_lock = threading.RLock()

    def _git(self, args: list[str], cwd: str | None = None, timeout_s: int | None = 120) -> tuple[int, str, str]:
        cmd = ["git"] + args
        rc, out, err = run_cmd(cmd, cwd=cwd, timeout_s=timeout_s)
        self.jlog.emit("git", rc=rc, cmd=cmd, cwd=cwd, out_tail=clamp(out, 2000), err_tail=clamp(err, 2000))
        return rc, out, err

    def ensure_usable_repo(self) -> None:
        rc, out, err = self._git(["rev-parse", "--is-inside-work-tree"], cwd=self.cfg.project_root)
        if rc != 0 or out.strip() != "true":
            raise GitWorkdirError(
                f"Project root is not a git repo (or git not available).\nrc={rc}\nout={out}\nerr={err}"
            )
        # Ensure local identity (so automated commits work).
        self._ensure_git_identity()

    def _ensure_git_identity(self) -> None:
        # Only set local repo config if missing.
        def get_cfg(key: str) -> str | None:
            rc, out, _ = self._git(["config", "--local", "--get", key], cwd=self.cfg.project_root)
            if rc != 0:
                return None
            v = out.strip()
            return v or None

        name = get_cfg("user.name")
        email = get_cfg("user.email")
        if not name:
            self._git(["config", "--local", "user.name", self.cfg.git_user_name], cwd=self.cfg.project_root)
        if not email:
            self._git(["config", "--local", "user.email", self.cfg.git_user_email], cwd=self.cfg.project_root)

    def ensure_trunk_worktree(self) -> None:
        """Create/update the trunk worktree at cfg.trunk_workdir."""
        with self._git_lock:
            self.ensure_usable_repo()

            # If trunk already looks like a worktree, keep it.
            if os.path.isdir(self.cfg.trunk_workdir) and os.path.exists(os.path.join(self.cfg.trunk_workdir, ".git")):
                # Ensure it is on trunk branch.
                self._git(["checkout", self.cfg.trunk_branch], cwd=self.cfg.trunk_workdir)
                self._bootstrap_shared()
                return

            # Remove if partially created.
            if os.path.isdir(self.cfg.trunk_workdir):
                shutil.rmtree(self.cfg.trunk_workdir, ignore_errors=True)

            ensure_dir(os.path.dirname(self.cfg.trunk_workdir))

            # Create branch if needed & add worktree.
            # -B ensures branch exists/reset to HEAD the first time; if it exists, keeps as branch.
            rc, out, err = self._git(
                ["worktree", "add", "-B", self.cfg.trunk_branch, self.cfg.trunk_workdir, "HEAD"],
                cwd=self.cfg.project_root,
                timeout_s=600,
            )
            if rc != 0:
                raise GitWorkdirError(f"Failed to create trunk worktree: {err}\n{out}")
            self._bootstrap_shared()

    def _bootstrap_shared(self) -> None:
        """Ensure the 'shared/' directory exists in trunk and has a minimal structure."""
        shared = os.path.join(self.cfg.trunk_workdir, "shared")
        ensure_dir(shared)
        ensure_dir(os.path.join(shared, "nodes"))
        ensure_dir(os.path.join(shared, "seed"))
        ensure_dir(os.path.join(shared, "lessons"))

        readme = os.path.join(shared, "README.md")
        if not os.path.exists(readme):
            with open(readme, "w", encoding="utf-8") as f:
                f.write(
                    """# shared/

This directory is intended for *merge-friendly, human-readable artifacts* that help a long project stay on track.

Conventions:
- `shared/seed/`   : SEED snapshots and seed-related notes.
- `shared/nodes/`  : each work packet (DAG node) writes to `shared/nodes/<node_id>/`.
- `shared/lessons/`: accumulated lessons / guardrails discovered during execution.

Everything under `shared/` is safe to commit and merge.
"""
                )
            # Commit bootstrap (if repo is clean other than this).
            self._maybe_commit_trunk("bootstrap shared/")
        # Also ensure .oteam is ignored (best effort; not required).
        gi = os.path.join(self.cfg.trunk_workdir, ".gitignore")
        try:
            if os.path.exists(gi):
                with open(gi, "r", encoding="utf-8") as f:
                    txt = f.read()
            else:
                txt = ""
            if ".oteam/" not in txt:
                with open(gi, "a", encoding="utf-8") as f:
                    if txt and not txt.endswith("\n"):
                        f.write("\n")
                    f.write("\n.oteam/\n")
                self._maybe_commit_trunk("ignore .oteam/")
        except Exception:
            pass

    def trunk_head(self) -> str:
        rc, out, err = self._git(["rev-parse", "HEAD"], cwd=self.cfg.trunk_workdir)
        if rc != 0:
            raise GitWorkdirError(err)
        return out.strip()

    def trunk_is_clean(self) -> bool:
        rc, out, err = self._git(["status", "--porcelain"], cwd=self.cfg.trunk_workdir)
        if rc != 0:
            raise GitWorkdirError(err)
        return out.strip() == ""

    def checkpoint_trunk(self, reason: str) -> str:
        """Commit any outstanding changes in trunk so we can safely fan out."""
        with self._git_lock:
            if self.trunk_is_clean():
                return self.trunk_head()
            msg = f"oteam checkpoint: {reason}"
            rc, out, err = self._git(["add", "-A"], cwd=self.cfg.trunk_workdir)
            if rc != 0:
                raise GitWorkdirError(err)
            rc, out, err = self._git(["commit", "-m", msg], cwd=self.cfg.trunk_workdir)
            if rc != 0:
                # If nothing to commit, treat as clean.
                if "nothing to commit" in (out + err).lower():
                    return self.trunk_head()
                raise GitWorkdirError(err)
            return self.trunk_head()

    def _maybe_commit_trunk(self, reason: str) -> None:
        with contextlib.suppress(Exception):
            self.checkpoint_trunk(reason)

    # ---- lanes

    def lane_path(self, node_id: str) -> str:
        return os.path.join(self.cfg.lanes_root, node_id)

    def create_lane(self, node_id: str, base_commit: str) -> str:
        """Create a detached worktree for the node."""
        with self._git_lock:
            lane = self.lane_path(node_id)
            if os.path.isdir(lane):
                # Try to remove cleanly, then re-create.
                self.remove_lane(node_id)
            ensure_dir(os.path.dirname(lane))
            rc, out, err = self._git(
                ["worktree", "add", "--detach", lane, base_commit],
                cwd=self.cfg.project_root,
                timeout_s=600,
            )
            if rc != 0:
                raise GitWorkdirError(f"Failed to create lane {node_id}: {err}\n{out}")
            # Ensure node shared dir exists.
            ensure_dir(os.path.join(lane, "shared", "nodes", node_id))
            return lane

    def lane_commit_changes(self, node_id: str, title: str) -> str | None:
        lane = self.lane_path(node_id)
        with self._git_lock:
            rc, out, err = self._git(["status", "--porcelain"], cwd=lane)
            if rc != 0:
                raise GitWorkdirError(err)
            if out.strip() == "":
                return None
            rc, out2, err2 = self._git(["add", "-A"], cwd=lane)
            if rc != 0:
                raise GitWorkdirError(err2)
            msg = f"oteam node {node_id}: {title}"
            rc, out3, err3 = self._git(["commit", "-m", msg], cwd=lane)
            if rc != 0:
                if "nothing to commit" in (out3 + err3).lower():
                    return None
                raise GitWorkdirError(err3)
            rc, out4, err4 = self._git(["rev-parse", "HEAD"], cwd=lane)
            if rc != 0:
                raise GitWorkdirError(err4)
            return out4.strip()

    def integrate_commit_to_trunk(self, commit_hash: str, node_id: str) -> bool:
        """Cherry-pick a lane commit into trunk. Returns True if merged, False if conflict."""
        with self._git_lock:
            # Ensure trunk is on trunk branch.
            self._git(["checkout", self.cfg.trunk_branch], cwd=self.cfg.trunk_workdir)
            rc, out, err = self._git(["cherry-pick", commit_hash], cwd=self.cfg.trunk_workdir, timeout_s=600)
            if rc == 0:
                self.jlog.emit("integrate_ok", node_id=node_id, commit=commit_hash)
                return True

            # Conflict: abort and let orchestrator decide next steps.
            self.jlog.emit("integrate_conflict", node_id=node_id, commit=commit_hash, err=err, out=out)
            self._git(["cherry-pick", "--abort"], cwd=self.cfg.trunk_workdir, timeout_s=120)
            return False

    def remove_lane(self, node_id: str) -> None:
        with self._git_lock:
            lane = self.lane_path(node_id)
            if not os.path.isdir(lane):
                return
            # Force remove worktree via git (preferred).
            rc, out, err = self._git(["worktree", "remove", "--force", lane], cwd=self.cfg.project_root, timeout_s=120)
            if rc != 0:
                # Fallback to rm -rf
                shutil.rmtree(lane, ignore_errors=True)
            # Prune
            self._git(["worktree", "prune"], cwd=self.cfg.project_root, timeout_s=120)



# ---------------------------------------------------------------------------
# Prompt templates (Jinja2) written to disk at startup
# ---------------------------------------------------------------------------

STATIONMASTER_PROMPT_J2 = r"""
You are **OTeam Stationmaster**: a long-horizon project execution controller.

This system runs for weeks and the total project may be far larger than any context window.
Your job is not to remember everything verbatim; your job is to keep the project *on track* by:
- Maintaining a high-level plan as a **DAG of work packets** (fan out into parallel lanes; fan back in via join nodes).
- Continuously verifying progress against the current **SEED**.
- Detecting drift / dead-ends early and correcting by replanning, splitting work, or rolling back.
- Minimizing irreversible one-way choices: when unsure, create probes and verification steps.
- Reducing cognitive load: join lanes frequently enough that the graph does not explode.

Vocabulary guidance (IMPORTANT):
- Use *deliverable*, *work packet*, *lane*, *join*, *verification*, *evidence*.
- Avoid tying the plan to “software development” terms unless the SEED clearly describes a software project.

You will receive periodic CONTEXT PACKETS from the orchestrator. You must respond with **JSON ONLY**
matching the schema below. No markdown. No extra prose outside JSON.

The orchestrator can execute your actions, spawn lanes, merge commits, ask the customer questions, etc.

### Output JSON Schema

Top-level:

{
  "mode": "SEEDING|PLANNING|EXECUTING|VERIFYING|MIGRATING|IDLE",
  "notes": "short internal note for logs",
  "actions": [
    ...
  ]
}

Action types (all optional):

1) Ask the customer a question (to refine the SEED):
{ "type":"ask_customer", "text":"..." }

2) Record a new seed draft item (atomic requirement / constraint / preference / definition):
{ "type":"record_seed_item", "text":"...", "tags":["optional","tags"] }

3) Declare the seed ready to compile (only when enough to begin meaningful work):
{ "type":"set_seed_ready", "ready": true, "rationale":"..." }

4) Create DAG nodes (work packets). Use these for *parallelization*.
{
  "type":"create_nodes",
  "nodes":[
     {
       "node_id":"optional stable id (if omitted orchestrator assigns)",
       "kind":"work|verify|join|reflect|migrate",
       "title":"short name",
       "objective":"crisp objective with definition-of-done",
       "deliverables":["list of expected outputs (paths, artifacts, decisions)"],
       "req_links":["optional: requirement chunk ids or references"],
       "scope_hints":["optional: file paths / conceptual scope labels to avoid conflicts"],
       "priority": 0
     }
  ],
  "deps":[
     {"node_id":"child","depends_on":["parent1","parent2"]}
  ]
}

5) Request operator approval (for risky, irreversible, or expensive steps):
{ "type":"request_approval", "title":"...", "details":"...", "data":{...} }

6) Mark the current epoch as complete/idle:
{ "type":"set_epoch_status", "status":"VERIFYING|DONE|IDLE", "notes":"..." }

Rules:
- Prefer small incremental DAG expansions (<= 8 new nodes per call).
- For parallel work: split by independent deliverables, not by “roles”.
- Use join nodes to reconverge and produce consolidated summaries/evidence.
- Always include verification (verify nodes or checks) before declaring DONE.
- If SEED changes, switch to MIGRATING and propose a clean transition plan.

"""



STATIONMASTER_CRITIC_PROMPT_J2 = r"""
You are **OTeam Stationmaster Critic**.

You review the Stationmaster Actor's proposal and output a corrected proposal.
This system runs for weeks; your job is to prevent drift, dead-ends, and runaway DAG growth.

You will receive a message containing:
- CONTEXT_PACKET: the same packet the actor saw
- ACTOR_PROPOSAL_JSON: the actor's raw output (may be invalid or schema-incomplete)

You MUST output **JSON ONLY**. No markdown.

Return a JSON object in the same base schema as Stationmaster:

{
  "mode": "SEEDING|PLANNING|EXECUTING|VERIFYING|MIGRATING|IDLE",
  "notes": "short critic note for logs",
  "actions": [...],

  "critic_verdict": "accept|revise|reject",
  "critic_issues": ["optional list of issues found"],
  "lessons": ["optional: guardrails/lessons to record"]
}

Guidance / invariants you must enforce:
- If pending seed draft items exist while an epoch is active (not DONE/IDLE), the correct mode is MIGRATING
  and actions should prioritize a safe transition (questions, seed ledger items, requesting seed compile, migration nodes).
- Prefer small incremental expansions: <= 8 new nodes per tick (and <= 12 deps).
- Parallelism is for independent deliverables; avoid role-pipelines.
- Ensure fan-in exists: when fanning out, create (or plan) join nodes to reconverge and reduce the graph.
- Require verification: do not allow DONE without verify nodes and explicit evidence.
- Keep vocabulary generic (deliverable/evidence/work packet). Avoid software-dev terms unless SEED demands them.
- If the actor proposal is unsalvageable, set critic_verdict="reject" and output a minimal safe plan:
  - ask_customer for missing info and/or create 1-2 reflect nodes to replan/diagnose.

Your output should be either:
- The actor proposal unchanged (critic_verdict="accept"),
- A minimally edited/corrected proposal (critic_verdict="revise"),
- A safe fallback (critic_verdict="reject").

"""

WORKER_PROMPT_J2 = r"""
You are **OTeam Lane Worker**.

You operate inside a dedicated workspace that is a git-backed snapshot (“lane”) of the project.
Your mission is to complete **one work packet** only.

You MUST:
- Stay within the objective and deliverables described in this work packet.
- Write merge-friendly outputs inside `shared/nodes/{{ node_id }}/` whenever possible.
- Leave the workspace in a coherent state: if you touched files, ensure they make sense together.
- Prefer producing artifacts + evidence over long explanations.
- If you discover the packet is too big or blocked, stop early and propose follow-up packets.

You MUST NOT:
- Try to “finish the whole project”.
- Spawn parallel role pipelines inside this packet.
- Assume you have the whole SEED in context; instead, rely on local repo files and precise searching.

When finished, respond with **JSON ONLY**:

{
  "status": "done|blocked|needs_input|failed",
  "summary": "short summary (for logs)",
  "outputs": [
    {"path":"relative/path", "desc":"what it is"}
  ],
  "scope_touched": ["paths or labels"],
  "suggested_checks": ["shell commands the orchestrator can run in trunk to verify"],
  "followups": [
    {"title":"...", "objective":"...", "kind":"work|verify|reflect", "priority":0}
  ],
  "lessons": ["optional: lessons/guardrails discovered"]
}

If you need customer input, set status="needs_input" and include a follow-up with kind="reflect" describing the question.

Work packet metadata:
- node_id: {{ node_id }}
- title: {{ node_title }}

Objective:
{{ objective }}

Deliverables:
{% for d in deliverables %}
- {{ d }}
{% endfor %}

"""


VERIFIER_PROMPT_J2 = r"""
You are **OTeam Verifier**.

Your job is to verify that deliverables satisfy the current SEED.
Be skeptical. Look for missing acceptance criteria, contradictions, and unverified claims.

You have access to the repository workspace (usually trunk) and can inspect files and run commands.

Respond with **JSON ONLY**:

{
  "status": "pass|fail|inconclusive",
  "summary": "short verdict",
  "evidence": [
     {"claim":"...", "evidence_path":"...", "notes":"..."}
  ],
  "gaps": [
     {"gap":"...", "suggested_node": {"title":"...", "objective":"...", "kind":"work|verify|reflect", "priority":0}}
  ],
  "suggested_checks": ["commands"]
}

"""


def write_prompt_templates(cfg: Config) -> dict[str, str]:
    """Write prompt templates to cfg.prompts_dir and return their absolute paths."""
    ensure_dir(cfg.prompts_dir)
    paths: dict[str, str] = {}
    mapping = {
        "stationmaster.j2": STATIONMASTER_PROMPT_J2,
        "stationmaster_critic.j2": STATIONMASTER_CRITIC_PROMPT_J2,
        "worker.j2": WORKER_PROMPT_J2,
        "verifier.j2": VERIFIER_PROMPT_J2,
    }
    for name, content in mapping.items():
        p = os.path.join(cfg.prompts_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        paths[name] = p
    return paths



# ---------------------------------------------------------------------------
# Repo indexing (optional, lightweight) for fast search
# ---------------------------------------------------------------------------

class RepoIndexer(threading.Thread):
    def __init__(self, cfg: Config, db: StateDB, logger: logging.Logger, jlog: JsonlEventLog):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.db = db
        self.logger = logger
        self.jlog = jlog
        self._stop = threading.Event()
        self._last_scan = 0.0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover (long-running)
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:
                self.logger.error("repo indexer error: %s", e)
                self.jlog.emit("repo_index_error", err=str(e), tb=traceback.format_exc())
            self._stop.wait(self.cfg.repo_index_tick_s)

    def tick(self) -> None:
        root = self.cfg.trunk_workdir
        if not os.path.isdir(root):
            return
        # Walk a bounded number of files per tick.
        n_indexed = 0
        t0 = now_ts()
        for dirpath, dirnames, filenames in os.walk(root):
            # prune
            dn = set(dirnames)
            for bad in [".git", ".oteam", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"]:
                if bad in dn:
                    dirnames.remove(bad)
            # skip shared? no, include shared.
            for fn in filenames:
                if n_indexed >= self.cfg.repo_index_max_files_per_tick:
                    break
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, root)
                # Skip binary-ish extensions
                if any(rel.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz", ".7z", ".bin"]):
                    continue
                try:
                    st = os.stat(path)
                except FileNotFoundError:
                    continue
                if st.st_size > self.cfg.repo_index_max_file_bytes:
                    continue
                mtime = st.st_mtime
                if not self.db.repo_index_needs_update(rel, mtime, st.st_size):
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    # naive text check
                    if b"\x00" in data:
                        continue
                    txt = data.decode("utf-8", errors="replace")
                except Exception:
                    continue
                sha = sha256_bytes(txt.encode("utf-8"))
                self.db.repo_index_upsert(rel, txt, mtime, st.st_size, sha)
                n_indexed += 1
            if n_indexed >= self.cfg.repo_index_max_files_per_tick:
                break
        dt = now_ts() - t0
        if n_indexed:
            self.jlog.emit("repo_index_tick", indexed=n_indexed, seconds=dt)



def git_commit_all(gm: GitWorkdirManager, cwd: str, message: str) -> str:
    """Commit all changes in cwd (a git worktree). Returns HEAD commit hash."""
    rc, out, err = gm._git(["add", "-A"], cwd=cwd)
    if rc != 0:
        raise GitWorkdirError(err)
    rc, out, err = gm._git(["commit", "-m", message], cwd=cwd)
    if rc != 0 and "nothing to commit" not in (out + err).lower():
        raise GitWorkdirError(err)
    rc, out, err = gm._git(["rev-parse", "HEAD"], cwd=cwd)
    if rc != 0:
        raise GitWorkdirError(err)
    return out.strip()


class SeedManager:
    def __init__(self, cfg: Config, db: StateDB, git: GitWorkdirManager, logger: logging.Logger, jlog: JsonlEventLog):
        self.cfg = cfg
        self.db = db
        self.git = git
        self.logger = logger
        self.jlog = jlog

    @property
    def seed_path(self) -> str:
        return os.path.join(self.cfg.trunk_workdir, "SEED.md")

    def load_seed_text(self) -> str | None:
        if os.path.exists(self.seed_path):
            try:
                with open(self.seed_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return None
        return None

    def has_active_seed(self) -> bool:
        return self.db.get_active_seed_version_id() is not None

    def last_compiled_draft_item_id(self) -> int:
        v = self.db.meta_get("seed_last_compiled_item_id", "0") or "0"
        try:
            return int(v)
        except Exception:
            return 0

    def set_last_compiled_draft_item_id(self, item_id: int) -> None:
        self.db.meta_set("seed_last_compiled_item_id", str(item_id))

    def pending_draft_items(self) -> list[sqlite3.Row]:
        last = self.last_compiled_draft_item_id()
        return self.db.list_seed_draft_items(since_id=last, limit=10_000)

    def compile_seed_from_draft(self, reason: str = "seed compile") -> str:
        """Generate SEED.md from seed_draft_items and commit to trunk.
        Returns seed_version_id.
        """
        items = self.db.list_seed_draft_items(limit=50_000)
        if not items:
            raise RuntimeError("No seed draft items to compile.")
        # Deterministic compilation: we do NOT rely on the LLM to regenerate the whole seed.
        # We preserve everything as atomic bullet points.
        lines: list[str] = []
        lines.append("# SEED")
        lines.append("")
        lines.append("> This SEED is generated by OTeam from an atomic ledger of requirements.")
        lines.append("> Edit by adding new messages/requirements; the system will generate a new SEED version.")
        lines.append("")
        lines.append("## Ledger (atomic requirements, constraints, definitions, preferences)")
        lines.append("")
        for r in items:
            status = r["status"]
            if status.upper() == "REJECTED":
                continue
            tags = []
            try:
                tags = json.loads(r["tags_json"] or "[]")
            except Exception:
                tags = []
            tag_str = (" [" + ", ".join(tags) + "]") if tags else ""
            lines.append(f"- ({r['item_id']}) **{r['source']}**{tag_str}: {r['text']}")
        lines.append("")
        lines.append("## Working assumptions")
        lines.append("")
        lines.append("- Unless the ledger states otherwise, deliverables should be placed under `shared/`.")
        lines.append("- Verification should produce evidence (paths, logs, checks) also stored under `shared/`.")
        lines.append("")

        seed_text = "\n".join(lines)

        # Write SEED.md (latest) + snapshot.
        ensure_dir(os.path.join(self.cfg.trunk_workdir, "shared", "seed"))
        snap_name = f"SEED.{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.md"
        snap_path = os.path.join(self.cfg.trunk_workdir, "shared", "seed", snap_name)

        with open(self.seed_path, "w", encoding="utf-8") as f:
            f.write(seed_text)
        with open(snap_path, "w", encoding="utf-8") as f:
            f.write(seed_text)

        # Commit.
        msg = f"oteam: update SEED ({reason})"
        commit = git_commit_all(self.git, self.cfg.trunk_workdir, msg)

        seed_version_id = self.db.insert_seed_version(seed_text=seed_text, commit_hash=commit, seed_path="SEED.md")
        self.db.set_active_seed_version(seed_version_id)

        # Chunk SEED into requirement_chunks for retrieval.
        chunks = chunk_markdown_by_headings(seed_text, seed_version_id=seed_version_id)
        self.db.replace_requirement_chunks(seed_version_id, chunks)

        # Update last compiled draft item id.
        last_item_id = int(items[-1]["item_id"])
        self.set_last_compiled_draft_item_id(last_item_id)

        self.jlog.emit("seed_compiled", seed_version_id=seed_version_id, commit=commit, items=last_item_id)
        return seed_version_id


def chunk_markdown_by_headings(md: str, seed_version_id: str) -> list[dict[str, t.Any]]:
    """Very simple chunker: split on '## ' headings."""
    lines = md.splitlines()
    chunks: list[dict[str, t.Any]] = []
    cur_heading = "ROOT"
    cur_lines: list[str] = []
    def flush() -> None:
        nonlocal cur_heading, cur_lines
        content = "\n".join(cur_lines).strip()
        if content:
            cid = sha256_bytes((seed_version_id + cur_heading + content).encode("utf-8"))[:20]
            chunks.append(
                {
                    "chunk_id": f"R-{cid}",
                    "heading": cur_heading,
                    "content": content,
                    "token_est": max(1, len(content) // 4),
                }
            )
        cur_lines = []

    for ln in lines:
        if ln.startswith("## "):
            flush()
            cur_heading = ln[3:].strip() or "SECTION"
            cur_lines = [ln]
        else:
            cur_lines.append(ln)
    flush()
    return chunks



# ---------------------------------------------------------------------------
# OpenHands runtime wrapper
# ---------------------------------------------------------------------------

class OpenHandsRuntime:
    def __init__(self, cfg: Config, logger: logging.Logger, jlog: JsonlEventLog, prompt_paths: dict[str, str]):
        self.cfg = cfg
        self.logger = logger
        self.jlog = jlog
        self.prompt_paths = prompt_paths

        if not OH_AVAILABLE:
            raise RuntimeError(
                "OpenHands SDK is not available. Install it before running this daemon. "
                "See https://docs.openhands.dev/sdk for installation."
            )
        if not cfg.llm_api_key:
            raise RuntimeError("LLM_API_KEY is required.")

        self.oh_logger = oh_get_logger(__name__) if OH_AVAILABLE else None
        self.llm = self._make_llm(cfg)

    def _make_llm(self, cfg: Config) -> "LLM":
        api_key = cfg.llm_api_key or ""
        if SecretStr is not None:
            key_obj = SecretStr(api_key)
        else:
            key_obj = api_key  # type: ignore
        llm = LLM(
            usage_id="agent",
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
            api_key=key_obj,  # type: ignore[arg-type]
        )
        return llm

    def _make_condenser(self) -> "LLMSummarizingCondenser":
        # Use the same model; usage_id separates metrics.
        try:
            llm2 = self.llm.model_copy(update={"usage_id": "condenser"})  # type: ignore[attr-defined]
        except Exception:
            llm2 = self.llm
        condenser = LLMSummarizingCondenser(
            llm=llm2, max_size=self.cfg.condenser_max_events, keep_first=self.cfg.condenser_keep_first
        )
        return condenser

    def _tools(self, *, allow_edit: bool) -> list["Tool"]:
        tools: list[Tool] = []
        # Terminal is essential even for non-software: it allows inspection and running local commands.
        tools.append(Tool(name=TerminalTool.name))
        if allow_edit:
            tools.append(Tool(name=FileEditorTool.name))
        if TaskTrackerTool is not None and allow_edit:
            tools.append(Tool(name=TaskTrackerTool.name))
        return tools

    def run_conversation(
        self,
        *,
        conversation_id: str,
        agent: "Agent",
        workspace: str,
        user_message: str,
        persistence_dir: str,
    ) -> str:
        """Send one user_message and run the agent loop. Returns last assistant text."""
        last_assistant: list[str] = []

        def cb(event: "Event") -> None:
            try:
                if isinstance(event, LLMConvertibleEvent):
                    msg = event.to_llm_message()
                    role = getattr(msg, "role", None)
                    if role is None and isinstance(msg, dict):
                        role = msg.get("role")
                    if role == "assistant":
                        try:
                            content = "".join(content_to_str(msg.content))  # type: ignore[attr-defined]
                        except Exception:
                            # Fallback: best-effort string conversion.
                            content = str(getattr(msg, "content", msg))
                        last_assistant.append(content)
            except Exception:
                # Don't let callback failures kill the run.
                return

        conv = Conversation(
            agent=agent,
            callbacks=[cb],
            persistence_dir=persistence_dir,
            conversation_id=conversation_id,
            workspace=workspace,
        )
        conv.send_message(user_message)
        self.jlog.emit(
            "oh_run_start",
            conversation_id=conversation_id,
            workspace=os.path.abspath(workspace),
            user_msg_tail=clamp(user_message, 800),
        )
        try:
            conv.run()
        except Exception as e:
            self.jlog.emit("oh_run_error", conversation_id=conversation_id, err=str(e), tb=traceback.format_exc())
            raise
        self.jlog.emit(
            "oh_run_done",
            conversation_id=conversation_id,
            assistant_tail=clamp(last_assistant[-1] if last_assistant else "", 800),
        )
        return last_assistant[-1] if last_assistant else ""
    def _stationmaster_actor(self, user_message: str, conversation_id: str = "stationmaster_actor") -> str:
        """Stationmaster Actor: proposes actions (DAG edits, seed actions, etc.)."""
        condenser = self._make_condenser()
        tools = self._tools(allow_edit=False)  # planning; no file edits
        agent = Agent(
            llm=self.llm,
            tools=tools,
            condenser=condenser,
            system_prompt_filename=self.prompt_paths["stationmaster.j2"],
            system_prompt_kwargs={},
            agent_context=AgentContext(skills=[Skill(name="Always respond with JSON only.")]),
        )
        return self.run_conversation(
            conversation_id=conversation_id,
            agent=agent,
            workspace=self.cfg.trunk_workdir,
            user_message=user_message,
            persistence_dir=self.cfg.conversations_dir,
        )

    def _stationmaster_critic(
        self,
        context_packet: str,
        actor_output_text: str,
        conversation_id: str = "stationmaster_critic",
    ) -> str:
        """Stationmaster Critic: audits the actor proposal and returns a corrected proposal."""
        condenser = self._make_condenser()
        tools = self._tools(allow_edit=False)  # critique; no file edits
        agent = Agent(
            llm=self.llm,
            tools=tools,
            condenser=condenser,
            system_prompt_filename=self.prompt_paths["stationmaster_critic.j2"],
            system_prompt_kwargs={},
            agent_context=AgentContext(skills=[Skill(name="Always respond with JSON only.")]),
        )

        critic_packet = (
            "CRITIC_INPUT\n"
            "CONTEXT_PACKET:\n"
            + context_packet
            + "\n\nACTOR_PROPOSAL_JSON:\n"
            + actor_output_text
            + "\n"
        )
        return self.run_conversation(
            conversation_id=conversation_id,
            agent=agent,
            workspace=self.cfg.trunk_workdir,
            user_message=critic_packet,
            persistence_dir=self.cfg.conversations_dir,
        )

    def stationmaster_step(self, context_packet: str) -> str:
        """Actor-critic Stationmaster.

        Flow:
        - Actor proposes a Stationmaster output JSON.
        - Critic validates/edits it (and may reject).
        - If rejected, a revision loop may be triggered (configurable).

        Returns the final Stationmaster JSON (critic output preferred).
        """
        if not getattr(self.cfg, "stationmaster_actor_critic", True):
            return self._stationmaster_actor(context_packet, conversation_id="stationmaster")

        max_retries = max(0, int(getattr(self.cfg, "stationmaster_ac_max_retries", 1)))

        last_actor = self._stationmaster_actor(context_packet, conversation_id="stationmaster_actor")

        for attempt in range(max_retries + 1):
            critic_out = self._stationmaster_critic(context_packet, last_actor, conversation_id="stationmaster_critic")
            cobj = extract_first_json_object(critic_out) or {}
            verdict = str(cobj.get("critic_verdict") or "").strip().lower()

            # Default to accept if verdict missing.
            if verdict in ("", "accept", "revise"):
                self.jlog.emit(
                    "stationmaster_actor_critic",
                    verdict=verdict or "accept",
                    attempt=attempt,
                    actor_tail=clamp(last_actor, 900),
                    critic_tail=clamp(critic_out, 900),
                )
                return critic_out if extract_first_json_object(critic_out) else last_actor

            issues = cobj.get("critic_issues")
            issues_list = [str(x) for x in issues] if isinstance(issues, list) else []
            self.jlog.emit(
                "stationmaster_actor_critic_reject",
                attempt=attempt,
                issues=issues_list[:25],
                actor_tail=clamp(last_actor, 900),
                critic_tail=clamp(critic_out, 900),
            )

            if attempt >= max_retries:
                # If critic gave a valid stationmaster JSON despite rejection, use it; otherwise keep actor.
                return critic_out if extract_first_json_object(critic_out) else last_actor

            revision_req = {"attempt": attempt + 1, "critic_issues": issues_list[:25]}
            revision_msg = context_packet + "\n\nREVISION_REQUEST:\n" + json.dumps(revision_req, ensure_ascii=False)
            last_actor = self._stationmaster_actor(revision_msg, conversation_id="stationmaster_actor")

        return last_actor


    def worker_step(self, node_id: str, node_title: str, objective: str, deliverables: list[str], workspace: str, conversation_id: str, context_packet: str | None = None) -> str:
        condenser = self._make_condenser()
        tools = self._tools(allow_edit=True)
        agent = Agent(
            llm=self.llm,
            tools=tools,
            condenser=condenser,
            system_prompt_filename=self.prompt_paths["worker.j2"],
            system_prompt_kwargs={
                "node_id": node_id,
                "node_title": node_title,
                "objective": objective,
                "deliverables": deliverables,
            },
            agent_context=AgentContext(skills=[Skill(name="Always respond with JSON only.")]),
        )
        return self.run_conversation(
            conversation_id=conversation_id,
            agent=agent,
            workspace=workspace,
            user_message=context_packet or "Begin work now. Remember: respond with JSON only when finished.",
            persistence_dir=self.cfg.conversations_dir,
        )

    def verifier_step(self, context_packet: str, conversation_id: str = "verifier") -> str:
        condenser = self._make_condenser()
        tools = self._tools(allow_edit=False) + [Tool(name=FileEditorTool.name)]
        agent = Agent(
            llm=self.llm,
            tools=tools,
            condenser=condenser,
            system_prompt_filename=self.prompt_paths["verifier.j2"],
            system_prompt_kwargs={},
            agent_context=AgentContext(skills=[Skill(name="Always respond with JSON only.")]),
        )
        return self.run_conversation(
            conversation_id=conversation_id,
            agent=agent,
            workspace=self.cfg.trunk_workdir,
            user_message=context_packet,
            persistence_dir=self.cfg.conversations_dir,
        )



# ---------------------------------------------------------------------------
# Customer HTTP bridge (simple pull-based outbox)
# ---------------------------------------------------------------------------

class ServiceContext(t.NamedTuple):
    cfg: Config
    db: StateDB
    git: GitWorkdirManager
    seed: SeedManager
    runtime: OpenHandsRuntime | None
    logger: logging.Logger
    jlog: JsonlEventLog


_SERVICE_CTX: ServiceContext | None = None


class CustomerHTTPHandler(http.server.BaseHTTPRequestHandler):
    server_version = "oteam-customer-http/1.0"

    def _send(self, code: int, body: dict[str, t.Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(n) if n > 0 else b""

    def do_GET(self) -> None:  # noqa: N802
        ctx = _SERVICE_CTX
        if ctx is None:
            self._send(500, {"ok": False, "error": "service not ready"})
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/healthz":
            self._send(200, {"ok": True})
            return
        if parsed.path == "/customer/outbox":
            qs = urllib.parse.parse_qs(parsed.query)
            customer_id = (qs.get("customer_id") or ["default"])[0]
            ack = (qs.get("ack") or ["1"])[0] == "1"
            rows = ctx.db.dequeue_outbox(customer_id=customer_id, ack=ack)
            self._send(200, {"ok": True, "messages": [{"ts": r["ts"], "text": r["text"]} for r in rows]})
            return
        if parsed.path == "/customer/status":
            # Minimal status for customer-facing UI.
            seed_id = ctx.db.get_active_seed_version_id()
            epoch_id = ctx.db.get_active_epoch_id()
            epoch = ctx.db.get_epoch(epoch_id) if epoch_id else None
            self._send(
                200,
                {
                    "ok": True,
                    "seed_version_id": seed_id,
                    "epoch": dict(epoch) if epoch else None,
                },
            )
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        ctx = _SERVICE_CTX
        if ctx is None:
            self._send(500, {"ok": False, "error": "service not ready"})
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/customer/inbound":
            self._send(404, {"ok": False, "error": "not found"})
            return

        body = self._read_body()
        text = None
        customer_id = "default"
        try:
            if self.headers.get("Content-Type", "").startswith("application/json"):
                obj = json.loads(body.decode("utf-8"))
                customer_id = str(obj.get("customer_id") or customer_id)
                text = obj.get("text")
            else:
                form = urllib.parse.parse_qs(body.decode("utf-8"))
                customer_id = (form.get("customer_id") or [customer_id])[0]
                text = (form.get("text") or [None])[0]
        except Exception:
            text = None

        if not text:
            self._send(400, {"ok": False, "error": "missing text"})
            return

        ctx.db.add_message("customer", customer_id, str(text), raw={"path": parsed.path})
        ctx.db.add_seed_draft_item("customer", str(text))
        ctx.jlog.emit("customer_inbound", customer_id=customer_id, text_tail=clamp(str(text), 400))
        self._send(200, {"ok": True})

    def log_message(self, fmt: str, *args: t.Any) -> None:  # noqa: A003
        # Silence default HTTP server logging; we use our own logs.
        return


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True



# ---------------------------------------------------------------------------
# Telegram operator interface (polling)
# ---------------------------------------------------------------------------

class TelegramBot(threading.Thread):
    def __init__(self, ctx: ServiceContext):
        super().__init__(daemon=True)
        self.ctx = ctx
        self._stop = threading.Event()
        self.offset = None  # will load from DB meta

    def stop(self) -> None:
        self._stop.set()

    def _api(self, method: str, params: dict[str, t.Any] | None = None) -> dict[str, t.Any]:
        token = self.ctx.cfg.telegram_bot_token
        if not token:
            raise RuntimeError("Telegram token not set")
        url = f"https://api.telegram.org/bot{token}/{method}"
        params = params or {}
        if requests is not None:
            resp = requests.post(url, json=params, timeout=30)
            resp.raise_for_status()
            return t.cast(dict[str, t.Any], resp.json())
        # Fallback: urllib
        data = json.dumps(params).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})  # type: ignore
        with urllib.request.urlopen(req, timeout=30) as r:  # type: ignore
            return t.cast(dict[str, t.Any], json.loads(r.read().decode("utf-8")))

    def send(self, chat_id: str, text: str) -> None:
        try:
            self._api("sendMessage", {"chat_id": chat_id, "text": text})
        except Exception as e:
            self.ctx.logger.error("telegram send failed: %s", e)

    def run(self) -> None:  # pragma: no cover (long-running)
        if not self.ctx.cfg.telegram_bot_token:
            return
        # restore offset
        off = self.ctx.db.meta_get("telegram_update_offset")
        if off:
            try:
                self.offset = int(off)
            except Exception:
                self.offset = None

        self.ctx.logger.info("telegram bot started")
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as e:
                self.ctx.logger.error("telegram poll error: %s", e)
                self.ctx.jlog.emit("telegram_error", err=str(e), tb=traceback.format_exc())
                time.sleep(2.0)

    def _poll_once(self) -> None:
        params: dict[str, t.Any] = {"timeout": 25}
        if self.offset is not None:
            params["offset"] = self.offset
        res = self._api("getUpdates", params)
        if not res.get("ok"):
            time.sleep(1.0)
            return
        updates = res.get("result") or []
        for upd in updates:
            try:
                upd_id = upd.get("update_id")
                if isinstance(upd_id, int):
                    self.offset = upd_id + 1
                    self.ctx.db.meta_set("telegram_update_offset", str(self.offset))
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id"))
                text = msg.get("text") or ""
                if not text:
                    continue
                self._handle_message(chat_id, text, raw=upd)
            except Exception as e:
                self.ctx.logger.error("telegram update handling error: %s", e)
        # no sleep: long-poll already

    def _is_admin(self, chat_id: str) -> bool:
        if not self.ctx.cfg.telegram_admin_chat_id:
            return True
        return str(chat_id) == str(self.ctx.cfg.telegram_admin_chat_id)

    def _handle_message(self, chat_id: str, text: str, raw: dict[str, t.Any]) -> None:
        text = text.strip()
        self.ctx.jlog.emit("telegram_inbound", chat_id=chat_id, text_tail=clamp(text, 400))
        if not self._is_admin(chat_id):
            # Treat as customer input by default.
            self.ctx.db.add_message("customer", f"telegram:{chat_id}", text, raw=raw)
            self.ctx.db.add_seed_draft_item("customer", text, tags=["telegram"])
            self.send(chat_id, "Received. OTeam will incorporate this into the project SEED / plan.")
            return

        # Operator command
        self.ctx.db.add_message("operator", f"telegram:{chat_id}", text, raw=raw)

        if text in ("/help", "help"):
            self.send(chat_id, self._help_text())
            return
        if text.startswith("/pause"):
            self.ctx.db.meta_set("paused", "1")
            self.send(chat_id, "Paused (will stop launching new lanes).")
            return
        if text.startswith("/resume"):
            self.ctx.db.meta_set("paused", "0")
            self.send(chat_id, "Resumed.")
            return
        if text.startswith("/status"):
            self.send(chat_id, self._status_text())
            return
        if text.startswith("/seed_compile"):
            self.ctx.db.meta_set("force_seed_compile", "1")
            self.send(chat_id, "Requested SEED compile.")
            return
        if text.startswith("/seed_add"):
            payload = text[len("/seed_add") :].strip()
            if not payload:
                self.send(chat_id, "Usage: /seed_add <requirement text>")
                return
            self.ctx.db.add_seed_draft_item("operator", payload, tags=["operator"])
            self.send(chat_id, "Added to seed draft ledger.")
            return
        if text.startswith("/seed"):
            self.send(chat_id, self._seed_text())
            return
        if text.startswith("/graph"):
            self.send(chat_id, self._graph_text())
            return
        if text.startswith("/nodes"):
            parts = text.split(maxsplit=1)
            status = parts[1].strip().upper() if len(parts) > 1 else None
            self.send(chat_id, self._nodes_text(status))
            return
        if text.startswith("/node"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                self.send(chat_id, "Usage: /node <node_id>")
                return
            self.send(chat_id, self._node_text(parts[1].strip()))
            return
        if text.startswith("/approvals"):
            self.send(chat_id, self._approvals_text())
            return
        if text.startswith("/approve") or text.startswith("/reject"):
            parts = text.split(maxsplit=2)
            if len(parts) < 2:
                self.send(chat_id, "Usage: /approve <approval_id> [reason]  OR  /reject <approval_id> [reason]")
                return
            appr_id = parts[1]
            reason = parts[2] if len(parts) > 2 else ""
            decision = "APPROVED" if text.startswith("/approve") else "REJECTED"
            self.ctx.db.decide_approval(appr_id, decision, by=f"telegram:{chat_id}", reason=reason or None)
            self.send(chat_id, f"{decision}: {appr_id}")
            return
        if text.startswith("/kick"):
            # force stationmaster tick
            self.ctx.db.meta_set("force_stationmaster", "1")
            self.send(chat_id, "Requested stationmaster tick.")
            return

        # Fallback: treat as seed draft input.
        self.ctx.db.add_seed_draft_item("operator", text, tags=["operator", "freeform"])
        self.send(chat_id, "Recorded as seed draft input. Use /seed to view state.")

    def _help_text(self) -> str:
        return (
            "OTeam / OpenHands trainyard commands:\n"
            "/status - show current state\n"
            "/seed - show active seed + pending draft count\n"
            "/seed_add <text> - add requirement to seed draft\n"
            "/seed_compile - force compile new SEED\n"
            "/graph - DAG summary\n"
            "/nodes [STATUS] - list nodes\n"
            "/node <id> - node detail\n"
            "/approvals - pending approvals\n"
            "/approve <id> [reason]\n"
            "/reject <id> [reason]\n"
            "/pause | /resume\n"
            "/kick - force stationmaster step\n"
        )

    def _status_text(self) -> str:
        seed_id = self.ctx.db.get_active_seed_version_id()
        epoch_id = self.ctx.db.get_active_epoch_id()
        epoch = self.ctx.db.get_epoch(epoch_id) if epoch_id else None
        paused = self.ctx.db.meta_get("paused", "0")
        parts = []
        parts.append(f"paused={paused}")
        parts.append(f"seed={seed_id}")
        if epoch:
            parts.append(f"epoch={epoch['epoch_id']} status={epoch['status']} start={epoch['ts_start']}")
        else:
            parts.append("epoch=<none>")
        return "\n".join(parts)

    def _seed_text(self) -> str:
        seed_id = self.ctx.db.get_active_seed_version_id()
        latest = self.ctx.db.get_latest_seed_version()
        pending = len(self.ctx.seed.pending_draft_items())
        if not latest:
            return f"No compiled SEED yet. Pending draft items: {pending}"
        summary = (latest["summary"] or "").strip()
        return f"active_seed={seed_id}\nlatest_seed={latest['seed_version_id']}\ncommit={latest['commit_hash']}\npend_draft_items={pending}\nsummary={summary or '<none>'}"

    def _graph_text(self) -> str:
        epoch_id = self.ctx.db.get_active_epoch_id()
        if not epoch_id:
            return "No active epoch."
        nodes = self.ctx.db.list_nodes(epoch_id, limit=2000)
        counts: dict[str, int] = {}
        for n in nodes:
            counts[n["status"]] = counts.get(n["status"], 0) + 1
        # show ready/running/done
        parts = [f"epoch={epoch_id} nodes={len(nodes)}"]
        for k in ["PENDING", "READY", "RUNNING", "DONE", "FAILED", "BLOCKED", "STALE"]:
            if k in counts:
                parts.append(f"{k}={counts[k]}")
        return " ".join(parts)

    def _nodes_text(self, status: str | None) -> str:
        epoch_id = self.ctx.db.get_active_epoch_id()
        if not epoch_id:
            return "No active epoch."
        nodes = self.ctx.db.list_nodes(epoch_id, status=status, limit=50)
        if not nodes:
            return "No nodes."
        lines = []
        for n in nodes:
            lines.append(f"{n['node_id']} [{n['status']}] ({n['kind']}) {clamp(n['title'], 50)}")
        return "\n".join(lines)

    def _node_text(self, node_id: str) -> str:
        n = self.ctx.db.get_node(node_id)
        if not n:
            return f"Unknown node {node_id}"
        deps = self.ctx.db.list_deps(node_id)
        return (
            f"{n['node_id']} status={n['status']} kind={n['kind']}\n"
            f"title: {n['title']}\n"
            f"objective: {clamp(n['objective'], 400)}\n"
            f"deps: {', '.join(deps) if deps else '<none>'}\n"
            f"attempts: {n['attempt_count']}\n"
            f"base_commit: {n.get('base_commit')}\nresult_commit: {n.get('result_commit')}\n"
            f"summary: {clamp(n.get('summary') or '', 400)}"
        )

    def _approvals_text(self) -> str:
        apps = self.ctx.db.list_approvals(status="PENDING", limit=20)
        if not apps:
            return "No pending approvals."
        lines = []
        for a in apps:
            lines.append(f"{a['approval_id']} - {clamp(a.get('title') or '', 60)}")
        return "\n".join(lines)



# ---------------------------------------------------------------------------
# Orchestrator (continuous loop)
# ---------------------------------------------------------------------------

import concurrent.futures  # placed here to keep imports together enough


class Orchestrator:
    def __init__(self, ctx: ServiceContext):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.db = ctx.db
        self.git = ctx.git
        self.seed = ctx.seed
        self.runtime = ctx.runtime
        self.logger = ctx.logger
        self.jlog = ctx.jlog

        self._stop = threading.Event()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.node_workers)
        self._running: dict[str, concurrent.futures.Future] = {}
        self._last_stationmaster_ts = 0.0
        self._last_stationmaster_msg_id = int(self.db.meta_get("stationmaster_last_msg_id", "0") or "0")
        self._seed_ready_flag = False  # stationmaster requested compile
        self._seed_ready_reason = ""

    def stop(self) -> None:
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ----------------- context packets -----------------

    def _customer_ids(self) -> list[str]:
        rows = self.db.query("SELECT DISTINCT sender FROM messages WHERE channel='customer' ORDER BY msg_id DESC LIMIT 20")
        ids = [r["sender"] for r in rows if r["sender"]]
        return ids or ["default"]

    def _node_counts(self, epoch_id: str) -> dict[str, int]:
        nodes = self.db.list_nodes(epoch_id, limit=5000)
        counts: dict[str, int] = {}
        for n in nodes:
            counts[n["status"]] = counts.get(n["status"], 0) + 1
        return counts

    def build_stationmaster_context(self, mode_hint: str | None = None) -> str:
        seed_id = self.db.get_active_seed_version_id()
        epoch_id = self.db.get_active_epoch_id()
        epoch = self.db.get_epoch(epoch_id) if epoch_id else None

        pending_seed = self.seed.pending_draft_items()
        pending_seed_count = len(pending_seed)

        # Recent customer messages since last stationmaster tick
        recent_msgs = self.db.list_messages(channel="customer", since_id=self._last_stationmaster_msg_id, limit=20)
        if recent_msgs:
            self._last_stationmaster_msg_id = int(recent_msgs[-1]["msg_id"])
            self.db.meta_set("stationmaster_last_msg_id", str(self._last_stationmaster_msg_id))

        # Recent node summaries
        recent_nodes: list[dict[str, t.Any]] = []
        if epoch_id:
            all_nodes = self.db.list_nodes(epoch_id, limit=200)
            # take last done/failed
            for n in reversed(all_nodes):
                if n["status"] in ("DONE", "FAILED", "BLOCKED"):
                    recent_nodes.append(n)
                if len(recent_nodes) >= 8:
                    break
            recent_nodes.reverse()

        counts = self._node_counts(epoch_id) if epoch_id else {}

        forced = self.db.meta_get("force_stationmaster", "0") == "1"
        self.db.meta_set("force_stationmaster", "0")

        # Determine mode
        mode = mode_hint
        if not seed_id:
            mode = "SEEDING"
        elif pending_seed_count > 0 and epoch and epoch["status"] not in ("DONE", "IDLE"):
            # There are pending seed changes while working.
            mode = "MIGRATING"
        elif epoch is None:
            mode = "PLANNING"
        else:
            mode = epoch["status"] or "PLANNING"
            if mode not in ("PLANNING", "EXECUTING", "VERIFYING", "IDLE", "DONE", "SUPERSEDED"):
                mode = "PLANNING"

        # Build packet (text, not JSON).
        buf: list[str] = []
        buf.append("CONTEXT_PACKET")
        buf.append(f"mode_hint={mode}")
        buf.append(f"active_seed_version_id={seed_id}")
        if seed_id:
            sv = self.db.get_seed_version(seed_id)
            if sv:
                buf.append(f"seed_commit={sv['commit_hash']}")
                buf.append(f"seed_path={sv['seed_path']}")
        buf.append(f"pending_seed_draft_items={pending_seed_count}")
        if pending_seed_count:
            buf.append("pending_seed_items_preview:")
            for r in pending_seed[-10:]:
                buf.append(f"- ({r['item_id']}) {r['source']}: {clamp(r['text'], 200)}")
        if epoch:
            buf.append(f"epoch_id={epoch['epoch_id']} epoch_status={epoch['status']} base_commit={epoch['base_commit']}")
            buf.append(f"node_counts={stable_json(counts)}")
        else:
            buf.append("epoch=<none>")
        if recent_msgs:
            buf.append("recent_customer_messages:")
            for m in recent_msgs[-10:]:
                buf.append(f"- ({m['msg_id']}) {m['sender']}: {clamp(m['text'], 240)}")
        if recent_nodes:
            buf.append("recent_node_outcomes:")
            for n in recent_nodes:
                buf.append(
                    f"- {n['node_id']} [{n['status']}] {clamp(n['title'], 40)} :: {clamp(n.get('summary') or '', 140)}"
                )
        # Pending approvals snapshot
        pending_apps = self.db.list_approvals(status="PENDING", limit=5)
        if pending_apps:
            buf.append("pending_approvals:")
            for a in pending_apps:
                buf.append(f"- {a['approval_id']}: {clamp(a.get('title') or '', 60)}")
        buf.append("")
        buf.append("Remember: respond with JSON only following the schema in your system prompt.")
        if forced:
            buf.append("(Operator forced this step: do not idle if there is something useful to do.)")
        return "\n".join(buf)

    # ----------------- stationmaster application -----------------

    def apply_stationmaster_output(self, output_text: str) -> None:
        obj = extract_first_json_object(output_text)
        if not obj:
            self.jlog.emit("stationmaster_parse_fail", text_tail=clamp(output_text, 1200))
            self.logger.error("stationmaster output not parseable JSON; ignoring")
            return
        actions = obj.get("actions") or []
        mode = obj.get("mode") or ""
        self.jlog.emit("stationmaster_actions", mode=mode, actions=actions)

        lessons = obj.get("lessons")
        if isinstance(lessons, list) and lessons:
            self._append_lessons([str(x) for x in lessons if str(x).strip()])

        for act in actions:
            if not isinstance(act, dict):
                continue
            typ = act.get("type")
            if typ == "ask_customer":
                text = str(act.get("text") or "").strip()
                if not text:
                    continue
                for cid in self._customer_ids():
                    self.db.enqueue_outbox(cid, text)
                continue
            if typ == "record_seed_item":
                text = str(act.get("text") or "").strip()
                if not text:
                    continue
                tags = act.get("tags") if isinstance(act.get("tags"), list) else []
                self.db.add_seed_draft_item("stationmaster", text, tags=[str(x) for x in tags])
                continue
            if typ == "set_seed_ready":
                ready = bool(act.get("ready"))
                if ready:
                    self._seed_ready_flag = True
                    self._seed_ready_reason = str(act.get("rationale") or "")
                continue
            if typ == "create_nodes":
                epoch_id = self.db.get_active_epoch_id()
                if not epoch_id:
                    # Create epoch if seed exists.
                    seed_id = self.db.get_active_seed_version_id()
                    if not seed_id:
                        continue
                    base_commit = self.git.trunk_head()
                    epoch_id = self.db.create_epoch(seed_id, base_commit=base_commit)
                    self.jlog.emit("epoch_created", epoch_id=epoch_id, seed_version_id=seed_id, base_commit=base_commit)
                nodes = act.get("nodes") or []
                deps = act.get("deps") or []
                id_map: dict[str, str] = {}
                # Create nodes
                for n in nodes:
                    if not isinstance(n, dict):
                        continue
                    node_id = str(n.get("node_id") or "")
                    if not node_id:
                        node_id = f"N-{uuid.uuid4().hex[:8]}"
                    id_map[str(n.get("node_id") or node_id)] = node_id
                    self.db.upsert_node(
                        {
                            "node_id": node_id,
                            "epoch_id": epoch_id,
                            "kind": str(n.get("kind") or "work"),
                            "title": str(n.get("title") or node_id),
                            "objective": str(n.get("objective") or ""),
                            "deliverables": n.get("deliverables") or [],
                            "req_links": n.get("req_links") or [],
                            "scope_hints": n.get("scope_hints") or [],
                            "status": str(n.get("status") or "PENDING").upper(),
                            "priority": int(n.get("priority") or 0),
                        }
                    )
                # Create deps
                for d in deps:
                    if not isinstance(d, dict):
                        continue
                    child = str(d.get("node_id") or "")
                    if not child:
                        continue
                    child = id_map.get(child, child)
                    depends_on = d.get("depends_on") or []
                    if not isinstance(depends_on, list):
                        continue
                    for parent in depends_on:
                        pid = id_map.get(str(parent), str(parent))
                        if pid:
                            self.db.add_dep(child, pid)
                continue
            if typ == "request_approval":
                title = str(act.get("title") or "Approval needed")
                details = str(act.get("details") or "")
                data = act.get("data") if isinstance(act.get("data"), dict) else {}
                appr = self.db.create_approval(title, details, data, requested_by="stationmaster")
                self.jlog.emit("approval_requested", approval_id=appr, title=title)
                continue
            if typ == "set_epoch_status":
                epoch_id = self.db.get_active_epoch_id()
                if not epoch_id:
                    continue
                status = str(act.get("status") or "").upper()
                notes = str(act.get("notes") or "")
                if status:
                    self.db.set_epoch_status(epoch_id, status, notes=notes or None)
                continue

    # ----------------- seed / epoch transitions -----------------

    def maybe_compile_seed(self) -> None:
        force = self.db.meta_get("force_seed_compile", "0") == "1"
        if force:
            self._seed_ready_flag = True
            self._seed_ready_reason = "operator forced"
            self.db.meta_set("force_seed_compile", "0")

        if not self._seed_ready_flag:
            return

        pending = self.seed.pending_draft_items()
        if not pending and self.seed.has_active_seed():
            self._seed_ready_flag = False
            return

        reason = self._seed_ready_reason or "stationmaster ready"
        self._seed_ready_flag = False
        self._seed_ready_reason = ""

        # Compile seed
        seed_version_id = self.seed.compile_seed_from_draft(reason=reason)

        # If there is an active epoch with a different seed, supersede it.
        active_epoch_id = self.db.get_active_epoch_id()
        if active_epoch_id:
            epoch = self.db.get_epoch(active_epoch_id)
            if epoch and epoch["seed_version_id"] != seed_version_id and epoch["status"] not in ("SUPERSEDED", "DONE"):
                self.db.set_epoch_status(active_epoch_id, "SUPERSEDED", notes="seed changed")
                # Mark nodes stale (best-effort)
                nodes = self.db.list_nodes(active_epoch_id, limit=50000)
                for n in nodes:
                    if n["status"] in ("RUNNING", "DONE"):
                        continue
                    self.db.set_node_status(n["node_id"], "STALE")
                self.jlog.emit("epoch_superseded", epoch_id=active_epoch_id, new_seed_version_id=seed_version_id)

        # Create a fresh epoch if none active or superseded/done.
        active_epoch_id = self.db.get_active_epoch_id()
        epoch = self.db.get_epoch(active_epoch_id) if active_epoch_id else None
        if epoch is None or epoch["status"] in ("SUPERSEDED", "DONE", "IDLE"):
            base_commit = self.git.trunk_head()
            new_epoch_id = self.db.create_epoch(seed_version_id, base_commit=base_commit)
            self.db.set_epoch_status(new_epoch_id, "PLANNING", notes="new seed epoch")
            self.jlog.emit("epoch_created", epoch_id=new_epoch_id, seed_version_id=seed_version_id, base_commit=base_commit)

    # ----------------- node readiness + scheduling -----------------

    def update_ready_nodes(self) -> None:
        epoch_id = self.db.get_active_epoch_id()
        if not epoch_id:
            return
        nodes = self.db.list_nodes(epoch_id, limit=50000)
        status_map = {n["node_id"]: n["status"] for n in nodes}
        for n in nodes:
            if n["status"] != "PENDING":
                continue
            deps = self.db.list_deps(n["node_id"])
            if all(status_map.get(d) == "DONE" for d in deps):
                self.db.set_node_status(n["node_id"], "READY")
        # If epoch has no runnable work and no running nodes, stationmaster will be invoked by tick logic.

    def _node_has_join_dependent(self, node_id: str) -> bool:
        deps = self.db.list_dependents(node_id)
        for child_id in deps:
            child = self.db.get_node(child_id)
            if child and child.get("kind") == "join" and child.get("status") not in ("DONE", "STALE"):
                return True
        return False

    def _integrate_if_needed(self, node: dict[str, t.Any]) -> None:
        # If this node will be integrated by a join node, skip.
        if self._node_has_join_dependent(node["node_id"]):
            return
        commit = node.get("result_commit")
        if not commit:
            return
        ok = self.git.integrate_commit_to_trunk(commit, node_id=node["node_id"])
        if not ok:
            appr = self.db.create_approval(
                title=f"Merge conflict integrating {node['node_id']}",
                details="Cherry-pick failed; manual resolution needed or spawn a conflict-resolution node.",
                data={"node_id": node["node_id"], "commit": commit},
                requested_by="integrator",
            )
            self.db.set_node_status(node["node_id"], "BLOCKED", last_error="merge conflict")
            self.jlog.emit("node_blocked_merge_conflict", node_id=node["node_id"], approval_id=appr)

    def launch_work_node(self, node: dict[str, t.Any]) -> None:
        assert self.runtime is not None
        node_id = node["node_id"]
        if node_id in self._running:
            return
        paused = self.db.meta_get("paused", "0") == "1"
        if paused:
            return
        # fan-out checkpoint
        base_commit = self.git.checkpoint_trunk(f"fanout for {node_id}")
        self.db.upsert_node(
            {
                **node,
                "base_commit": base_commit,
                "status": "RUNNING",
                "attempt_count": self.db.bump_node_attempt(node_id),
            }
        )
        # Create lane worktree
        lane = self.git.create_lane(node_id, base_commit)
        conv_id = node.get("conversation_id") or f"node-{node_id}"
        self.db.upsert_node({**node, "workspace_path": lane, "conversation_id": conv_id})

        # Build a minimal context packet for the worker (avoid huge context).
        seed_id = self.db.get_active_seed_version_id()
        ctx_lines = []
        ctx_lines.append("WORK_PACKET_CONTEXT")
        ctx_lines.append(f"active_seed_version_id={seed_id}")
        ctx_lines.append("If you need to reference the SEED, search in SEED.md or shared/seed/.")
        if node.get("req_links"):
            ctx_lines.append(f"req_links={stable_json(node['req_links'])}")
        ctx_packet = "\n".join(ctx_lines)

        fut = self._executor.submit(self._run_lane_worker, node_id, conv_id, node, lane, ctx_packet)
        self._running[node_id] = fut
        self.jlog.emit("node_started", node_id=node_id, lane=lane, base_commit=base_commit)

    def _run_lane_worker(self, node_id: str, conv_id: str, node: dict[str, t.Any], lane: str, ctx_packet: str) -> dict[str, t.Any]:
        """Runs inside thread pool."""
        assert self.runtime is not None
        try:
            output_text = self.runtime.worker_step(
                node_id=node_id,
                node_title=node["title"],
                objective=node["objective"],
                deliverables=node.get("deliverables") or [],
                workspace=lane,
                conversation_id=conv_id,
                context_packet=ctx_packet,
            )
            out = extract_first_json_object(output_text) or {}
            return {"ok": True, "output_text": output_text, "output": out}
        except Exception as e:
            return {"ok": False, "error": str(e), "tb": traceback.format_exc()}

    def finalize_completed_nodes(self) -> None:
        done: list[str] = []
        for node_id, fut in list(self._running.items()):
            if fut.done():
                done.append(node_id)
        for node_id in done:
            fut = self._running.pop(node_id)
            res = fut.result()
            node = self.db.get_node(node_id)
            if not node:
                continue
            try:
                if not res.get("ok"):
                    err = res.get("error") or "worker error"
                    self.db.set_node_status(node_id, "FAILED", last_error=err)
                    self.db.set_node_output(node_id, summary=err, output={"error": err, "tb": res.get("tb")})
                    self.jlog.emit("node_failed", node_id=node_id, err=err)
                    # lane cleanup
                    self.git.remove_lane(node_id)
                    continue

                out = res.get("output") if isinstance(res.get("output"), dict) else {}
                status = str(out.get("status") or "done").lower()
                summary = str(out.get("summary") or "").strip()
                self.db.set_node_output(node_id, summary=summary, output=out)

                # Commit changes in lane
                try:
                    commit = self.git.lane_commit_changes(node_id, node["title"])
                except Exception as e:
                    commit = None
                    self.jlog.emit("lane_commit_error", node_id=node_id, err=str(e))
                if commit:
                    self.db.upsert_node({**node, "result_commit": commit})

                # Map status
                if status == "done":
                    self.db.set_node_status(node_id, "DONE")
                elif status == "needs_input":
                    self.db.set_node_status(node_id, "BLOCKED", last_error="needs input")
                    # Ask customer if worker provided a question in followups/summary.
                    q = summary or "Need additional input to proceed. Please clarify."
                    for cid in self._customer_ids():
                        self.db.enqueue_outbox(cid, q)
                elif status == "blocked":
                    self.db.set_node_status(node_id, "BLOCKED", last_error=summary or "blocked")
                else:
                    self.db.set_node_status(node_id, "FAILED", last_error=summary or "failed")

                # Spawn followups (as new nodes) if provided
                followups = out.get("followups")
                if isinstance(followups, list) and followups:
                    epoch_id = node["epoch_id"]
                    for fu in followups[:12]:
                        if not isinstance(fu, dict):
                            continue
                        nid = f"N-{uuid.uuid4().hex[:8]}"
                        self.db.upsert_node(
                            {
                                "node_id": nid,
                                "epoch_id": epoch_id,
                                "kind": str(fu.get("kind") or "work"),
                                "title": str(fu.get("title") or nid),
                                "objective": str(fu.get("objective") or ""),
                                "deliverables": fu.get("deliverables") or [],
                                "req_links": fu.get("req_links") or [],
                                "scope_hints": fu.get("scope_hints") or [],
                                "status": "PENDING",
                                "priority": int(fu.get("priority") or 0),
                            }
                        )
                        # By default, depend on this node so we keep causality.
                        self.db.add_dep(nid, node_id)
                # Lessons -> append into shared/lessons
                lessons = out.get("lessons")
                if isinstance(lessons, list) and lessons:
                    self._append_lessons([str(x) for x in lessons if str(x).strip()])

                # Cleanup lane
                self.git.remove_lane(node_id)

                # Integrate immediately unless join-dependent
                node2 = self.db.get_node(node_id)
                if node2 and node2.get("status") == "DONE":
                    self._integrate_if_needed(node2)

                self.jlog.emit("node_done", node_id=node_id, status=self.db.get_node(node_id)["status"], commit=commit, summary=summary)
            except Exception as e:
                self.logger.error("finalize node %s error: %s", node_id, e)
                self.jlog.emit("finalize_error", node_id=node_id, err=str(e), tb=traceback.format_exc())
                with contextlib.suppress(Exception):
                    self.git.remove_lane(node_id)

    def _append_lessons(self, lessons: list[str]) -> None:
        if not lessons:
            return
        path = os.path.join(self.cfg.trunk_workdir, "shared", "lessons", "LESSONS.md")
        ensure_dir(os.path.dirname(path))
        ts = utcnow()
        with open(path, "a", encoding="utf-8") as f:
            for ln in lessons:
                f.write(f"- {ts}: {ln}\n")
        # commit best-effort
        with contextlib.suppress(Exception):
            git_commit_all(self.git, self.cfg.trunk_workdir, "oteam: update lessons")

    # ----------------- join / verify nodes (central track) -----------------

    def run_join_nodes_if_ready(self) -> None:
        epoch_id = self.db.get_active_epoch_id()
        if not epoch_id:
            return
        nodes = self.db.list_nodes(epoch_id, status="READY", limit=200)
        join_nodes = [n for n in nodes if n.get("kind") == "join"]
        # Join nodes should run serialized; pick highest priority.
        join_nodes.sort(key=lambda n: (-int(n.get("priority") or 0), n.get("created_at") or ""))
        if not join_nodes:
            return
        node = join_nodes[0]
        node_id = node["node_id"]
        self.db.set_node_status(node_id, "RUNNING")
        self.jlog.emit("join_start", node_id=node_id)

        # Ensure node shared dir exists in trunk.
        ensure_dir(os.path.join(self.cfg.trunk_workdir, "shared", "nodes", node_id))

        # Integrate dependency commits first.
        deps = self.db.list_deps(node_id)
        for dep_id in deps:
            dep = self.db.get_node(dep_id)
            if not dep:
                continue
            commit = dep.get("result_commit")
            if not commit:
                continue
            # If already integrated earlier, cherry-pick will fail with empty change sometimes.
            ok = self.git.integrate_commit_to_trunk(commit, node_id=dep_id)
            if not ok:
                appr = self.db.create_approval(
                    title=f"Merge conflict in join {node_id}",
                    details=f"Conflict integrating dependency {dep_id} ({commit}).",
                    data={"join_node": node_id, "dep_node": dep_id, "commit": commit},
                    requested_by="integrator",
                )
                self.db.set_node_status(node_id, "BLOCKED", last_error="merge conflict")
                self.jlog.emit("join_blocked", node_id=node_id, approval_id=appr)
                return

        # Now run a harmonization worker directly in trunk (central lane).
        if self.runtime is None:
            self.db.set_node_status(node_id, "FAILED", last_error="no runtime")
            return
        conv_id = node.get("conversation_id") or f"join-{node_id}"
        try:
            output_text = self.runtime.worker_step(
                node_id=node_id,
                node_title=node["title"],
                objective=node["objective"],
                deliverables=node.get("deliverables") or [],
                workspace=self.cfg.trunk_workdir,
                conversation_id=conv_id,
                context_packet="You are running on trunk. Harmonize integrated changes and update shared/ as needed.",
            )
            out = extract_first_json_object(output_text) or {}
            summary = str(out.get("summary") or "").strip()
            self.db.set_node_output(node_id, summary=summary, output=out)
        except Exception as e:
            self.db.set_node_status(node_id, "FAILED", last_error=str(e))
            self.jlog.emit("join_failed", node_id=node_id, err=str(e))
            return

        # Commit any trunk changes.
        try:
            commit = None
            if not self.git.trunk_is_clean():
                commit = git_commit_all(self.git, self.cfg.trunk_workdir, f"oteam join {node_id}: {node['title']}")
            self.db.upsert_node({**node, "result_commit": commit} if commit else node)
        except Exception as e:
            self.jlog.emit("join_commit_error", node_id=node_id, err=str(e))

        self.db.set_node_status(node_id, "DONE")
        self.jlog.emit("join_done", node_id=node_id)

    def run_verify_nodes_if_ready(self) -> None:
        epoch_id = self.db.get_active_epoch_id()
        if not epoch_id:
            return
        epoch = self.db.get_epoch(epoch_id)
        if not epoch:
            return
        nodes = self.db.list_nodes(epoch_id, status="READY", limit=200)
        verify_nodes = [n for n in nodes if n.get("kind") == "verify"]
        verify_nodes.sort(key=lambda n: (-int(n.get("priority") or 0), n.get("created_at") or ""))
        if not verify_nodes:
            return
        node = verify_nodes[0]
        node_id = node["node_id"]
        self.db.set_node_status(node_id, "RUNNING")
        self.jlog.emit("verify_start", node_id=node_id)

        if self.runtime is None:
            self.db.set_node_status(node_id, "FAILED", last_error="no runtime")
            return

        seed_id = self.db.get_active_seed_version_id()
        # Provide verifier with small context: links + commands.
        ctx_lines = []
        ctx_lines.append("VERIFY_CONTEXT_PACKET")
        ctx_lines.append(f"seed_version_id={seed_id}")
        ctx_lines.append("Check the repository state and shared/ artifacts. Be skeptical.")
        if node.get("req_links"):
            ctx_lines.append(f"req_links={stable_json(node['req_links'])}")
            # include snippets
            for rid in node["req_links"][:6]:
                r = self.db.get_requirement_chunk(str(rid))
                if r:
                    ctx_lines.append(f"--- requirement {rid} ({r['heading']}) ---")
                    ctx_lines.append(clamp(r["content"], 1200))
        ctx_packet = "\n".join(ctx_lines)

        try:
            output_text = self.runtime.verifier_step(ctx_packet, conversation_id=node.get("conversation_id") or f"verify-{node_id}")
            out = extract_first_json_object(output_text) or {}
            status = str(out.get("status") or "inconclusive").lower()
            summary = str(out.get("summary") or "").strip()
            self.db.set_node_output(node_id, summary=summary, output=out)
            if status == "pass":
                self.db.set_node_status(node_id, "DONE")
            elif status == "fail":
                self.db.set_node_status(node_id, "FAILED", last_error=summary or "verification failed")
            else:
                self.db.set_node_status(node_id, "BLOCKED", last_error=summary or "inconclusive")

            # Spawn gaps as nodes
            gaps = out.get("gaps")
            if isinstance(gaps, list) and gaps:
                for g in gaps[:10]:
                    if not isinstance(g, dict):
                        continue
                    sn = g.get("suggested_node")
                    if isinstance(sn, dict):
                        nid = f"N-{uuid.uuid4().hex[:8]}"
                        self.db.upsert_node(
                            {
                                "node_id": nid,
                                "epoch_id": epoch_id,
                                "kind": str(sn.get("kind") or "work"),
                                "title": str(sn.get("title") or nid),
                                "objective": str(sn.get("objective") or ""),
                                "deliverables": sn.get("deliverables") or [],
                                "req_links": sn.get("req_links") or node.get("req_links") or [],
                                "scope_hints": sn.get("scope_hints") or [],
                                "status": "PENDING",
                                "priority": int(sn.get("priority") or 0),
                            }
                        )
                        # depend on verify node so it happens after fix attempt
                        self.db.add_dep(nid, node_id)

        except Exception as e:
            self.db.set_node_status(node_id, "FAILED", last_error=str(e))
            self.jlog.emit("verify_failed", node_id=node_id, err=str(e))
            return

        self.jlog.emit("verify_done", node_id=node_id)

    # ----------------- main loop -----------------

    def tick(self) -> None:
        # Ensure trunk exists
        self.git.ensure_trunk_worktree()

        # Seed compile if requested
        self.maybe_compile_seed()

        # If no seed yet, invoke stationmaster in SEEDING mode frequently.
        seed_id = self.db.get_active_seed_version_id()
        if not seed_id:
            self._stationmaster_tick(force=True)
            # If stationmaster requested seed compile, try it now.
            self.maybe_compile_seed()
            return

        # Ensure epoch exists
        epoch_id = self.db.get_active_epoch_id()
        epoch = self.db.get_epoch(epoch_id) if epoch_id else None
        if epoch is None or epoch["status"] in ("SUPERSEDED", "DONE", "IDLE"):
            base_commit = self.git.trunk_head()
            epoch_id = self.db.create_epoch(seed_id, base_commit=base_commit)
            self.db.set_epoch_status(epoch_id, "PLANNING", notes="auto new epoch")
            epoch = self.db.get_epoch(epoch_id)

        # If seed changed mid-flight, request migration planning.
        pending_seed = len(self.seed.pending_draft_items())
        if pending_seed > 0 and epoch and epoch["status"] not in ("DONE", "IDLE", "SUPERSEDED"):
            self._stationmaster_tick(force=True, mode_hint="MIGRATING")
            self.maybe_compile_seed()
            return

        # Stationmaster planning/expansion tick when needed.
        self._stationmaster_tick_if_needed(epoch)

        # Update readiness
        self.update_ready_nodes()

        # Finalize worker threads
        self.finalize_completed_nodes()

        # Run central join/verify nodes first (serialized).
        self.run_join_nodes_if_ready()
        self.run_verify_nodes_if_ready()

        # Launch work nodes (parallel lanes).
        self._launch_ready_work_nodes()

        # If nothing is happening, nudge stationmaster.
        self._maybe_nudge_stationmaster(epoch)

    def _stationmaster_tick(self, force: bool = False, mode_hint: str | None = None) -> None:
        if self.runtime is None:
            return
        now = now_ts()
        if not force and (now - self._last_stationmaster_ts) < self.cfg.conductor_tick_s:
            return
        self._last_stationmaster_ts = now
        ctx_packet = self.build_stationmaster_context(mode_hint=mode_hint)
        try:
            out_text = self.runtime.stationmaster_step(ctx_packet)
        except Exception as e:
            self.jlog.emit("stationmaster_error", err=str(e), tb=traceback.format_exc())
            return
        self.apply_stationmaster_output(out_text)

    def _stationmaster_tick_if_needed(self, epoch: sqlite3.Row | None) -> None:
        force = self.db.meta_get("force_stationmaster", "0") == "1"
        if force:
            self.db.meta_set("force_stationmaster", "0")
            self._stationmaster_tick(force=True)
            self.maybe_compile_seed()
            return
        if epoch is None:
            self._stationmaster_tick(force=True)
            return
        # If planning and no nodes exist -> plan
        epoch_id = epoch["epoch_id"]
        nodes = self.db.list_nodes(epoch_id, limit=5)
        if epoch["status"] == "PLANNING" and not nodes:
            self._stationmaster_tick(force=True, mode_hint="PLANNING")
            return
        # If no READY nodes and nothing running -> expand
        if not self._running:
            ready = self.db.list_nodes(epoch_id, status="READY", limit=1)
            pending = self.db.list_nodes(epoch_id, status="PENDING", limit=1)
            if not ready and not pending:
                self._stationmaster_tick(force=True, mode_hint="EXECUTING")
                return
            if not ready and pending:
                # Maybe dependencies not satisfied; still can plan next join/verify or split.
                self._stationmaster_tick(force=False, mode_hint="EXECUTING")
                return

    def _launch_ready_work_nodes(self) -> None:
        epoch_id = self.db.get_active_epoch_id()
        if not epoch_id:
            return
        if self.db.meta_get("paused", "0") == "1":
            return
        # available slots
        slots = max(0, min(self.cfg.max_parallel_lanes, self.cfg.node_workers) - len(self._running))
        if slots <= 0:
            return
        ready = self.db.list_nodes(epoch_id, status="READY", limit=200)
        # Only lane-executed kinds
        lane_kinds = {"work", "reflect", "migrate"}
        ready = [n for n in ready if n.get("kind") in lane_kinds]
        # sort by priority then age
        ready.sort(key=lambda n: (-int(n.get("priority") or 0), n.get("created_at") or ""))
        for n in ready[:slots]:
            self.launch_work_node(n)

    def _maybe_nudge_stationmaster(self, epoch: sqlite3.Row | None) -> None:
        if epoch is None:
            return
        epoch_id = epoch["epoch_id"]
        if self._running:
            return
        ready = self.db.list_nodes(epoch_id, status="READY", limit=1)
        pending = self.db.list_nodes(epoch_id, status="PENDING", limit=1)
        if not ready and not pending:
            self._stationmaster_tick(force=True, mode_hint="EXECUTING")

    def loop_forever(self) -> None:  # pragma: no cover
        self.logger.info("oteam orchestrator loop starting")
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:
                self.logger.error("orchestrator tick error: %s", e)
                self.jlog.emit("orchestrator_error", err=str(e), tb=traceback.format_exc())
            time.sleep(self.cfg.scheduler_tick_s)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OTeam trainyard daemon (OpenHands SDK)")
    ap.add_argument("--project", required=True, help="Path to the project root (git repo)")
    ap.add_argument("--state-dir-name", default=".oteam", help="State directory name (default: .oteam)")
    ap.add_argument("--http-host", default=None)
    ap.add_argument("--http-port", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true", help="Run without OpenHands (no planning/execution)")
    args = ap.parse_args(argv)

    cfg = Config.from_project(args.project, state_dir_name=args.state_dir_name)
    if args.http_host:
        cfg.http_host = args.http_host
    if args.http_port:
        cfg.http_port = args.http_port

    cfg.ensure_layout()
    logger, jlog = setup_logging(cfg)
    logger.info("oteam starting; project=%s state=%s", cfg.project_root, cfg.state_dir)

    db = StateDB(cfg.db_path)

    git = GitWorkdirManager(cfg, logger, jlog)
    git.ensure_trunk_worktree()

    # prompts
    prompt_paths = write_prompt_templates(cfg)

    runtime: OpenHandsRuntime | None = None
    if not args.no_llm:
        runtime = OpenHandsRuntime(cfg, logger, jlog, prompt_paths)

    seed = SeedManager(cfg, db, git, logger, jlog)

    # If SEED.md exists but DB has no active seed, import it.
    if seed.load_seed_text() and not db.get_active_seed_version_id():
        seed_text = seed.load_seed_text() or ""
        commit = git.trunk_head()
        seed_version_id = db.insert_seed_version(seed_text, commit_hash=commit, seed_path="SEED.md", summary="imported existing SEED.md")
        db.set_active_seed_version(seed_version_id)
        db.replace_requirement_chunks(seed_version_id, chunk_markdown_by_headings(seed_text, seed_version_id))
        jlog.emit("seed_imported", seed_version_id=seed_version_id, commit=commit)

    ctx = ServiceContext(cfg=cfg, db=db, git=git, seed=seed, runtime=runtime, logger=logger, jlog=jlog)
    global _SERVICE_CTX
    _SERVICE_CTX = ctx

    # Start customer HTTP server
    httpd = ThreadingHTTPServer((cfg.http_host, cfg.http_port), CustomerHTTPHandler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    logger.info("customer HTTP listening on http://%s:%s", cfg.http_host, cfg.http_port)

    # Start repo indexer
    indexer = RepoIndexer(cfg, db, logger, jlog)
    indexer.start()

    # Telegram
    tbot = TelegramBot(ctx)
    tbot.start()

    orch = Orchestrator(ctx)

    # Signals
    def _shutdown(signum: int, frame: t.Any) -> None:
        logger.info("shutdown signal %s", signum)
        orch.stop()
        indexer.stop()
        tbot.stop()
        with contextlib.suppress(Exception):
            httpd.shutdown()
        with contextlib.suppress(Exception):
            httpd.server_close()
        with contextlib.suppress(Exception):
            db.close()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        orch.loop_forever()
    finally:
        _shutdown(0, None)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

