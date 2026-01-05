# OTeam (OpenHands) Trainyard

This repo contains a **single-file daemon**: `oteam_openhands_trainyard.py`.

The daemon is designed to run *one long project* (days/weeks) using **one LLM model** through the
OpenHands Software Agent SDK.

It is **not software-development specific**: it uses Git as a universal, content-addressed
workspace ledger for *any* long-context project (software, documentation, research, ops playbooks, etc).

---

## Mental model

Think of the project as a **train yard**:

- The **trunk** is the main track: a git worktree on branch `oteam/trunk`.
- The Stationmaster can **fan out** into multiple parallel **lanes** (independent work packets).
- Lanes later **fan back in** via **join nodes** that integrate lane commits into trunk and
  produce consolidated evidence.

Parallelism is about **independent deliverables**, not a serialized “implement/test/review” pipeline.

---

## Key artifacts

### SEED (specification backbone)

- `SEED.md` in trunk is the current compiled project SEED.
- The SEED is compiled deterministically from an **atomic ledger**:
  - stored in SQLite table `seed_draft_items`
  - each item is a small requirement/constraint/definition/preference from customer/operator/Stationmaster.
- Seed versions are tracked in `seed_versions`, and chunked into `requirement_chunks` + `requirement_fts`
  for search/retrieval.

**Seed changes are first-class events.**
When new ledger items materially change the SEED, the orchestrator will:
1. compile a new SEED version,
2. mark the current epoch `SUPERSEDED`,
3. start a new epoch and re-plan (migration).

### DAG nodes (work packets)

Nodes are stored in SQLite `nodes` with dependencies in `node_deps`.

Node kinds:
- `work`    : do a concrete deliverable
- `join`    : merge lane outputs back into trunk and harmonize
- `verify`  : check evidence against the SEED
- `reflect` : analysis / replanning / unblocking
- `migrate` : handle seed-version transitions

Each node writes merge-friendly outputs in:
- `shared/nodes/<node_id>/`

---

## Git workflow (worktrees)

- Trunk worktree: `.oteam/workdir/trunk` on branch `oteam/trunk`
- Lane worktrees: `.oteam/workdir/lanes/<node_id>` in detached HEAD

Before a fan-out, trunk is **checkpoint-committed** so every lane starts from a clean base commit.

When a lane completes:
1. it is committed (detached commit hash stored in node.result_commit),
2. it is integrated into trunk **only if**:
   - it is *not* intended to be merged by a downstream join node.

Join nodes integrate their dependency commits into trunk (cherry-pick), then run a harmonization worker
directly in trunk to consolidate and produce evidence.

---

## Interfaces

### Customer channel (HTTP)

- `POST /customer/inbound`  (JSON or form)
  - `{ "customer_id": "default", "text": "..." }`
- `GET /customer/outbox?customer_id=default&ack=1`
- `GET /customer/status`

This is intentionally **run-id free**. The customer never sees internal DAG ids unless you choose to.

### Operator channel (Telegram)

Telegram commands:
- `/status`
- `/seed`
- `/seed_add <text>`
- `/seed_compile`
- `/graph`
- `/nodes [STATUS]`
- `/node <id>`
- `/approvals`
- `/approve <id> [reason]`
- `/reject <id> [reason]`
- `/pause` / `/resume`
- `/kick` (force Stationmaster tick)

---

## Prompts

Prompt templates are written at runtime into:
- `.oteam/prompts/stationmaster.j2` (Stationmaster **Actor**)
- `.oteam/prompts/stationmaster_critic.j2` (Stationmaster **Critic**)
- `.oteam/prompts/worker.j2`
- `.oteam/prompts/verifier.j2`

### Stationmaster actor-critic

The Stationmaster runs as an **actor-critic controller**:

- **Actor** (`stationmaster.j2`) proposes a small set of actions (seed questions/items, DAG node creation, epoch status changes).
- **Critic** (`stationmaster_critic.j2`) audits the proposal for:
  - schema correctness (parseable JSON),
  - SEED alignment and correct mode (especially during MIGRATING),
  - DAG health (branch width, fan-in via joins, verification coverage),
  - avoidance of irreversible one-way choices.

The critic returns the final Stationmaster JSON (it may copy the actor output unchanged, revise it, or reject and emit a safe fallback).
The orchestrator also accepts optional `lessons` from the critic output and appends them to `shared/lessons/LESSONS.md`.

Runtime controls (env vars):
- `OTEAM_STATIONMASTER_ACTOR_CRITIC=0` disables the critic and runs the actor only.
- `OTEAM_STATIONMASTER_AC_MAX_RETRIES=0` disables the revision loop (still one critic pass if enabled).
  Default is `1` retry if the critic rejects.

They enforce:
- **generic vocabulary** (deliverables / evidence / work packets)
- **JSON-only outputs** (for robust parsing + automation)
- **DAG-first planning** with explicit fan-out/fan-in

---

## Extending the system

### Add new action types
- Update `STATIONMASTER_PROMPT_J2` schema
- Implement action handling in `Orchestrator.apply_stationmaster_output()`

### Add new node kinds
- Teach the Stationmaster prompt
- Add execution logic (either as a lane node or a trunk node)

### Add safety / approvals
- Use `StateDB.create_approval()` to force human approval for risky steps
- Wire approval checks into scheduling/integration as needed

### Add custom tools to the agents
The current design keeps tools minimal (Terminal + FileEditor + optional TaskTracker).
If you want state/query tools (seed search, DAG query, etc.), prefer:
- **sqlite-backed tools** (fast, grounded, avoids context bloat)
- or **read-only generated state files** under `shared/`

---

## Running

Required:
- `LLM_API_KEY` (and optionally `LLM_MODEL`, `LLM_BASE_URL`)

Optional:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ADMIN_CHAT_ID`

Start:
```bash
python oteam_openhands_trainyard.py --project /path/to/repo
```

Logs:
```bash
tail -f .oteam/logs/oteam.log
tail -f .oteam/logs/events.jsonl
```

---

## Operational notes / guarantees

- The orchestrator never parallelizes “roles” within a single work packet.
  Parallelism is at the **node (work packet)** level, controlled by the Stationmaster DAG decisions.
- Long-context safety comes from:
  - seed ledger + deterministic SEED compilation
  - git commits as an immutable audit trail
  - join nodes that reconsolidate and summarize before the graph grows too wide
  - verifier nodes that produce explicit evidence and spawn new work when gaps are found
