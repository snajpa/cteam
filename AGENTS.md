# Coding agent instructions (repo-wide)

This repository is a **scaffold** intended for software agents (and humans) to implement the system specified in `doc/`.

## Non-negotiable invariants

1) **Server never touches workspace files directly**
- Workspace files + box sandboxes live behind `ap-wsmgr` (`no “local workspace mode”`).
- Source: `doc/README.md` L14-L17.

2) **Hard control-plane / data-plane split**
- control-plane: `ap-server` (users, agents, runs, LLM, streaming, event log)
- data-plane: `ap-wsmgr` (workspace filesystem, boxes, snapshots)
- Source: `doc/02-architecture.md` L1-L44.

3) **Event log is the source of truth for UI + replay**
- UI reconstructs from snapshot + subscribe.
- Replay/cold-resume depends on deterministic boundaries recorded.
- Source: `doc/05-event-log.md` L1-L52, `doc/21-replay-and-resume.md` L1-L106.

4) **mruby worker is out-of-process and has no network**
- The worker communicates with server only via RPC.
- Source: `doc/07-mruby-runtime.md` L1-L16, `doc/17-rpc-protocol.md` L20-L33.

If an implementation choice conflicts with these, the implementation choice is wrong.

## How to use the spec

- `doc/` is the primary design spec. Do not “invent” behavior that contradicts it.
- Every TODO in code must point to specific `doc/*` line ranges.
- If you change `doc/*`, you MUST update any TODO line references that mention the changed lines.

Tip: use `nl -ba doc/<file>.md | sed -n 'X,Yp'` to view the exact line ranges referenced by TODOs.

## Milestone-driven implementation order

Follow `doc/15-implementation-plan.md`:

- Milestone 0: repo skeleton (already scaffolded) – `doc/15-implementation-plan.md` L5-L13.
- Milestone 1: `ap-wsmgr` MVP filesystem + leases – `doc/15-implementation-plan.md` L15-L26.
- Milestone 2: `ap-server` core DB + event log + SSE – `doc/15-implementation-plan.md` L27-L44.
- Milestone 3: contexts + streaming messages – `doc/15-implementation-plan.md` L45-L57.
- Milestone 4: runs + mruby worker (no LLM) – `doc/15-implementation-plan.md` L58-L70.
- Milestone 5: LLM providers + tool loop – `doc/15-implementation-plan.md` L71-L85 and `doc/18-llm-providers.md`.
- Milestone 6: bidirectional widgets – `doc/15-implementation-plan.md` L86-L99 and `doc/08-ui-widgets.md`.
- Milestone 7: git agent registry + workspace creation from agent ref – `doc/15-implementation-plan.md` L100-L112 and `doc/10-agent-registry.md` L86-L103.
- Milestone 8: boxes via bubblewrap + streaming NDJSON – `doc/15-implementation-plan.md` L113-L124 and `doc/22-workspace-manager.md` L197-L217.
- Milestone 9: snapshots – `doc/15-implementation-plan.md` L125-L136 and `doc/22-workspace-manager.md` L114-L141.
- Milestone 10: CLI TUI attach – `doc/15-implementation-plan.md` L137-L143 and `doc/12-cli.md` L68-L114.

## Testing expectations

The goal is "boring reliability".

Minimum test coverage targets (MVP):
- Path traversal + path normalization tests (server + wsmgr).
  - Sources: `doc/14-security.md` L20-L31 and `doc/19-policy-and-tools.md` L118-L129.
- Event sequence allocation correctness under concurrency.
  - Source: `doc/04-data-model.md` L208-L215.
- Integration test: wsmgr workspace create/open/fs operations.
  - Source: `doc/22-workspace-manager.md` L153-L196.
- Integration test: server append events + SSE subscribe from `after=SEQ`.
  - Source: `doc/05-event-log.md` L30-L52.

See also testing strategy list: `doc/15-implementation-plan.md` L157-L171.

## Repo conventions

- Prefer small crates with explicit boundaries.
- Keep protocol/data types in shared crates (`ap-core`, `ap-events`, `ap-rpc`) to avoid drift.
- Do not add new networked daemons beyond `ap-server` and `ap-wsmgr`.
- Keep everything async-first (`tokio`).
- Log with `tracing`, not `println!`.

## Tooling note (oteam)

- oteam expects the `opencode` CLI binary name. If only `codex` is installed, create an `opencode` symlink to `codex` instead of changing `opencode_cmd`; pane health checks rely on the configured name.

## Security posture (MVP)

- This platform is not aiming for perfect hostile-code isolation.
- It must still prevent obvious escapes: path traversal, cross-user leakage, accidental network access from mruby.
- Sources: `doc/14-security.md` L1-L116 and `doc/11-boxes.md` L15-L30.

## “Doc-first TODOs” rule

Every component has a top-level TODO plan (usually in `AGENTS.md` and/or `src/lib.rs`) with:
- the intended behavior,
- implementation steps,
- and precise `doc/*` backreferences.

When you implement a TODO:
1) Link the PR/commit to the TODO block you’re completing.
2) Replace TODO text with actual code.
3) Add/extend tests.
4) Keep the backreference notes as comments (or move them into docstrings) so future changes can be traced to the spec.
