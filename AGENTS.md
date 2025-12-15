# AGENTS — cteam

## What this is
- `cteam.py` is a single-file orchestration tool that spins up a tmux-based team of Codex agents working against a shared git repo.
- It bootstraps a bare repo (`project.git`), an integration checkout (`project/`), per-agent clones (`agents/<name>/proj/`), and out-of-repo coordination assets under `shared/`.
- Messaging is file-based (`shared/mail/<agent>/message.md` + inbox/outbox copies); a router loop (`cteam watch`) nudges agents and can auto-start Codex when assignments arrive. A dedicated `customer` window runs `cteam customer-chat` for PM/customer conversations. When sending messages via cteam CLI, tmux focus jumps to the recipient window by default (use `--no-follow` to stay put); customer chat also focuses PM when the customer sends. Use `cteam chat . --to <agent|customer>` for an interactive chat (with readline) from any terminal.
- Shared drive: `shared/drive/` (linked into each agent as `shared-drive/`) for large/non-repo artifacts; **all code/config stays in git**. Keep branches tidy (`agent/<name>/<topic>`), push/pull frequently, avoid history rewrites, and reference shared-drive artifacts in notes/PRs instead of committing them.
- Roles clarity: PM coordinates; Architect designs; Developers code; Tester/QA tests (customer should not be asked to test); Researcher researches. Customer only provides inputs/feedback through PM—never ask them to do engineering tasks.

## Workspace model
- State lives in `cteam.json` (versioned; upgraded in `upgrade_state_if_needed`). Paths are stored absolute + relative; keep compatibility when editing.
- Key dirs created by init/import: `shared/` (GOALS/PLAN/TASKS/DECISIONS/TIMELINE/PROTOCOL/TEAM_ROSTER), `seed/`, `seed-extras/`, `logs/`, `agents/`, `shared/runtime/`.
- Each agent dir links to the canonical mailbox and shared docs; AGENTS/STATUS templates are generated per role via `render_agent_agents_md` and `render_agent_status_template`.
- Defaults are PM-led: **all agent windows autostart Codex**, but non-PM agents must still wait for explicit `ASSIGNMENT` before coding; router nudges on new mail.

## CLI surfaces (python3 cteam.py …)
- Workspace lifecycle: `init`, `import --src <git|dir>`, `resume`, `open`, `attach`, `kill`.
- Coordination: `msg`, `broadcast`, `assign` (Type=`ASSIGNMENT`), `nudge`, `watch` (router loop).
- Operator convenience: `msg/assign/nudge` default to following (selecting) the recipient window; add `--no-follow` to stay put. Workspace starters (`init/import/resume/open`) attach to tmux by default; add `--no-attach` to just launch without attaching.
- Repo/utilities: `sync` (shows statuses/pulls), `seed-sync` (propagate seed/seed-extras), `restart` (respawn Codex), `add-agent`, `doc-walk` (kick off doc sprint).
- Flags of note: `--codex-cmd/--model/--sandbox/--ask-for-approval/--full-auto/--yolo/--no-search`, `--autostart`, `--no-router`, `--no-codex`, `--no-tmux`.
- Default Codex posture is permissive (`--sandbox danger-full-access`, approvals `never`) but capabilities are auto-detected; adjust defaults carefully.

## Coordination defaults to preserve
- PM is the coordinator; non-PM agents must wait for explicit `Type: ASSIGNMENT` before coding.
- Router tmux window name: `router`; session name: `cteam_<slugified project>`.
- Mail append-only logs: `shared/MESSAGES.log.md`, `shared/ASSIGNMENTS.log.md`.
- Branch naming guidance emitted to agents: `agent/<agent>/<topic>`.

## Developing cteam.py
- Keep it single-file, stdlib-only. Major sections: constants/layout, git helpers, tmux helpers, Codex capability detection, state build/upgrade, templates, workspace creation, messaging/router, commands, CLI parser.
- If changing state shape, bump `STATE_VERSION`, add upgrade logic, and ensure `compute_agent_abspaths` is called.
- Mind symlink fallbacks: `safe_link_file/dir` replace with copies/pointer files if the FS disallows symlinks.
- When touching tmux/Codex startup, keep PM-led flow intact (`autostart_agent_names`, `start_codex_in_window`, `maybe_start_agent_on_message`).
- External deps: Python 3, git, tmux, Codex CLI; everything else is stdlib.

## Manual test recipe (no automated tests yet)
- Dry run help: `python3 cteam.py --help`.
- Init sandbox: `python3 cteam.py init /tmp/cteam-sandbox --devs 2 --no-codex --attach` (ensures tmux/windows/mailboxes scaffold correctly).
- Import flow: `python3 cteam.py import /tmp/cteam-import --src <repo> --no-codex --recon` (verifies copy+commit to bare repo and recon assignments).
- Router: inside a workspace, run `python3 cteam.py watch .` and append to `shared/mail/dev1/message.md` to confirm nudges + auto-start behavior.

## Updating this doc
- Keep this file in sync with defaults in `cteam.py` (mail locations, autostart rules, router window name, state version).
- Note any safety-sensitive defaults changes (sandbox/approval) here so downstream agents know the posture.
