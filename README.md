# Clanker Team (cteam)

Clanker Team (`cteam`) is a single-file orchestration tool that spins up a tmux-based team of Codex agents working in a shared git repo. It handles agent workdirs, messaging, nudging, and (optionally) a customer chat bridge so you can run coordinated multi-agent sessions quickly.

## Features
- PM-led workflow with explicit assignments (`Type: ASSIGNMENT`) to keep agents in sync.
- tmux windows for each agent, a router (`cteam watch`) that nudges on new mail, and a customer chat window.
- Workspace scaffold: bare repo (`project.git`), integration checkout (`project/`), per-agent clones (`agents/<name>/proj/`), and shared coordination docs (`shared/`).
- Telegram bridge option for the customer channel.
- Sensible Codex defaults: `--sandbox danger-full-access` and `--ask-for-approval never` (or `--approval-policy never`) for minimal interruptions.

## Requirements
- Python 3.9+
- `git` and `tmux` available in `PATH`
- Codex CLI installed (`codex` by default). For the intended experience, run with `--sandbox danger-full-access` and `--ask-for-approval never` (or `--approval-policy never` if your CLI uses that flag).

## Quick start (new project)
```bash
# Initialize a fresh Clanker Team workspace with two devs
python3 cteam.py init /path/to/workspace --name "My Project" --devs 2

# Attach to tmux (if not auto-attached)
python3 cteam.py attach /path/to/workspace
```

What happens:
- Bare repo `project.git` and integration checkout `project/` are created.
- Agent workdirs (`agents/pm`, `agents/architect`, `agents/dev1`, …) are cloned from the bare repo.
- Shared coordination docs live in `shared/` (GOALS, PLAN, TASKS, DECISIONS, TIMELINE, PROTOCOL, TEAM_ROSTER).
- A router window (`router`) tails mailboxes and nudges agents; agent windows start Codex using the defaults above.

## Import an existing repo
```bash
python3 cteam.py import /path/to/workspace --src /path/to/existing/repo --devs 2 --recon
```
- `--recon` seeds safe, read-only reconnaissance assignments to architect/tester/researcher.
- The existing repo is mirrored into `project.git` and `project/`, then cloned into each agent workdir.

## Everyday commands
- Resume/open workspace + tmux: `python3 cteam.py open .`
- Pause tmux (park windows): `python3 cteam.py pause .`
- Send a message: `python3 cteam.py msg . --to dev1 --subject "Note" "Text..."`
- Send an assignment (starts Codex if needed): `python3 cteam.py assign . --to dev1 --task T001 --subject "Build feature" "details..."`
- Broadcast: `python3 cteam.py broadcast . "Short update"`
- Manual nudge: `python3 cteam.py nudge . --to dev1 --reason "Check inbox"`
- Customer chat window (for PM channel): `python3 cteam.py customer-chat .`
- Add an agent (developer/tester/architect/researcher; PM is unique): `python3 cteam.py add-agent . --role developer --name dev3` (PM is notified to rebalance work)

## Customer channel & Telegram
- PM-only channel lives at `shared/mail/customer/message.md`.
- Local chat: run `python3 cteam.py customer-chat .` in its own tmux window (the `customer` window is auto-created).
- Telegram bridge (optional):
  1) Configure: `python3 cteam.py telegram-configure . --token <bot_token> --phone <authorized_phone>`
  2) Enable: `python3 cteam.py telegram-enable .`
  3) Run/open the workspace so the router (`cteam watch`) can bridge messages.

## Workspace layout
- `project.git` — bare origin
- `project/` — integration checkout (for reviews/merges)
- `agents/<name>/proj/` — per-agent clone (work here)
- `shared/` — coordination docs, message logs, and shared drive (`shared/drive/` for large/binary artifacts; keep code in git)
- `seed/`, `seed-extras/` — non-repo inputs the PM/architect/researcher can read
- `cteam.json` — persisted workspace state (versioned)

## Codex posture
- Defaults favor low-friction collaboration:
  - `--sandbox danger-full-access`
  - `--ask-for-approval never` (or `--approval-policy never` on newer CLIs)
- Override via CLI flags (`--sandbox`, `--ask-for-approval`, `--model`, `--no-search`, `--full-auto`, `--yolo`) when you start a workspace.

## Testing
Run the automated test suite from the repo root:
```bash
python -m unittest
```

## Tips
- Branch naming: `agent/<agent>/<topic>`; push/pull often.
- Non-PM agents wait for `Type: ASSIGNMENT` before coding; recon notes are fine.
- Keep customer comms PM-only and logged via the customer channel.
- When new agents join, the PM should onboard them and rebalance tasks to keep workloads healthy.
