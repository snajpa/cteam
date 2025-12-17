# Clanker Team (cteam)

Clanker Team is a single-file orchestrator that spins up tmux-based Codex
agents around a shared git repo. It keeps work PM-led, nudges agents when mail
arrives, and can bridge a customer chat (local or Telegram).

## What it does
- Starts a tmux session with windows for PM/architect/devs/tester/researcher.
- Creates a bare repo (`project.git`), an integration checkout (`project/`), and
  per-agent clones (`agents/<name>/proj/`).
- Maintains shared coordination docs (GOALS, PLAN, TASKS, DECISIONS, TIMELINE).
- Routes mail via `shared/mail/<agent>/message.md` and nudges/auto-starts Codex
  on assignments.
- Optional Telegram bridge for the customer channel.

## Requirements
- Python 3.9+
- `git` and `tmux` in `PATH`
- Codex CLI (`codex` in `PATH`)
- Recommended Codex posture for low-friction work:
  - `--sandbox danger-full-access`
  - `--ask-for-approval never` (or `--approval-policy never` on newer CLIs)

## Quick start (new project)
1) Initialize a workspace (example with two devs):
   ```bash
   python3 cteam.py init /path/to/workspace --name "My Project" --devs 2
   ```
2) Attach to tmux if not auto-attached:
   ```bash
   python3 cteam.py attach /path/to/workspace
   ```
3) PM fills `shared/GOALS.md`, `shared/PLAN.md`, `shared/TASKS.md`, then sends
   assignments (`cteam assign ...`).

## Import an existing repo
```bash
python3 cteam.py import /path/to/workspace \
  --src /path/to/existing/repo \
  --devs 2 \
  --recon
```
`--recon` seeds safe recon assignments (no code changes) to architect/tester/
researcher. The source repo becomes `project.git` and `project/`; each agent
gets a clone under `agents/<name>/proj/`.

## Everyday loop (PM-led)
- Open/resume: `python3 cteam.py open .`
- Customer channel: `python3 cteam.py customer-chat .`
- Assign work (auto-starts Codex):  
  `python3 cteam.py assign . --to dev1 --task T001 --subject "Feature" "details..."`
- Message/broadcast: `python3 cteam.py msg . --to dev1 "Ping"` /
  `python3 cteam.py broadcast . "Standup in 10"`
- Nudge: `python3 cteam.py nudge . --to dev1 --reason "Check inbox"`; add
  `--interrupt` to send an immediate interrupt (sends Esc before the message).
- Add agent (PM is unique; PM gets balancing reminder):  
  `python3 cteam.py add-agent . --role developer --name dev3`
- Router loop (usually in tmux window `router`): `python3 cteam.py watch .`

## Codex posture
- Defaults: `--sandbox danger-full-access`, `--ask-for-approval never`,
  `--search` on (if supported). Override with `--sandbox`, `--ask-for-approval`
  (or `--approval-policy`), `--model`, `--no-search`, `--full-auto`, `--yolo`.
- To mirror Clanker Team’s intended behavior, run Codex with approval policy set
  to `never`.

## Roles and etiquette
- PM is the coordinator. Non-PM agents wait for `Type: ASSIGNMENT` before
  coding; recon notes without code are fine.
- Customer channel is PM-only. Customer messages go to PM; PM replies via the
  customer channel (local chat or Telegram bridge).
- Branch naming: `agent/<agent>/<topic>`; push/pull often. Large/binary assets
  belong in `shared/drive/`, not git.
- Long/ongoing commands should be logged in each agent’s `STATUS.md`; PM tracks
  and nudges owners.

## Workspace layout
- `project.git` — bare origin
- `project/` — integration checkout
- `agents/<name>/proj/` — per-agent clones (work here)
- `shared/` — coordination docs, mail logs, and `drive/` for non-repo artifacts
- `seed/`, `seed-extras/` — input docs (not in git)
- `cteam.json` — workspace state (versioned)

## Testing
```bash
python3 -m unittest discover
```

## Flag cheatsheet (commonly used)
- Codex: `--codex-cmd`, `--model`, `--sandbox`, `--ask-for-approval`,
  `--no-search`, `--full-auto`, `--yolo`
- tmux/router: `--no-tmux`, `--no-codex`, `--no-attach`, `--window`,
  `--autostart pm|pm+architect|all`, `--no-router` / `--router`
- Help: usage errors print full `--help`; `cteam <command> --help` shows
  defaults.

## Command reference (with explanations)
Lifecycle
- `init WORKDIR [--name NAME] [--devs N] [--force] [--no-interactive]
  [--autostart pm|pm+architect|all]`  
  Create a new workspace with a fresh git repo and agent clones.
- `import WORKDIR --src <repo|dir> [--name NAME] [--devs N] [--force] [--recon]
  [--autostart ...]`  
  Import an existing project. `--recon` sends safe, read-only recon tasks to
  non-PM agents.
- `open WORKDIR [--no-router]` / `resume WORKDIR [--autostart ...]
  [--router|--no-router]`  
  Ensure structure exists, start tmux/router/Codex unless suppressed.
- `attach WORKDIR [--window NAME]`  
  Attach to the tmux session; `kill WORKDIR` kills the tmux session.

Coordination & chat
- `watch WORKDIR [--interval SECONDS]`  
  Router loop: watches mailboxes, nudges agents, auto-starts Codex on
  assignments, and nags PM if the customer is waiting.
- `msg WORKDIR --to NAME [--from SENDER] [--subject TEXT] [--file PATH]
  [--no-nudge] [--start-if-needed] [--no-follow] TEXT`  
  Send a message to one agent.
- `broadcast WORKDIR [--from SENDER] [--subject TEXT] [--file PATH]
  [--no-nudge] [--start-if-needed] TEXT`  
  Send a message to all agents.
- `assign WORKDIR --to NAME [--task ID] [--from SENDER] [--subject TEXT]
  [--file PATH] [--no-nudge] [--no-follow] TEXT`  
  Send an assignment (`Type: ASSIGNMENT`); auto-starts recipient Codex.
- `nudge WORKDIR [--to NAME|all] [--reason TEXT] [--no-follow]`  
  Manual nudge to a tmux window. Add `--interrupt` to send Escape before the
  message (use for urgent interrupts).
- `chat WORKDIR --to NAME [--sender NAME] [--subject TEXT] [--no-nudge]
  [--start-if-needed]`  
  Interactive chat with an agent.  
  `customer-chat WORKDIR` — dedicated PM/customer chat window.

Ops & repos
- `upload WORKDIR PATH... [--dest SUBPATH] [--from SENDER]`  
  Copy files/dirs into `shared/drive` and notify PM.
- `status WORKDIR`  
  Show agent STATUS/inbox timestamps and snippets.
- `sync WORKDIR [--all] [--agent names] [--fetch] [--pull]
  [--no-show-branches]`  
  Git fetch/pull and status for integration checkout and optional agent repos.
- `seed-sync WORKDIR [--clean]`  
  Propagate `seed/` and `seed-extras/` into agent workdirs.
- `update-workdir WORKDIR`  
  Refresh AGENTS.md templates and propagate the latest `cteam.py` into agents.
- `restart WORKDIR [--window NAME|all] [--hard]`  
  Respawn Codex in agent windows (best-effort Ctrl-C with `--hard`).
- `add-agent WORKDIR [--role developer|tester|researcher|architect]
  [--name NAME] [--title TITLE] [--persona TEXT] [--no-tmux] [--start-codex]`  
  Add an agent (PM is unique). PM is notified to onboard and rebalance work.
- `doc-walk WORKDIR [--from SENDER] [--subject TEXT] [--task ID] [--auto]`  
  Kick off a documentation sprint; `--auto` seeds doc tasks to other roles.

Customer & Telegram
- `customer-chat WORKDIR`  
  Local PM/customer chat window (tails `shared/mail/customer/message.md`).
- `telegram-configure WORKDIR [--token TOKEN] [--phone PHONE] [--no-interactive]`  
  Store bot credentials for the authorized customer (send `/start` then share your
  phone/contact from that number to link the chat; if it seems quiet, re-share
  contact to re-authorize).
- `telegram-enable WORKDIR` / `telegram-disable WORKDIR`  
  Enable/disable Telegram bridge (honored by the router).
- Telegram images from the authorized customer are saved to `shared/drive/telegram/`
  and referenced in the PM/customer mailboxes.
