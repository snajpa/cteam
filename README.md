# OpenCode Team (oteam)

oteam is a multi-agent coordination system for tmux + OpenCode. Agents self-organize around a shared git repo.

## Core Philosophy

**Old (PM-led):** PM assigns → agent waits → implements → PM closes (bottlenecked)

**New (Self-organized):** Agent grabs → plans → implements → CI auto-closes

## Quick Start

```bash
# Initialize workspace
python3 oteam.py /path/to/workspace init --name "My Project" --devs 2

# Attach to tmux
python3 oteam.py /path/to/workspace attach
```

## Self-Assignment Workflow

### 1. List ready tickets

```bash
oteam . tickets list --ready
```

Shows tickets tagged "ready" or "auto" that are open and unassigned.

### 2. Grab a ticket

```bash
oteam . ticket-grab 3 --agent dev1
```

Assigns ticket T3 to dev1 and shows next steps:
```
Ticket T3 assigned to dev1
Branch: agent/dev1/T3

Next steps:
  1. cd agents/dev1/proj
  2. git checkout agent/dev1/T3
  3. Read ticket context (if available)
  4. Press Tab to switch to Plan mode
```

### 3. Plan mode

Press Tab to switch OpenCode to Plan mode. Plan your approach, then Tab again to switch to Build mode.

## Commands

| Command | Description |
|---------|-------------|
| `tickets list --ready` | List self-assignable tickets |
| `ticket-grab <id> --agent <name>` | Self-assign a ticket |
| `pre-push --agent <name>` | Run safety checks before pushing |
| `stuck --on "reason"` | Mark yourself as stuck (notifies PM) |

## Directory Structure

```
oteam_pkg/
├── cmd/                    # CLI commands
│   ├── tickets/list.py     # list --ready
│   └── ticket_grab.py      # self-assign
├── state/tickets.py        # Ready detection, grab logic
├── ctx/                    # Context injection
│   ├── inject.py           # Generate ticket context
│   └── templates/ticket_context.md
├── tmux/                   # OpenCode integration
│   ├── detect_mode.py      # Plan/Build detection
│   └── send_keys.py        # Tab key sending
├── coord/                  # Coordination
│   ├── broadcast.py        # Agent messaging (0 latency)
│   ├── activity.py         # Activity logging
│   ├── stream.py           # Real-time feed
│   └── file_history.py     # "What changed" summary
├── safety/                 # Quality gates
│   ├── pre_push.py         # Pre-push checks
│   └── stuck.py            # Stuck flag
└── hooks/                  # Git hooks
    ├── post-update
    └── post-receive
```

## Context Injection

When grabbing a ticket, oteam generates a 50-line context file with:
- Ticket ID and title
- Previous related work
- Test command suggestion
- Scope IN/OUT
- Verification command

Context files live in `.oteam/runtime/contexts/T{ticket_id}.md`.

## Coordination Layer

### Broadcast (0 latency)

Agents can message each other instantly via file-based broadcast:

```python
from oteam_pkg.coord import broadcast
broadcast.broadcast(root, "dev1", "Hey dev2, need help")
```

No PM involvement, no 15-second mail latency.

### Activity Stream

All agent actions are logged:

```python
from oteam_pkg.coord import activity
activity.log_ticket_grabbed(root, "dev1", "3")
activity.log_ticket_pushed(root, "dev1", "3")
activity.log_ticket_merged(root, "dev1", "3", passed=True)
```

### File History

Get recent changes for a ticket:

```python
from oteam_pkg.coord import file_history
history = file_history.get_file_history(root, ["src/auth.py"])
```

## Safety Features

### Pre-push Check

Run before pushing to catch issues:

```bash
oteam . pre-push --agent dev1
```

Checks:
- Working tree is clean
- Upstream is configured
- No commits behind upstream
- Ticket is assigned to correct agent

### Stuck Flag

Mark yourself as stuck when blocked:

```bash
oteam . stuck --on "waiting on API response"
```

Notifies PM automatically.

### Git Hooks

- `post-update`: Notifies agents of upstream changes
- `post-receive`: Handles push events, logs activity, queues CI

## Testing

```bash
python3 -m pytest oteam_pkg/ -v
```

All 139 tests pass.

## Requirements

- Python 3.9+
- `git` and `tmux` in PATH
- `opencode` CLI in PATH
