# oteam - Self-Organizing Multi-Agent Coordination System

## Overview

oteam is a Python package for coordinating OpenCode agents around a shared git repo. Agents self-assign tickets, plan their work, and push changes with safety checks.

## Package Structure

```
oteam_pkg/
├── cmd/                      # CLI command implementations
│   ├── tickets/
│   │   ├── list.py           # `oteam tickets list --ready`
│   │   └── list_test.py
│   ├── ticket_grab.py        # `oteam ticket-grab <id> --agent <name>`
│   └── ticket_grab_test.py
├── state/                    # State management
│   ├── tickets.py            # Ready detection, grab logic
│   └── tickets_test.py
├── ctx/                      # Context injection
│   ├── inject.py             # Generate ticket context files
│   ├── inject_test.py
│   └── templates/
│       └── ticket_context.md # 50-line context template
├── tmux/                     # OpenCode pane control
│   ├── detect_mode.py        # Detect Plan/Build/Unknown mode
│   ├── detect_mode_test.py
│   ├── send_keys.py          # Send Tab, Enter, text to panes
│   └── send_keys_test.py
├── coord/                    # Coordination layer
│   ├── broadcast.py          # File-based agent messaging (0 latency)
│   ├── broadcast_test.py
│   ├── activity.py           # Log agent actions
│   ├── activity_test.py
│   ├── stream.py             # Real-time activity feed CLI
│   ├── file_history.py       # "What changed" summary
│   └── file_history_test.py
├── safety/                   # Quality gates
│   ├── pre_push.py           # Pre-push safety checks
│   ├── pre_push_test.py
│   └── stuck.py              # Stuck flag for agents
└── hooks/                    # Git hooks
    ├── post-update
    └── post-receive
```

## Key Functions

### Self-Assignment

`oteam_pkg.state.tickets`:
- `is_ready_ticket(ticket)` - Check if ticket has "ready" or "auto" tag
- `load_ready_tickets(root)` - Load all self-assignable tickets
- `can_grab_ticket(root, ticket_id, agent_name)` - Check if agent can grab
- `grab_ticket(root, ticket_id, agent_name, user)` - Assign ticket to agent
- `get_ticket_branch_name(agent_name, ticket_id)` - Get branch name format

### Context Injection

`oteam_pkg.ctx.inject`:
- `generate_context(root, ticket, agent_name)` - Generate context from template
- `inject_context(root, ticket, agent_name)` - Write context file
- `get_context(root, ticket_id)` - Read existing context
- `_parse_scope(description, marker)` - Extract Scope IN/OUT from description

### Mode Detection

`oteam_pkg.tmux.detect_mode`:
- `detect_mode(pane_output)` - Returns "plan", "build", or "unknown"
- `is_plan_mode(pane_output)` - Check if pane is in Plan mode
- `is_build_mode(pane_output)` - Check if pane is in Build mode
- `parse_mode_indicator(pane_output)` - Parse model selector line

### Key Sending

`oteam_pkg.tmux.send_keys`:
- `switch_to_plan_mode(session, window)` - Press Tab to switch to Plan
- `switch_to_build_mode(session, window)` - Press Tab to switch to Build
- `send_enter(session, window)` - Send Enter key
- `send_text(session, window, text)` - Send text to pane
- `capture_pane(session, window)` - Get pane output
- `is_pane_ready(session, window)` - Check if OpenCode is waiting

### Broadcast

`oteam_pkg.coord.broadcast`:
- `broadcast(root, sender, message, recipient, priority)` - Send message
- `notify_agent(root, sender, recipient, message, priority)` - Direct message
- `get_latest_broadcasts(root, after)` - Get broadcasts after timestamp
- `clear_broadcasts(root)` - Clear all broadcasts

### Activity

`oteam_pkg.coord.activity`:
- `log_activity(root, agent, action, details)` - Log an action
- `log_ticket_grabbed(root, agent, ticket_id)` - Log ticket grab
- `log_ticket_pushed(root, agent, ticket_id)` - Log push
- `log_ticket_merged(root, agent, ticket_id, passed)` - Log merge with CI result
- `get_activity(root, limit)` - Get recent activity
- `get_activity_since(root, timestamp)` - Get activity after timestamp

### File History

`oteam_pkg.coord.file_history`:
- `get_file_history(root, files, limit)` - Get recent commits for files
- `get_ticket_file_history(root, ticket, limit)` - Get history for ticket files
- `get_related_changes(root, agent_name, ticket_id)` - Get changes for ticket
- `get_verify_command(root, ticket)` - Suggest verification command
- `suggest_test_command(root, ticket)` - Suggest test command

### Pre-push Check

`oteam_pkg.safety.pre_push`:
- `check_pre_push(root, agent_name)` - Run all pre-push checks
- `check_conflicts_with_main(repo)` - Check for main branch conflicts
- `run_pre_push_check(workdir, agent)` - CLI entry point

### Stuck Flag

`oteam_pkg.safety.stuck`:
- `mark_stuck(root, agent, reason)` - Mark agent as stuck
- `clear_stuck(root, agent)` - Clear stuck status
- `is_stuck(root, agent)` - Check if agent is stuck
- `get_stuck_reason(root, agent)` - Get stuck reason
- `get_stuck_agents(root)` - List all stuck agents

## Ready Ticket Criteria

A ticket is "ready" for self-assignment when:
1. Status is `open`
2. Assignee is `None`
3. Has tag `ready` or `auto`

## Branch Naming

Branch format: `agent/{agent_name}/{ticket_id}`

Examples:
- `agent/dev1/T3`
- `agent/dev1/3`

## Context Template

Located at `oteam_pkg/ctx/templates/ticket_context.md`

Variables:
- `{{ticket_id}}` - Ticket ID
- `{{title}}` - Ticket title
- `{{agent_name}}` - Agent name
- `{{context_file_path}}` - Path to context file
- `{{test_command}}` - Suggested test command
- `{{verify_command}}` - Verification command
- `{{previous_ticket}}` - Previous related ticket
- `{{previous_file}}` - Previous related file
- `{{related_agent}}` - Related agent working on same files
- `{{related_ticket}}` - Related ticket

## Activity Log Format

```
[HH:MM:SS] {agent} {action}: {details}
```

Examples:
```
[14:32:15] dev1 grabbed ticket T3
[14:35:42] dev1 pushed agent/dev1/T3
[14:40:00] dev1 merged T3 (CI passed)
```

## Git Hooks

### post-update

Location: `oteam_pkg/hooks/post-update`

Fetches updates for all agent repos and logs activity.

### post-receive

Location: `oteam_pkg/hooks/post-receive`

Handles push events:
- Logs push activity
- Queues for CI processing
- Logs merge to main (ready for close)

## Tests

All tests in `oteam_pkg/*/*_test.py`:

```bash
python3 -m pytest oteam_pkg/ -v
```

139 tests, all passing.

## Import Pattern

```python
from oteam_pkg.state import tickets as state_tickets
from oteam_pkg.ctx import inject
from oteam_pkg.tmux import detect_mode, send_keys
from oteam_pkg.coord import broadcast, activity, file_history
from oteam_pkg.safety import pre_push, stuck
```
