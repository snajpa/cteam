# Stream - Real-time activity feed CLI
import sys
import time
from pathlib import Path

from oteam_pkg.coord import activity


def stream_activity(root: Path, poll_interval: float = 0.5) -> None:
    """Stream activity in real-time.

    Args:
        root: Workspace root path
        poll_interval: Seconds between polls
    """
    activity_log = activity.get_activity_log(root)
    last_pos = 0

    if activity_log.exists():
        last_pos = activity_log.stat().st_size

    print("Streaming activity... (Ctrl-C to stop)")

    try:
        while True:
            if activity_log.exists():
                size = activity_log.stat().st_size
                if size > last_pos:
                    content = activity_log.read_text(encoding="utf-8")
                    new_content = content[last_pos:]
                    print(new_content, end="")
                    sys.stdout.flush()
                    last_pos = size
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nStopped streaming.")


def tail_activity(root: Path, lines: int = 20) -> None:
    """Tail the last N lines of activity.

    Args:
        root: Workspace root path
        lines: Number of lines to show
    """
    activities = activity.get_activity(root, limit=lines)
    for act in activities:
        print(act)


def watch_activity(root: Path, timeout: float = None) -> list:
    """Watch for new activity and return when new activity appears.

    Args:
        root: Workspace root path
        timeout: Optional timeout in seconds

    Returns:
        List of new activity entries
    """
    activity_log = activity.get_activity_log(root)
    initial_count = activity.activity_count(root)

    if not activity_log.exists():
        return []

    deadline = time.time() + timeout if timeout else None

    while True:
        current_count = activity.activity_count(root)
        if current_count > initial_count:
            new_activities = activity.get_activity(root)[
                -(current_count - initial_count) :
            ]
            return new_activities

        if deadline and time.time() > deadline:
            return []

        time.sleep(0.2)
