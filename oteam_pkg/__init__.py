#!/usr/bin/env python3
"""OTeam - Self-Organizing Multi-Agent Coordination System."""

__version__ = "1.0.0"


def main():
    """Main entry point for oteam CLI."""
    import sys
    from pathlib import Path

    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print("Usage: oteam <workdir> <command> [args]")
        print("")
        print("Commands:")
        print("  telegram    Start the Telegram bot")
        print("  tickets     List ready tickets")
        print("  grab        Grab a ticket")
        print("")
        print("Examples:")
        print("  oteam /path/to/workspace telegram start --token TOKEN")
        print("  oteam . tickets --ready")
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"] else 1)

    workdir = Path(sys.argv[1]).resolve()
    command = sys.argv[2] if len(sys.argv) > 2 else None

    if command == "telegram":
        from oteam_pkg.cmd.telegram import main as telegram_main

        telegram_argv = [sys.argv[0], str(workdir)] + sys.argv[3:]
        sys.argv = telegram_argv
        telegram_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: telegram, tickets, grab")
        sys.exit(1)


if __name__ == "__main__":
    main()
