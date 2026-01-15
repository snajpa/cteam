#!/usr/bin/env python3
"""
Telegram bot CLI entry point.

Usage:
    python3 -m oteam_pkg.cmd.telegram <workdir> start --token TELEGRAM_BOT_TOKEN
    python3 -m oteam_pkg.cmd.telegram <workdir> config --token TOKEN
"""

import argparse
import sys
import asyncio
from pathlib import Path
from oteam_pkg.tg_bot.bot import OTeamBot


def main():
    parser = argparse.ArgumentParser(description="OTeam Telegram Bot")
    parser.add_argument("workdir", type=Path, help="oteam workspace directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start the bot")
    start_parser.add_argument("--token", required=True, help="Telegram bot token")

    config_parser = subparsers.add_parser("config", help="Configure bot")
    config_parser.add_argument("--token", required=True, help="Telegram bot token")

    args = parser.parse_args()

    if args.command == "config":
        config_file = args.workdir / "config" / "telegram.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(f'{{"bot_token": "{args.token}"}}\n')
        print(f"Bot token saved to {config_file}")
        return

    if args.command == "start":
        config_file = args.workdir / "config" / "telegram.json"
        if config_file.exists():
            import json

            token = json.loads(config_file.read_text()).get("bot_token", args.token)
        else:
            token = args.token

        bot = OTeamBot(args.workdir, token)
        asyncio.run(bot.run())


if __name__ == "__main__":
    main()
