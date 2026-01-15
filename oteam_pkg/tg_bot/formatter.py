#!/usr/bin/env python3
"""
Telegram message formatter.
"""

from typing import List, Dict


class TelegramFormatter:
    """Format oteam data for Telegram."""

    STATUS_EMOJI = {"open": "🟢", "in_progress": "🔵", "blocked": "🟡", "closed": "⚪"}

    def format_status(self, status: dict) -> str:
        """Format team status."""
        lines = ["👥 **Team Status**", ""]

        for agent in status.get("agents", []):
            emoji = "🔵" if agent.get("status") == "active" else "⚪"
            lines.append(f"{emoji} *{agent['name']}* - {agent.get('task', 'Unknown')}")

        counts = status.get("ticket_counts", {})
        lines.extend(
            [
                "",
                f"📋 Tickets: {counts.get('open', 0)} open, "
                f"{counts.get('ready', 0)} ready, "
                f"{counts.get('blocked', 0)} blocked",
                f"💬 Pending messages: {status.get('pending_messages', 0)}",
            ]
        )

        return "\n".join(lines)

    def format_ticket(self, ticket) -> str:
        """Format single ticket."""
        emoji = self.STATUS_EMOJI.get(ticket.status, "⚪")
        tags = ", ".join(ticket.tags[:3]) if ticket.tags else "none"

        lines = [
            f"{emoji} **T{ticket.id}: {ticket.title}**",
            "-" * 40,
            f"Assignee: {ticket.assignee or 'unassigned'}",
            f"Tags: {tags}",
            "",
            ticket.description[:500] + "..."
            if len(ticket.description) > 500
            else ticket.description,
        ]

        return "\n".join(lines)

    def format_agents(self, agents: List[dict]) -> str:
        """Format agent list."""
        lines = ["👥 **Agents**", "-" * 40]

        for agent in agents:
            emoji = "🔵" if agent.get("status") == "active" else "⚪"
            lines.extend(
                [
                    f"{emoji} **{agent['name']}** ({agent['role']})",
                    f"   Status: {agent.get('task', 'Unknown')}",
                    "",
                ]
            )

        return "\n".join(lines)

    def format_activity(self, activity: List[dict]) -> str:
        """Format activity log."""
        lines = ["📊 **Recent Activity**", "-" * 40]

        for item in activity[-15:]:
            lines.append(item.get("line", ""))

        return "\n".join(lines)
