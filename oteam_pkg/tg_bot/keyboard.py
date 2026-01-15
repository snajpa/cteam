#!/usr/bin/env python3
"""
Inline keyboard builders for Telegram bot.
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from typing import List


class TicketKeyboard:
    """Build ticket-related keyboards."""

    @staticmethod
    def build(
        tickets, current_filter: str = "open"
    ) -> List[List[InlineKeyboardButton]]:
        """Build ticket list keyboard with filter tabs."""
        keyboard = [
            [
                InlineKeyboardButton(
                    f"🔵 Open ({len([t for t in tickets if t.status == 'open'])})",
                    callback_data="tickets_filter:open",
                ),
                InlineKeyboardButton(f"🟢 Ready", callback_data="tickets_filter:ready"),
                InlineKeyboardButton(
                    f"🟡 Blocked", callback_data="tickets_filter:blocked"
                ),
            ]
        ]

        row = []
        for ticket in tickets[:10]:
            row.append(
                InlineKeyboardButton(
                    f"T{ticket.id}", callback_data=f"ticket:{ticket.id}"
                )
            )
            if len(row) == 3:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)

        if len(tickets) > 10:
            keyboard.append(
                [
                    InlineKeyboardButton("Prev", callback_data="tickets_page:prev"),
                    InlineKeyboardButton("Next", callback_data="tickets_page:next"),
                ]
            )

        return keyboard

    @staticmethod
    def actions(ticket_id: str) -> List[List[InlineKeyboardButton]]:
        """Build ticket action keyboard."""
        return [
            [
                InlineKeyboardButton(
                    "Assign", callback_data=f"ticket_assign:{ticket_id}"
                ),
                InlineKeyboardButton(
                    "Block", callback_data=f"ticket_block:{ticket_id}"
                ),
                InlineKeyboardButton(
                    "Close", callback_data=f"ticket_close:{ticket_id}"
                ),
            ]
        ]


class AgentKeyboard:
    """Build agent-related keyboards."""

    @staticmethod
    def add_role_selection() -> List[List[InlineKeyboardButton]]:
        """Role selection for /agents add."""
        return [
            [
                InlineKeyboardButton("Developer", callback_data="agent_role:developer"),
                InlineKeyboardButton("Tester", callback_data="agent_role:tester"),
            ],
            [
                InlineKeyboardButton(
                    "Researcher", callback_data="agent_role:researcher"
                ),
                InlineKeyboardButton("Architect", callback_data="agent_role:architect"),
            ],
            [InlineKeyboardButton("Cancel", callback_data="agent_add_cancel")],
        ]

    @staticmethod
    def confirm(name: str, role: str, reason: str) -> List[List[InlineKeyboardButton]]:
        """Confirm agent addition."""
        return [
            [
                InlineKeyboardButton("✓ Confirm", callback_data="agent_add_confirm"),
                InlineKeyboardButton("✗ Cancel", callback_data="agent_add_cancel"),
            ]
        ]
