#!/usr/bin/env python3
"""
Tests for telegram module.
"""

import unittest
from pathlib import Path
from oteam_pkg.tg_bot.formatter import TelegramFormatter
from oteam_pkg.tg_bot.keyboard import TicketKeyboard, AgentKeyboard
from oteam_pkg.tg_bot.adapter import OTeamAdapter, Ticket


class FormatterTests(unittest.TestCase):
    def setUp(self):
        self.formatter = TelegramFormatter()

    def test_status_format(self):
        """Status formats correctly."""
        status = {
            "agents": [
                {"name": "dev1", "status": "active", "task": "Working on T001"},
                {"name": "dev2", "status": "inactive", "task": "Idle"},
            ],
            "ticket_counts": {"open": 5, "ready": 2, "blocked": 1},
            "pending_messages": 3,
        }
        result = self.formatter.format_status(status)
        self.assertIn("👥 **Team Status**", result)
        self.assertIn("dev1", result)
        self.assertIn("dev2", result)
        self.assertIn("5 open", result)

    def test_ticket_format(self):
        """Ticket formats correctly."""
        ticket = Ticket(
            id=1,
            title="Test Ticket",
            description="This is a test",
            status="open",
            assignee="dev1",
            tags=["ready", "backend"],
        )
        result = self.formatter.format_ticket(ticket)
        self.assertIn("T1: Test Ticket", result)
        self.assertIn("dev1", result)
        self.assertIn("ready, backend", result)

    def test_agents_format(self):
        """Agent list formats correctly."""
        agents = [
            {"name": "dev1", "role": "developer", "status": "active", "task": "Working"}
        ]
        result = self.formatter.format_agents(agents)
        self.assertIn("👥 **Agents**", result)
        self.assertIn("dev1", result)
        self.assertIn("developer", result)

    def test_activity_format(self):
        """Activity log formats correctly."""
        activity = [
            {"line": "14:32 dev1 pushed to agent/dev1/T001"},
            {"line": "14:30 architect created T005"},
        ]
        result = self.formatter.format_activity(activity)
        self.assertIn("📊 **Recent Activity**", result)
        self.assertIn("dev1", result)


class KeyboardTests(unittest.TestCase):
    def test_ticket_keyboard_build(self):
        """Ticket keyboard builds correctly."""
        tickets = [
            Ticket(1, "Test1", "Desc", "open", None, []),
            Ticket(2, "Test2", "Desc", "open", None, []),
            Ticket(3, "Test3", "Desc", "open", None, []),
        ]
        keyboard = TicketKeyboard.build(tickets, "open")
        self.assertIsInstance(keyboard, list)
        self.assertGreater(len(keyboard), 0)
        self.assertEqual(len(keyboard[0]), 3)

    def test_ticket_keyboard_actions(self):
        """Ticket action keyboard builds correctly."""
        keyboard = TicketKeyboard.actions("T001")
        self.assertIsInstance(keyboard, list)
        self.assertEqual(len(keyboard), 1)
        self.assertEqual(len(keyboard[0]), 3)

    def test_agent_keyboard_role_selection(self):
        """Agent role selection keyboard builds correctly."""
        keyboard = AgentKeyboard.add_role_selection()
        self.assertIsInstance(keyboard, list)
        self.assertGreater(len(keyboard), 0)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path("/root/cteam/oteam_test")
        self.adapter = OTeamAdapter(self.workspace)

    def test_adapter_creates(self):
        """Adapter creates correctly."""
        self.assertEqual(self.adapter.root, self.workspace)

    def test_list_tickets(self):
        """Tickets are listed correctly."""
        tickets = self.adapter.list_tickets()
        self.assertIsInstance(tickets, list)

    def test_list_agents(self):
        """Agents are listed correctly."""
        agents = self.adapter.list_agents()
        self.assertIsInstance(agents, list)
        self.assertGreater(len(agents), 0)
        self.assertEqual(agents[0].name, "dev1")

    def test_get_team_status(self):
        """Team status retrieved correctly."""
        status = self.adapter.get_team_status()
        self.assertIn("agents", status)
        self.assertIn("ticket_counts", status)
        self.assertGreater(len(status["agents"]), 0)


if __name__ == "__main__":
    unittest.main()
