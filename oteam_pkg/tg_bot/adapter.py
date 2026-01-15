#!/usr/bin/env python3
"""
OTeam Adapter - Bridge between Telegram bot and oteam workspace.
"""

import json
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

DIR_SHARED = "shared"
DIR_MAIL = f"{DIR_SHARED}/mail"
DIR_RUNTIME = ".oteam/runtime"


@dataclass
class Ticket:
    id: int
    title: str
    description: str
    status: str
    assignee: Optional[str]
    tags: List[str]


@dataclass
class Agent:
    name: str
    role: str
    status: str
    current_task: str
    last_activity: str


class OTeamAdapter:
    """Bot ↔ oteam workspace adapter."""

    def __init__(self, workspace_root: Path):
        self.root = workspace_root
        self.state_file = workspace_root / "oteam.json"
        self.tickets_db = workspace_root / DIR_RUNTIME / "tickets.db"
        self.mail_root = workspace_root / DIR_MAIL

    def load_state(self) -> dict:
        """Load oteam state."""
        return json.loads(self.state_file.read_text())

    def get_team_status(self) -> dict:
        """Get team status."""
        state = self.load_state()
        agents = state.get("agents", [])

        agent_statuses = []
        for agent in agents:
            status_file = self.root / agent["dir_rel"] / "STATUS.md"
            if status_file.exists():
                content = status_file.read_text()
                current_task = "Working"
            else:
                current_task = "Unknown"

            agent_statuses.append(
                {
                    "name": agent["name"],
                    "role": agent["role"],
                    "title": agent.get("title", ""),
                    "status": "active",
                    "task": current_task,
                }
            )

        tickets = self.list_tickets()
        ticket_counts = {
            "open": len([t for t in tickets if t.status == "open"]),
            "ready": len([t for t in tickets if "ready" in t.tags]),
            "blocked": len([t for t in tickets if t.status == "blocked"]),
        }

        return {
            "agents": agent_statuses,
            "ticket_counts": ticket_counts,
            "pending_messages": 0,
        }

    def list_tickets(self, filter_type: str = "open") -> List[Ticket]:
        """List tickets."""
        if not self.tickets_db.exists():
            return []

        conn = sqlite3.connect(self.tickets_db)
        cursor = conn.cursor()

        query = "SELECT id, title, description, status, assignee, tags FROM tickets"
        params = []

        if filter_type == "open":
            query += " WHERE status = 'open'"
        elif filter_type == "ready":
            query += " WHERE status = 'open' AND tags LIKE '%ready%'"
        elif filter_type == "blocked":
            query += " WHERE status = 'blocked'"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            Ticket(
                id=row[0],
                title=row[1],
                description=row[2],
                status=row[3],
                assignee=row[4],
                tags=json.loads(row[5]) if row[5] else [],
            )
            for row in rows
        ]

    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Get single ticket."""
        tickets = self.list_tickets()
        normalized = ticket_id.lstrip("Tt")
        for t in tickets:
            if str(t.id) == normalized:
                return t
        return None

    def list_agents(self) -> List[Agent]:
        """List agents."""
        state = self.load_state()
        agents = []
        for a in state.get("agents", []):
            agents.append(
                Agent(
                    name=a["name"],
                    role=a["role"],
                    status="active",
                    current_task="Unknown",
                    last_activity="Unknown",
                )
            )
        return agents

    def get_activity(self) -> List[dict]:
        """Get recent activity."""
        activity_file = self.root / DIR_SHARED / "MESSAGES.log.md"
        if not activity_file.exists():
            return []

        lines = activity_file.read_text().strip().split("\n")[-50:]
        activities = []
        for line in lines[-20:]:
            if line.strip():
                activities.append({"line": line})
        return activities

    def send_to_concierge(
        self, from_customer: str, message: str, msg_type: str, nudge: bool = False
    ) -> str:
        """Send message to concierge agent via mail."""
        concierge_mail = self.mail_root / "concierge" / "message.md"

        timestamp = datetime.now().isoformat()
        token = timestamp.replace(":", "").replace("-", "").replace("T", "_")[:16]

        entry = f"""ID: {token}
Timestamp: {timestamp}
From: {from_customer}
To: concierge
Type: {msg_type}

Subject: {msg_type.upper()} from customer

{message}

---
"""

        concierge_mail.parent.mkdir(parents=True, exist_ok=True)
        with open(concierge_mail, "a") as f:
            f.write(entry)

        return token

    def log_feedback(self, from_customer: str, message: str):
        """Log feedback."""
        feedback_file = self.root / DIR_SHARED / "feedback.log.md"
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] @{from_customer}: {message}\n"
        with open(feedback_file, "a") as f:
            f.write(entry)

    def add_agent(self, role: str, name: str, reason: str) -> dict:
        """Add a new agent."""
        result = subprocess.run(
            ["python3", "oteam.py", ".", "add-agent", "--role", role, "--name", name],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return {"success": True, "message": f"Agent {name} ({role}) added"}
        return {"success": False, "message": result.stderr}

    def remove_agent(self, name: str) -> dict:
        """Remove an agent."""
        result = subprocess.run(
            ["python3", "oteam.py", ".", "remove-agent", "--name", name],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        return {
            "success": result.returncode == 0,
            "message": result.stdout + result.stderr,
        }

    def sync_all(self) -> dict:
        """Sync all agents."""
        result = subprocess.run(
            ["python3", "oteam.py", ".", "sync", "--all"],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        return {"synced": "all agents", "output": result.stdout}

    def restart_agent(self, name: str) -> str:
        """Restart an agent."""
        result = subprocess.run(
            ["python3", "oteam.py", ".", "restart", "--window", name],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        return (
            f"Restarted {name}"
            if result.returncode == 0
            else f"Failed: {result.stderr}"
        )

    def restart_all(self) -> str:
        """Restart all agents."""
        result = subprocess.run(
            ["python3", "oteam.py", ".", "restart", "--window", "all"],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        return (
            "Restarted all agents"
            if result.returncode == 0
            else f"Failed: {result.stderr}"
        )
