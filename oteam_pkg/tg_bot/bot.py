#!/usr/bin/env python3
"""
OTeam Telegram Bot
Polls Telegram every 2 seconds, interfaces with concierge agent.
"""

import asyncio
import logging
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)
from oteam_pkg.tg_bot.adapter import OTeamAdapter
from oteam_pkg.tg_bot.formatter import TelegramFormatter
from oteam_pkg.tg_bot.keyboard import TicketKeyboard, AgentKeyboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POLLING_INTERVAL = 2


class OTeamBot:
    def __init__(self, workspace_root: Path, bot_token: str):
        self.root = workspace_root
        self.adapter = OTeamAdapter(workspace_root)
        self.formatter = TelegramFormatter()
        self.app = Application.builder().token(bot_token).build()
        self._setup_handlers()

    def _setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("tickets", self.cmd_tickets))
        self.app.add_handler(CommandHandler("ticket", self.cmd_ticket))
        self.app.add_handler(CommandHandler("agents", self.cmd_agents))
        self.app.add_handler(CommandHandler("activity", self.cmd_activity))
        self.app.add_handler(CommandHandler("ask", self.cmd_ask))
        self.app.add_handler(CommandHandler("feedback", self.cmd_feedback))
        self.app.add_handler(CommandHandler("urgent", self.cmd_urgent))
        self.app.add_handler(CommandHandler("sync", self.cmd_sync))
        self.app.add_handler(CommandHandler("restart", self.cmd_restart))
        self.app.add_handler(CommandHandler("agents_add", self.cmd_agents_add))

        self.app.add_handler(CallbackQueryHandler(self.handle_callback))

        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text)
        )

    async def cmd_start(self, update: Update):
        """Welcome message."""
        status = self.adapter.get_team_status()
        await update.message.reply_text(
            self.formatter.format_status(status), parse_mode="Markdown"
        )

    async def cmd_help(self, update: Update, args: list[str]):
        """Show help."""
        if args:
            topic = args[0]
            help_text = self.get_topic_help(topic)
        else:
            help_text = self.get_main_help()
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def cmd_status(self, update: Update):
        """Team status."""
        status = self.adapter.get_team_status()
        await update.message.reply_text(
            self.formatter.format_status(status), parse_mode="Markdown"
        )

    async def cmd_tickets(self, update: Update, args: list[str]):
        """List tickets with inline keyboard."""
        filter_type = args[0] if args else "open"
        tickets = self.adapter.list_tickets(filter_type)
        keyboard = TicketKeyboard.build(tickets, filter_type)
        await update.message.reply_text(
            "📋 **Tickets**",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )

    async def cmd_ticket(self, update: Update, args: list[str]):
        """Show ticket details."""
        if not args:
            await update.message.reply_text("Usage: /ticket T001")
            return
        ticket_id = args[0]
        ticket = self.adapter.get_ticket(ticket_id)
        if not ticket:
            await update.message.reply_text(f"Ticket {ticket_id} not found")
            return
        keyboard = TicketKeyboard.actions(ticket_id)
        await update.message.reply_text(
            self.formatter.format_ticket(ticket),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )

    async def cmd_agents(self, update: Update):
        """List agents."""
        agents = self.adapter.list_agents()
        await update.message.reply_text(
            self.formatter.format_agents(agents), parse_mode="Markdown"
        )

    async def cmd_activity(self, update: Update):
        """Recent activity."""
        activity = self.adapter.get_activity()
        await update.message.reply_text(
            self.formatter.format_activity(activity), parse_mode="Markdown"
        )

    async def cmd_ask(self, update: Update, args: list[str]):
        """Ask the team a question."""
        message = " ".join(args)
        if not message:
            await update.message.reply_text("Usage: /ask <message>")
            return
        token = self.adapter.send_to_concierge(
            from_customer=update.effective_user.username or "customer",
            message=message,
            msg_type="ask",
        )
        await update.message.reply_text(
            f"✅ Your question has been routed to the team.\nReference: `{token[:8]}`",
            parse_mode="Markdown",
        )

    async def cmd_feedback(self, update: Update, args: list[str]):
        """Leave feedback (FYI, logged)."""
        message = " ".join(args)
        if not message:
            await update.message.reply_text("Usage: /feedback <message>")
            return
        self.adapter.log_feedback(
            from_customer=update.effective_user.username or "customer", message=message
        )
        await update.message.reply_text(
            "✅ Feedback logged. Thank you!\nThe team will review it."
        )

    async def cmd_urgent(self, update: Update, args: list[str]):
        """Mark as urgent."""
        message = " ".join(args)
        if not message:
            await update.message.reply_text("Usage: /urgent <message>")
            return
        token = self.adapter.send_to_concierge(
            from_customer=update.effective_user.username or "customer",
            message=f"URGENT: {message}",
            msg_type="urgent",
            nudge=True,
        )
        await update.message.reply_text(
            f"🚨 **Urgent message sent!**\n"
            f"The team has been notified immediately.\n"
            f"Reference: `{token[:8]}`",
            parse_mode="Markdown",
        )

    async def cmd_sync(self, update: Update):
        """Sync all agents."""
        result = self.adapter.sync_all()
        await update.message.reply_text(
            f"✅ Sync complete: {result['synced']} agents synced"
        )

    async def cmd_restart(self, update: Update, args: list[str]):
        """Restart agent(s)."""
        if not args:
            await update.message.reply_text("Usage: /restart <agent> or /restart all")
            return
        target = args[0]
        if target == "all":
            result = self.adapter.restart_all()
        else:
            result = self.adapter.restart_agent(target)
        await update.message.reply_text(result)

    async def cmd_agents_add(self, update: Update, args: list[str]):
        """Start /agents add wizard or quick add."""
        if args:
            role = args[0]
            if role not in ["developer", "tester", "researcher", "architect"]:
                await update.message.reply_text(
                    "Invalid role. Use: developer, tester, researcher, architect"
                )
                return
            name = (
                args[1]
                if len(args) > 1
                else f"dev{len(self.adapter.list_agents()) + 1}"
            )
            reason = " ".join(args[2:]) if len(args) > 2 else "Added via Telegram"
            result = self.adapter.add_agent(role, name, reason)
            await update.message.reply_text(result["message"])
        else:
            keyboard = AgentKeyboard.add_role_selection()
            await update.message.reply_text(
                "**Add New Agent**\n\nStep 1/4: Role",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown",
            )

    async def handle_callback(self, update: Update):
        """Handle inline keyboard callbacks."""
        query = update.callback_query
        data = query.data

        if data.startswith("ticket:"):
            ticket_id = data.split(":")[1]
            ticket = self.adapter.get_ticket(ticket_id)
            keyboard = TicketKeyboard.actions(ticket_id)
            await query.edit_message_text(
                self.formatter.format_ticket(ticket),
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown",
            )
        elif data.startswith("tickets_filter:"):
            filter_type = data.split(":")[1]
            tickets = self.adapter.list_tickets(filter_type)
            keyboard = TicketKeyboard.build(tickets, filter_type)
            await query.edit_message_reply_markup(InlineKeyboardMarkup(keyboard))
        elif data.startswith("agent_role:"):
            role = data.split(":")[1]
            keyboard = AgentKeyboard.add_role_selection()
            await query.edit_message_text(
                f"**Add {role.title()}**\n\nStep 2/4: Name\n\nEnter agent name:",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown",
            )
        elif data == "agent_add_cancel":
            await query.edit_message_text("❌ Cancelled")

        await query.answer()

    async def handle_text(self, update: Update):
        """Handle text input during interactive flows."""
        pass

    def get_main_help(self) -> str:
        return """
🤖 **OTeam Bot Commands**

**Status & Oversight**
• `/status` - Team activity snapshot
• `/tickets` - List tickets (inline navigation)
• `/ticket T001` - Show ticket details
• `/agents` - List all agents
• `/activity` - Recent team activity

**Communication**
• `/ask <message>` - Ask the team a question
• `/feedback <message>` - Leave feedback (logged)
• `/urgent <message>` - Mark as urgent

**Team Management**
• `/agents add` - Interactive guide to add agent
• `/agents add <role> <name> <reason>` - Quick add
• `/agents remove <name>` - Remove agent
• `/agents suspend <name>` - Pause agent
• `/agents resume <name>` - Resume agent

**Workspace Control**
• `/sync` - Pull latest from all agents
• `/restart <name>` - Restart agent's OpenCode
• `/restart all` - Restart entire team

**Help**
• `/help` - Show this message
• `/help <topic>` - Topic-specific help

━━━━━━━━━━━━━━━━━━━━━━━
Tips:
• Use inline keyboards for navigation
• /ask requires a response
• /feedback is logged (FYI)
""".strip()

    def get_topic_help(self, topic: str) -> str:
        topics = {
            "tickets": """**Ticket Commands**

• `/tickets` - List all tickets (use inline keyboard to filter)
• `/ticket T001` - Show full ticket details
• Ticket statuses: Open → Ready → Blocked → Closed

Tickets are created by PM and can be grabbed by agents.""",
            "agents": """**Agent Commands**

• `/agents` - List all agents and their status
• `/agents add` - Interactive guide to add new agent
• `/restart <name>` - Restart an agent's OpenCode
• `/restart all` - Restart the entire team

Agents work on tickets assigned to them.""",
            "communication": """**Communication Commands**

• `/ask <message>` - Ask the team a question
  → Routed to PM, response expected
  
• `/feedback <message>` - Leave feedback
  → Logged for team review, no response needed
  
• `/urgent <message>` - Mark as urgent
  → Team notified immediately""",
        }
        return topics.get(topic, "Topic not found. /help for all commands.")

    async def run(self):
        """Start polling."""
        logger.info("Starting OTeam Telegram Bot...")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(
            poll_interval=POLLING_INTERVAL, allowed_updates=Update.ALL_TYPES
        )

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.app.stop()
