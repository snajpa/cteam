import argparse
import contextlib
import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
import unittest

import cteam


class UtilsTests(unittest.TestCase):
    def test_slugify_and_ts_helpers(self) -> None:
        self.assertEqual(cteam.slugify("Hello World!"), "hello-world")
        self.assertEqual(cteam.slugify("   "), "project")
        ts = "2024-01-02T03:04:05+00:00"
        self.assertEqual(cteam.ts_for_filename(ts), "20240102_030405p0000")

    def test_build_state_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = cteam.build_state(
                root,
                "Clanker Test",
                devs=1,
                mode="new",
                imported_from=None,
                codex_cmd="codex",
                codex_model=None,
                sandbox=cteam.DEFAULT_CODEX_SANDBOX,
                approval=cteam.DEFAULT_CODEX_APPROVAL,
                search=True,
                full_auto=False,
                yolo=False,
                autostart=cteam.DEFAULT_AUTOSTART,
                router=True,
            )
            self.assertEqual(state["codex"]["ask_for_approval"], cteam.DEFAULT_CODEX_APPROVAL)
            self.assertEqual(state["codex"]["sandbox"], cteam.DEFAULT_CODEX_SANDBOX)
            self.assertEqual(state["root_abs"], str(root.resolve()))
            agent_names = {a["name"] for a in state["agents"]}
            self.assertIn("pm", agent_names)
            self.assertIn("dev1", agent_names)
            self.assertTrue(state["shared_abs"].startswith(str(root)))

    def test_upgrade_state_prunes_customer_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_state = {
                "version": 5,
                "project_name": "Legacy",
                "root_abs": str(root),
                "git": {"default_branch": "main"},
                "codex": {
                    "cmd": "codex",
                    "model": None,
                    "sandbox": "workspace-write",
                    "ask_for_approval": "on-request",
                    "search": True,
                    "full_auto": False,
                    "yolo": False,
                },
                "coordination": {"assignment_from": ["pm", "customer"]},
                "tmux": {"session": "s", "router": True, "paused": False},
                "agents": [
                    {"name": "pm", "role": "project_manager", "dir_rel": "agents/pm", "repo_dir_rel": "agents/pm/proj"}
                ],
            }
            (root / cteam.STATE_FILENAME).write_text(json.dumps(old_state))
            upgraded = cteam.upgrade_state_if_needed(root, old_state)
            self.assertEqual(upgraded["version"], cteam.STATE_VERSION)
            self.assertNotIn("customer", upgraded["coordination"]["assignment_from"])
            self.assertEqual(upgraded["coordination"]["assignment_type"], cteam.ASSIGNMENT_TYPE)


class MessagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = cteam.build_state(
            self.root,
            "Messaging Test",
            devs=1,
            mode="new",
            imported_from=None,
            codex_cmd="codex",
            codex_model=None,
            sandbox=cteam.DEFAULT_CODEX_SANDBOX,
            approval=cteam.DEFAULT_CODEX_APPROVAL,
            search=True,
            full_auto=False,
            yolo=False,
            autostart=cteam.DEFAULT_AUTOSTART,
            router=True,
        )
        cteam.ensure_shared_scaffold(self.root, self.state)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_format_and_parse_round_trip(self) -> None:
        ts = cteam.now_iso()
        entry = cteam.format_message(ts, "pm", "dev1", "Hello", "Body line", msg_type="MESSAGE", ticket_id="T001")
        parsed_ts, sender, recipient, subject, body, ticket_id, msg_type = cteam._parse_entry(entry)
        self.assertEqual(parsed_ts, ts)
        self.assertEqual(sender, "pm")
        self.assertEqual(recipient, "dev1")
        self.assertEqual(subject, "Hello")
        self.assertIn("Body line", body)
        self.assertEqual(ticket_id, "T001")
        self.assertEqual(msg_type, "MESSAGE")

    def test_write_message_updates_mail_and_logs(self) -> None:
        cteam.write_message(
            self.root,
            self.state,
            sender="pm",
            recipient="dev1",
            subject="Test subject",
            body="Do the thing",
            nudge=False,
            start_if_needed=False,
        )
        msg_path = self.root / cteam.DIR_MAIL / "dev1" / "message.md"
        log_path = self.root / cteam.DIR_SHARED / "MESSAGES.log.md"
        outbox_dir = self.root / cteam.DIR_MAIL / "pm" / "outbox"
        self.assertIn("Test subject", msg_path.read_text(encoding="utf-8"))
        self.assertIn("Test subject", log_path.read_text(encoding="utf-8"))
        self.assertTrue(list(outbox_dir.glob("*.md")))

    def test_customer_inbound_triggers_ack(self) -> None:
        # Enable telegram to trigger automated ack path.
        self.state.setdefault("telegram", {})["enabled"] = True
        cteam.write_message(
            self.root,
            self.state,
            sender="customer",
            recipient="pm",
            subject="Need help",
            body="Hello from customer",
            nudge=False,
            start_if_needed=False,
        )
        cust_mail = (self.root / cteam.DIR_MAIL / "customer" / "message.md").read_text(encoding="utf-8")
        self.assertIn("Automated acknowledgement: received", cust_mail)

    def test_customer_inbound_sends_pm_summary(self) -> None:
        cteam.write_message(
            self.root,
            self.state,
            sender="customer",
            recipient="pm",
            subject="Need help",
            body="Hello from customer",
            nudge=False,
            start_if_needed=False,
        )
        pm_mail = (self.root / cteam.DIR_MAIL / "pm" / "message.md").read_text(encoding="utf-8")
        # Customer message should appear in PM mailbox.
        self.assertIn("From: customer", pm_mail)
        self.assertIn("Hello from customer", pm_mail)

    def test_write_message_nudge_includes_metadata(self) -> None:
        calls = []

        def fake_nudge(root: Path, state: Dict[str, Any], agent_name: str, *, reason: str = "MAILBOX UPDATED", interrupt: bool = False) -> bool:  # type: ignore
            calls.append(reason)
            return True

        orig_nudge = cteam.nudge_agent
        try:
            cteam.nudge_agent = fake_nudge  # type: ignore
            ticket = cteam.ticket_create(
                self.root,
                self.state,
                title="Test ticket",
                description="desc",
                creator="pm",
                assignee="dev1",
                tags=None,
                assign_note=None,
            )
            cteam.write_message(
                self.root,
                self.state,
                sender="pm",
                recipient="dev1",
                subject="Test subject",
                body="Do the thing",
                msg_type=cteam.ASSIGNMENT_TYPE,
                task=None,
                nudge=True,
                start_if_needed=False,
                ticket_id=ticket["id"],
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("MAILBOX UPDATED", calls[0])
            self.assertIn(ticket["id"], calls[0])
            self.assertIn(cteam.ASSIGNMENT_TYPE, calls[0])
        finally:
            cteam.nudge_agent = orig_nudge  # type: ignore


class CustomerCommandTests(unittest.TestCase):
    def test_ticket_summary_format(self) -> None:
        store = {
            "tickets": [
                {"id": "T002", "title": "Blocked work", "assignee": "dev1", "status": cteam.TICKET_STATUS_BLOCKED, "blocked_on": "T999"},
                {"id": "T001", "title": "Open work", "assignee": None, "status": cteam.TICKET_STATUS_OPEN},
                {"id": "T003", "title": "Closed work", "assignee": "dev2", "status": cteam.TICKET_STATUS_CLOSED},
            ]
        }
        out = cteam._format_ticket_summary(store)
        self.assertIn("T001 [open] @unassigned", out)
        self.assertIn("T002 [blocked] @dev1", out)
        self.assertIn("blocked on T999", out)
        self.assertNotIn("T003", out)


class CodexArgsTests(unittest.TestCase):
    def setUp(self) -> None:
        cteam._codex_caps_cache.clear()

    def test_build_codex_args_respects_flags(self) -> None:
        cteam._codex_caps_cache["fake-codex"] = cteam.CodexCaps(
            cmd="fake-codex",
            help_text="--sandbox --ask-for-approval --search --cd --add-dir --model",
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = cteam.build_state(
                root,
                "Args Test",
                devs=1,
                mode="new",
                imported_from=None,
                codex_cmd="fake-codex",
                codex_model="gpt-5",
                sandbox=cteam.DEFAULT_CODEX_SANDBOX,
                approval=cteam.DEFAULT_CODEX_APPROVAL,
                search=True,
                full_auto=False,
                yolo=False,
                autostart=cteam.DEFAULT_AUTOSTART,
                router=True,
            )
            agent = state["agents"][0]
            args = cteam.build_codex_args_for_agent(state, agent, start_dir=Path(agent["repo_dir_abs"]))
            joined = " ".join(args)
            self.assertIn("--sandbox danger-full-access", joined)
            self.assertIn("--ask-for-approval never", joined)
            self.assertIn("--cd", args)
            self.assertIn("--add-dir", args)
            self.assertIn("--model gpt-5", joined)


class AddAgentTests(unittest.TestCase):
    def test_notify_pm_new_agent_writes_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = cteam.build_state(
                root,
                "Notify PM",
                devs=1,
                mode="new",
                imported_from=None,
                codex_cmd="codex",
                codex_model=None,
                sandbox=cteam.DEFAULT_CODEX_SANDBOX,
                approval=cteam.DEFAULT_CODEX_APPROVAL,
                search=True,
                full_auto=False,
                yolo=False,
                autostart=cteam.DEFAULT_AUTOSTART,
                router=True,
            )
            cteam.ensure_shared_scaffold(root, state)
            new_agent = {"name": "dev2", "title": "Developer", "role": "developer"}
            cteam.notify_pm_new_agent(root, state, new_agent)
            pm_msg = (root / cteam.DIR_MAIL / "pm" / "message.md").read_text(encoding="utf-8")
            self.assertIn("New agent added", pm_msg)
            self.assertIn("balance work", pm_msg.lower())

    def test_add_agent_rejects_pm_role(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = cteam.build_state(
                root,
                "No Extra PM",
                devs=1,
                mode="new",
                imported_from=None,
                codex_cmd="codex",
                codex_model=None,
                sandbox=cteam.DEFAULT_CODEX_SANDBOX,
                approval=cteam.DEFAULT_CODEX_APPROVAL,
                search=True,
                full_auto=False,
                yolo=False,
                autostart=cteam.DEFAULT_AUTOSTART,
                router=True,
            )
            cteam.save_state(root, state)
            args = argparse.Namespace(
                workdir=root,
                role="project_manager",
                name=None,
                title=None,
                persona=None,
                no_tmux=True,
                start_codex=False,
            )
            with self.assertRaises(cteam.CTeamError):
                cteam.cmd_add_agent(args)

    def test_notify_pm_agent_removed_writes_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = cteam.build_state(
                root,
                "Notify PM removal",
                devs=1,
                mode="new",
                imported_from=None,
                codex_cmd="codex",
                codex_model=None,
                sandbox=cteam.DEFAULT_CODEX_SANDBOX,
                approval=cteam.DEFAULT_CODEX_APPROVAL,
                search=True,
                full_auto=False,
                yolo=False,
                autostart=cteam.DEFAULT_AUTOSTART,
                router=True,
            )
            cteam.ensure_shared_scaffold(root, state)
            agent = {"name": "dev2", "title": "Developer", "role": "developer"}
            cteam.notify_pm_agent_removed(root, state, agent, purged=True)
            pm_msg = (root / cteam.DIR_MAIL / "pm" / "message.md").read_text(encoding="utf-8")
            self.assertIn("Agent removed", pm_msg)
            self.assertIn("dev2", pm_msg)


class RemoveAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = cteam.build_state(
            self.root,
            "Remove Agent",
            devs=1,
            mode="new",
            imported_from=None,
            codex_cmd="codex",
            codex_model=None,
            sandbox=cteam.DEFAULT_CODEX_SANDBOX,
            approval=cteam.DEFAULT_CODEX_APPROVAL,
            search=True,
            full_auto=False,
            yolo=False,
            autostart=cteam.DEFAULT_AUTOSTART,
            router=True,
        )
        cteam.save_state(self.root, self.state)
        cteam.ensure_shared_scaffold(self.root, self.state)
        (self.root / cteam.DIR_AGENTS / "dev1" / "proj").mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_remove_agent_updates_state_and_unassigns_tickets(self) -> None:
        ticket = cteam.ticket_create(
            self.root,
            self.state,
            title="Remove me",
            description="desc",
            creator="pm",
            assignee="dev1",
            tags=None,
            assign_note=None,
        )
        args = argparse.Namespace(workdir=self.root, name="dev1", purge=True)
        cteam.cmd_remove_agent(args)

        state = cteam.load_state(self.root)
        self.assertNotIn("dev1", {a["name"] for a in state["agents"]})
        self.assertFalse((self.root / cteam.DIR_AGENTS / "dev1").exists())
        self.assertFalse((self.root / cteam.DIR_MAIL / "dev1").exists())

        roster = (self.root / cteam.DIR_SHARED / "TEAM_ROSTER.md").read_text(encoding="utf-8")
        self.assertNotIn("dev1", roster)

        store = cteam.load_tickets(self.root, state)
        t = next(t for t in store["tickets"] if t["id"] == ticket["id"])
        self.assertIsNone(t.get("assignee"))

        pm_mail = (self.root / cteam.DIR_MAIL / "pm" / "message.md").read_text(encoding="utf-8")
        self.assertIn("Agent removed", pm_mail)

class NudgeQueueTests(unittest.TestCase):
    def test_queue_deduplicates_recent_nudges(self) -> None:
        calls = []

        def fake_nudge(root: Path, state: Dict[str, Any], agent_name: str, *, reason: str = "MAILBOX UPDATED", interrupt: bool = False) -> bool:  # type: ignore
            calls.append((agent_name, reason, interrupt))
            cteam._nudge_history[agent_name] = (time.time(), cteam._split_reasons(reason))
            return True

        orig_nudge = cteam.nudge_agent
        try:
            cteam.nudge_agent = fake_nudge  # type: ignore
            queue = cteam.NudgeQueue(Path("/tmp"), min_interval=30.0, idle_for=0.0)
            queue._is_idle = lambda state, agent: True  # type: ignore
            dummy_state = {"tmux": {"session": "s"}}

            queue.request("dev1", "MAILBOX UPDATED")
            queue.flush(dummy_state)
            self.assertEqual(len(calls), 1)

            queue.request("dev1", "MAILBOX UPDATED")
            queue.flush(dummy_state)
            self.assertEqual(len(calls), 1)  # throttled duplicate

            queue.request("dev1", "NEW REASON")
            queue.flush(dummy_state)
            self.assertEqual(len(calls), 2)
            self.assertIn("NEW REASON", calls[-1][1])
        finally:
            cteam.nudge_agent = orig_nudge  # type: ignore
            cteam._nudge_history.clear()

    def test_queue_deduplicates_mail_variants(self) -> None:
        calls = []

        def fake_nudge(root: Path, state: Dict[str, Any], agent_name: str, *, reason: str = "MAILBOX UPDATED", interrupt: bool = False) -> bool:  # type: ignore
            calls.append((agent_name, reason, interrupt))
            cteam._nudge_history[agent_name] = (time.time(), cteam._normalize_reasons(cteam._split_reasons(reason)))
            return True

        orig_nudge = cteam.nudge_agent
        try:
            cteam.nudge_agent = fake_nudge  # type: ignore
            queue = cteam.NudgeQueue(Path("/tmp"), min_interval=30.0, idle_for=0.0)
            queue._is_idle = lambda state, agent: True  # type: ignore
            dummy_state = {"tmux": {"session": "s"}}

            cteam._nudge_history["dev1"] = (time.time(), {"MAILBOX UPDATED"})
            queue.request("dev1", "T123 MAILBOX UPDATED (ASSIGNMENT)")
            queue.flush(dummy_state)
            self.assertEqual(len(calls), 0)
        finally:
            cteam.nudge_agent = orig_nudge  # type: ignore
            cteam._nudge_history.clear()

    def test_queue_defers_until_idle(self) -> None:
        calls = []

        def fake_nudge(root: Path, state: Dict[str, Any], agent_name: str, *, reason: str = "MAILBOX UPDATED", interrupt: bool = False) -> bool:  # type: ignore
            calls.append((agent_name, reason, interrupt))
            return True

        orig_nudge = cteam.nudge_agent
        try:
            cteam.nudge_agent = fake_nudge  # type: ignore
            queue = cteam.NudgeQueue(Path("/tmp"), min_interval=0.0, idle_for=0.5)
            dummy_state = {"tmux": {"session": "s"}}

            busy_flag = {"idle": False}

            def mock_idle(state: Dict[str, Any], agent: str) -> bool:
                return busy_flag["idle"]

            queue._is_idle = mock_idle  # type: ignore

            queue.request("dev1", "MAILBOX UPDATED")
            results = queue.flush(dummy_state)
            self.assertTrue(any(r[3] == "busy" for r in results))
            self.assertEqual(len(calls), 0)

            busy_flag["idle"] = True
            queue.request("dev1", "MAILBOX UPDATED")
            results = queue.flush(dummy_state)
            self.assertTrue(any(r[1] for r in results))
            self.assertEqual(len(calls), 1)
        finally:
            cteam.nudge_agent = orig_nudge  # type: ignore


class TicketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = cteam.build_state(
            self.root,
            "Ticket Test",
            devs=1,
            mode="new",
            imported_from=None,
            codex_cmd="codex",
            codex_model=None,
            sandbox=cteam.DEFAULT_CODEX_SANDBOX,
            approval=cteam.DEFAULT_CODEX_APPROVAL,
            search=True,
            full_auto=False,
            yolo=False,
            autostart=cteam.DEFAULT_AUTOSTART,
            router=True,
        )
        cteam.save_state(self.root, self.state)
        cteam.ensure_shared_scaffold(self.root, self.state)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ticket_create_and_assign(self) -> None:
        ticket = cteam.ticket_create(
            self.root,
            self.state,
            title="Test ticket",
            description="desc",
            creator="pm",
            assignee=None,
            tags=["a", "b"],
            assign_note=None,
        )
        self.assertEqual(ticket["status"], cteam.TICKET_STATUS_OPEN)
        t2 = cteam.ticket_assign(self.root, self.state, ticket_id=ticket["id"], assignee="dev1", user="pm", note="assign")
        self.assertEqual(t2["assignee"], "dev1")
        store = cteam.load_tickets(self.root, self.state)
        self.assertEqual(store["tickets"][0]["assignee"], "dev1")

    def test_cmd_assign_auto_creates_ticket_and_links_mail(self) -> None:
        args = argparse.Namespace(
            workdir=self.root,
            to="dev1",
            task=None,
            sender="pm",
            subject="Do it",
            file=None,
            no_nudge=True,
            ticket=None,
            title="New work",
            desc="Do something important",
            assign_note=None,
            tags=None,
            text="Body text",
            no_follow=False,
        )
        cteam.cmd_assign(args)
        store = cteam.load_tickets(self.root, self.state)
        self.assertEqual(len(store["tickets"]), 1)

    def test_update_workdir_creates_status_and_links(self) -> None:
        # Remove STATUS to ensure update_workdir regenerates it.
        status = self.root / cteam.DIR_AGENTS / "dev1" / "STATUS.md"
        if status.exists():
            status.unlink()
        # Create bare repo so agent clones succeed.
        cteam.git_init_bare(self.root / cteam.DIR_PROJECT_BARE)
        args = argparse.Namespace(workdir=self.root)
        cteam.cmd_update_workdir(args)
        self.assertTrue(status.exists())
        repo_status = self.root / cteam.DIR_AGENTS / "dev1" / "proj" / "STATUS.md"
        self.assertTrue(repo_status.exists())

    def test_ticket_block_command_notifies(self) -> None:
        t = cteam.ticket_create(
            self.root,
            self.state,
            title="Blockable",
            description="desc",
            creator="pm",
            assignee="dev1",
            tags=None,
            assign_note=None,
        )
        args = argparse.Namespace(
            workdir=self.root,
            ticket_cmd="block",
            id=t["id"],
            on="T999",
            user="pm",
            note="waiting on API",
        )
        cteam.cmd_tickets(args)
        store = cteam.load_tickets(self.root, self.state)
        self.assertEqual(store["tickets"][0]["status"], cteam.TICKET_STATUS_BLOCKED)
        pm_mail = (self.root / cteam.DIR_MAIL / "pm" / "message.md").read_text(encoding="utf-8")
        self.assertIn("blocked", pm_mail.lower())
        self.assertIn(t["id"], pm_mail)


class UpdateWorkdirTests(unittest.TestCase):
    def test_sync_cteam_into_agents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = cteam.build_state(
                root,
                "Sync CTeam",
                devs=1,
                mode="new",
                imported_from=None,
                codex_cmd="codex",
                codex_model=None,
                sandbox=cteam.DEFAULT_CODEX_SANDBOX,
                approval=cteam.DEFAULT_CODEX_APPROVAL,
                search=True,
                full_auto=False,
                yolo=False,
                autostart=cteam.DEFAULT_AUTOSTART,
                router=True,
            )
            cteam.compute_agent_abspaths(state)
            # Prepare minimal agent dirs so sync can place files.
            agent = state["agents"][0]
            a_dir = root / agent["dir_rel"]
            repo_dir = root / agent["repo_dir_rel"]
            a_dir.mkdir(parents=True, exist_ok=True)
            repo_dir.mkdir(parents=True, exist_ok=True)

            cteam.install_self_into_root(root)
            cteam.sync_cteam_into_agents(root, state)

            root_cteam = (root / "cteam.py").read_bytes()
            self.assertEqual((a_dir / "cteam.py").read_bytes(), root_cteam)
            self.assertEqual((repo_dir / "cteam.py").read_bytes(), root_cteam)


class ParserHelpTests(unittest.TestCase):
    def test_usage_error_prints_help(self) -> None:
        parser = cteam.build_parser()
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stderr(buf):
            parser.parse_args([])
        self.assertEqual(ctx.exception.code, 2)
        out = buf.getvalue()
        self.assertIn("usage:", out.lower())
        self.assertIn("cteam error:", out.lower())

    def test_nudge_interrupt_flag_parses(self) -> None:
        parser = cteam.build_parser()
        args = parser.parse_args(["/tmp/workspace", "nudge", "--to", "dev1", "--interrupt"])
        self.assertTrue(hasattr(args, "interrupt"))
        self.assertTrue(args.interrupt)

    def test_telegram_initial_offset_resets_when_unauthorized(self) -> None:
        cfg = {"update_offset": 999999, "chat_id": None, "user_id": None}
        self.assertEqual(cteam._telegram_initial_offset(cfg), 0)
        cfg2 = {"update_offset": 5, "chat_id": 123, "user_id": 456}
        self.assertEqual(cteam._telegram_initial_offset(cfg2), 5)


class ArgRewriteTests(unittest.TestCase):
    def test_rewrite_infers_cwd_when_command_first(self) -> None:
        out = cteam.rewrite_argv_with_default_workdir(["telegram-enable"])
        self.assertEqual(out[:2], [".", "telegram-enable"])

    def test_rewrite_keeps_existing_workdir(self) -> None:
        out = cteam.rewrite_argv_with_default_workdir(["/tmp/ws", "status"])
        self.assertEqual(out[:2], ["/tmp/ws", "status"])

    def test_rewrite_skips_when_extra_positional_present(self) -> None:
        out = cteam.rewrite_argv_with_default_workdir(["init", "init"])
        self.assertEqual(out[:2], ["init", "init"])

    def test_rewrite_allows_flags_only(self) -> None:
        out = cteam.rewrite_argv_with_default_workdir(["status", "--no-attach"])
        self.assertEqual(out[:2], [".", "status"])

    def test_rewrite_handles_flag_with_value(self) -> None:
        out = cteam.rewrite_argv_with_default_workdir(["import", "--src", "/tmp/repo.git"])
        self.assertEqual(out[:2], [".", "import"])


if __name__ == "__main__":
    unittest.main()
