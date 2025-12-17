import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path
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
        entry = cteam.format_message(ts, "pm", "dev1", "Hello", "Body line", msg_type="MESSAGE")
        parsed_ts, sender, recipient, subject, body = cteam._parse_entry(entry)
        self.assertEqual(parsed_ts, ts)
        self.assertEqual(sender, "pm")
        self.assertEqual(recipient, "dev1")
        self.assertEqual(subject, "Hello")
        self.assertIn("Body line", body)

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
        self.assertIn("We received your message", cust_mail)


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
        args = parser.parse_args(["nudge", "/tmp/workspace", "--to", "dev1", "--interrupt"])
        self.assertTrue(hasattr(args, "interrupt"))
        self.assertTrue(args.interrupt)


if __name__ == "__main__":
    unittest.main()
