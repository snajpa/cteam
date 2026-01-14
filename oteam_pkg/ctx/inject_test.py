# Tests for ctx.inject module
import tempfile
import unittest
from pathlib import Path

import oteam
from oteam_pkg.ctx import inject


class ContextInjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = oteam.build_state(
            self.root,
            "Context Test",
            devs=1,
            mode="new",
            imported_from=None,
            opencode_cmd="opencode",
            opencode_model=None,
            sandbox=oteam.DEFAULT_OPENCODE_SANDBOX,
            approval=oteam.DEFAULT_OPENCODE_APPROVAL,
            search=True,
            full_auto=False,
            yolo=False,
            autostart=oteam.DEFAULT_AUTOSTART,
            router=True,
        )
        oteam.save_state(self.root, self.state)
        oteam.ensure_shared_scaffold(self.root, self.state)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_get_contexts_dir(self) -> None:
        ctx_dir = inject.get_contexts_dir(self.root)
        self.assertTrue(ctx_dir.name, "contexts")
        self.assertTrue(str(ctx_dir).endswith(".oteam/runtime/contexts"))

    def test_get_context_file_path(self) -> None:
        ctx_file = inject.get_context_file_path(self.root, "3")
        self.assertTrue(str(ctx_file).endswith("T3.md"))

    def test_generate_context(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Test ticket",
            description="Test description",
            creator="pm",
            assignee="dev1",
            tags=["ready"],
        )
        context = inject.generate_context(self.root, ticket, "dev1")
        self.assertIn("T" + str(ticket["id"]), context)
        self.assertIn("Test ticket", context)
        self.assertIn("dev1", context)

    def test_inject_context_creates_file(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Context test",
            description="Test",
            creator="pm",
            assignee="dev1",
            tags=["ready"],
        )
        ctx_file = inject.inject_context(self.root, ticket, "dev1")
        self.assertTrue(ctx_file.exists())
        content = ctx_file.read_text()
        self.assertIn("Context test", content)

    def test_get_context_when_exists(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Get context test",
            description="Test",
            creator="pm",
            assignee="dev1",
            tags=["ready"],
        )
        inject.inject_context(self.root, ticket, "dev1")
        context = inject.get_context(self.root, str(ticket["id"]))
        self.assertIsNotNone(context)
        self.assertIn("Get context test", context)

    def test_get_context_when_not_exists(self) -> None:
        context = inject.get_context(self.root, "999")
        self.assertIsNone(context)

    def test_parse_scope_in(self) -> None:
        desc = "Scope (IN):\n- item 1\n- item 2\nScope (OUT):\n- out 1"
        scope = inject._parse_scope(desc, "IN")
        self.assertEqual(scope, ["item 1", "item 2"])

    def test_parse_scope_out(self) -> None:
        desc = "Scope (in):\n- item 1\nScope (out):\n- out 1\n- out 2"
        scope = inject._parse_scope(desc, "OUT")
        self.assertEqual(scope, ["out 1", "out 2"])

    def test_guess_test_command(self) -> None:
        ticket = {"id": 1, "title": "Test", "description": "Test something"}
        cmd = inject._guess_test_command(self.root, ticket)
        self.assertEqual(cmd, "make test")


class ContextFilePathTests(unittest.TestCase):
    def test_context_file_in_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx_dir = inject.get_contexts_dir(root)
            self.assertTrue(str(ctx_dir).startswith(str(root)))
            self.assertTrue(str(ctx_dir).endswith(".oteam/runtime/contexts"))


if __name__ == "__main__":
    unittest.main()
