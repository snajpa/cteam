# Tests for cmd.tickets.list module
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse
import sys

import oteam
from oteam_pkg.cmd.tickets import list as list_cmd


class CmdTicketsListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = oteam.build_state(
            self.root,
            "List Command Test",
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

    def test_cmd_tickets_list_ready_shows_tickets(self) -> None:
        oteam.ticket_create(
            self.root,
            title="Ready ticket 1",
            description="This is ready",
            creator="pm",
            assignee=None,
            tags=["ready"],
        )
        oteam.ticket_create(
            self.root,
            title="Ready ticket 2",
            description="Also ready",
            creator="pm",
            assignee=None,
            tags=["auto"],
        )
        args = argparse.Namespace(workdir=str(self.root))

        output = []
        with patch.object(sys, "stdout", new_callable=lambda: MagicMock()) as mock:
            mock.write = lambda x: output.append(x)
            list_cmd.cmd_tickets_list(args)
        result = "".join(output)
        self.assertIn("Ready ticket 1", result)
        self.assertIn("Ready ticket 2", result)

    def test_cmd_tickets_list_ready_shows_empty_message(self) -> None:
        args = argparse.Namespace(workdir=str(self.root))

        output = []
        with patch.object(sys, "stdout", new_callable=lambda: MagicMock()) as mock:
            mock.write = lambda x: output.append(x)
            list_cmd.cmd_tickets_list(args)
        result = "".join(output)
        self.assertIn("No ready tickets", result)

    def test_cmd_tickets_list_not_workspace(self) -> None:
        args = argparse.Namespace(workdir="/nonexistent")
        with self.assertRaises(oteam.OTeamError) as ctx:
            list_cmd.cmd_tickets_list(args)
        self.assertIn("not an oteam workspace", str(ctx.exception))

    def test_cmd_tickets_list_excludes_non_ready(self) -> None:
        oteam.ticket_create(
            self.root,
            title="Not ready",
            description="Not ready",
            creator="pm",
            assignee=None,
            tags=None,
        )
        oteam.ticket_create(
            self.root,
            title="Ready",
            description="Ready",
            creator="pm",
            assignee=None,
            tags=["ready"],
        )
        args = argparse.Namespace(workdir=str(self.root))

        output = []
        with patch.object(sys, "stdout", new_callable=lambda: MagicMock()) as mock:
            mock.write = lambda x: output.append(x)
            list_cmd.cmd_tickets_list(args)
        result = "".join(output)
        self.assertIn("Ready", result)
        self.assertNotIn("Not ready", result)


if __name__ == "__main__":
    unittest.main()
