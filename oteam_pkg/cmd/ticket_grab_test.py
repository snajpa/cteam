# Tests for cmd.ticket_grab module
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

import oteam
from oteam_pkg.cmd import ticket_grab


class CmdTicketGrabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = oteam.build_state(
            self.root,
            "Grab Command Test",
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
        (self.root / oteam.DIR_AGENTS / "dev1" / "proj").mkdir(
            parents=True, exist_ok=True
        )
        oteam.git_init_bare(self.root / oteam.DIR_PROJECT_BARE)
        oteam.run_cmd(
            [
                "git",
                "clone",
                str(self.root / oteam.DIR_PROJECT_BARE),
                str(self.root / oteam.DIR_AGENTS / "dev1" / "proj"),
            ],
            cwd=self.root / oteam.DIR_AGENTS / "dev1",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_cmd_ticket_grab_success(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Grab me",
            description="Please grab this ticket",
            creator="pm",
            assignee=None,
            tags=["ready"],
        )
        args = argparse.Namespace(
            workdir=str(self.root),
            ticket_id=str(ticket["id"]),
            agent="dev1",
        )
        with patch("sys.stdout", new_callable=lambda: MagicMock()) as mock_stdout:
            ticket_grab.cmd_ticket_grab(args)
        updated_ticket = oteam._find_ticket_by_id(self.root, str(ticket["id"]))
        self.assertEqual(updated_ticket["assignee"], "dev1")

    def test_cmd_ticket_grab_not_workspace(self) -> None:
        args = argparse.Namespace(
            workdir="/nonexistent",
            ticket_id="1",
            agent="dev1",
        )
        with self.assertRaises(oteam.OTeamError) as ctx:
            ticket_grab.cmd_ticket_grab(args)
        self.assertIn("not an oteam workspace", str(ctx.exception))

    def test_cmd_ticket_grab_not_ready(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Not ready",
            description="Not ready for grab",
            creator="pm",
            assignee=None,
            tags=None,
        )
        args = argparse.Namespace(
            workdir=str(self.root),
            ticket_id=str(ticket["id"]),
            agent="dev1",
        )
        with self.assertRaises(oteam.OTeamError) as ctx:
            ticket_grab.cmd_ticket_grab(args)
        self.assertIn("ready", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
