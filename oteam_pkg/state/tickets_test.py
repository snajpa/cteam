# Tests for state.tickets module
import tempfile
import unittest
from pathlib import Path

import oteam
from oteam_pkg.state import tickets as state_tickets


class ReadyTicketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = oteam.build_state(
            self.root,
            "Ready Ticket Test",
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

    def test_is_ready_ticket_with_ready_tag(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Ready ticket",
            description="A ready ticket",
            creator="pm",
            assignee=None,
            tags=["ready"],
        )
        self.assertTrue(state_tickets.is_ready_ticket(ticket))

    def test_is_ready_ticket_with_auto_tag(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Auto ticket",
            description="An auto ticket",
            creator="pm",
            assignee=None,
            tags=["auto"],
        )
        self.assertTrue(state_tickets.is_ready_ticket(ticket))

    def test_is_ready_ticket_with_no_tags(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="No tag ticket",
            description="A ticket without ready tags",
            creator="pm",
            assignee=None,
            tags=None,
        )
        self.assertFalse(state_tickets.is_ready_ticket(ticket))

    def test_is_ready_ticket_assigned(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Assigned ticket",
            description="An assigned ticket",
            creator="pm",
            assignee="dev1",
            tags=["ready"],
        )
        self.assertFalse(state_tickets.is_ready_ticket(ticket))

    def test_is_ready_ticket_blocked(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Blocked ticket",
            description="A blocked ticket",
            creator="pm",
            assignee=None,
            tags=["ready"],
        )
        oteam.ticket_block(self.root, str(ticket["id"]), "waiting", "pm")
        ticket = oteam._find_ticket_by_id(self.root, str(ticket["id"]))
        self.assertFalse(state_tickets.is_ready_ticket(ticket))

    def test_load_ready_tickets(self) -> None:
        oteam.ticket_create(
            self.root,
            title="Ready 1",
            description="Ready",
            creator="pm",
            assignee=None,
            tags=["ready"],
        )
        oteam.ticket_create(
            self.root,
            title="Ready 2",
            description="Ready",
            creator="pm",
            assignee=None,
            tags=["auto"],
        )
        oteam.ticket_create(
            self.root,
            title="Not ready",
            description="Not ready",
            creator="pm",
            assignee=None,
            tags=None,
        )
        ready = state_tickets.load_ready_tickets(self.root)
        self.assertEqual(len(ready), 2)
        titles = {t["title"] for t in ready}
        self.assertIn("Ready 1", titles)
        self.assertIn("Ready 2", titles)


class GrabTicketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = oteam.build_state(
            self.root,
            "Grab Ticket Test",
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

    def test_can_grab_ready_ticket(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Grab me",
            description="Please",
            creator="pm",
            assignee=None,
            tags=["ready"],
        )
        can_grab, reason = state_tickets.can_grab_ticket(
            self.root, str(ticket["id"]), "dev1"
        )
        self.assertTrue(can_grab)
        self.assertIsNone(reason)

    def test_cannot_grab_unready_ticket(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Not ready",
            description="Not ready",
            creator="pm",
            assignee=None,
            tags=None,
        )
        can_grab, reason = state_tickets.can_grab_ticket(
            self.root, str(ticket["id"]), "dev1"
        )
        self.assertFalse(can_grab)
        self.assertIn("ready", reason.lower())

    def test_cannot_grab_assigned_ticket(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Taken",
            description="Already taken",
            creator="pm",
            assignee="dev2",
            tags=["ready"],
        )
        can_grab, reason = state_tickets.can_grab_ticket(
            self.root, str(ticket["id"]), "dev1"
        )
        self.assertFalse(can_grab)
        self.assertIn("assigned", reason.lower())

    def test_grab_ticket_assigns(self) -> None:
        ticket = oteam.ticket_create(
            self.root,
            title="Grab me",
            description="Please",
            creator="pm",
            assignee=None,
            tags=["ready"],
        )
        result = state_tickets.grab_ticket(self.root, str(ticket["id"]), "dev1", "dev1")
        self.assertEqual(result["assignee"], "dev1")


class BranchNameTests(unittest.TestCase):
    def test_get_ticket_branch_name(self) -> None:
        branch = state_tickets.get_ticket_branch_name("dev1", "T003")
        self.assertEqual(branch, "agent/dev1/T003")

    def test_get_ticket_branch_name_numeric_only(self) -> None:
        branch = state_tickets.get_ticket_branch_name("dev1", "3")
        self.assertEqual(branch, "agent/dev1/3")


if __name__ == "__main__":
    unittest.main()
