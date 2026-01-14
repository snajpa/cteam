# Tests for coord.broadcast module
import tempfile
import unittest
from pathlib import Path

import oteam
from oteam_pkg.coord import broadcast


class BroadcastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = oteam.build_state(
            self.root,
            "Broadcast Test",
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
        broadcast.clear_broadcasts(self.root)
        self.tmp.cleanup()

    def test_get_broadcast_dir(self) -> None:
        bcast_dir = broadcast.get_broadcast_dir(self.root)
        self.assertTrue(str(bcast_dir).endswith(".oteam/runtime/broadcasts"))

    def test_get_broadcast_file(self) -> None:
        bcast_file = broadcast.get_broadcast_file(self.root)
        self.assertTrue(str(bcast_file).endswith("broadcasts.log.md"))

    def test_broadcast_creates_file(self) -> None:
        bid = broadcast.broadcast(self.root, "dev1", "Hello all")
        self.assertIsNotNone(bid)
        bcast_file = broadcast.get_broadcast_file(self.root)
        self.assertTrue(bcast_file.exists())

    def test_broadcast_returns_id(self) -> None:
        bid = broadcast.broadcast(self.root, "dev1", "Test")
        self.assertIsNotNone(bid)
        self.assertTrue(len(bid) > 10)

    def test_broadcast_content(self) -> None:
        broadcast.broadcast(self.root, "dev1", "Hello everyone", priority="urgent")
        broadcasts = broadcast.get_latest_broadcasts(self.root)
        self.assertEqual(len(broadcasts), 1)
        self.assertEqual(broadcasts[0]["sender"], "dev1")
        self.assertEqual(broadcasts[0]["priority"], "urgent")

    def test_get_latest_broadcasts(self) -> None:
        broadcast.broadcast(self.root, "dev1", "First")
        broadcast.broadcast(self.root, "dev2", "Second")
        broadcasts = broadcast.get_latest_broadcasts(self.root)
        self.assertEqual(len(broadcasts), 2)

    def test_get_broadcasts_after_timestamp(self) -> None:
        broadcast.broadcast(self.root, "dev1", "First")
        first_ts = oteam.now_iso()
        import time

        time.sleep(1.1)
        broadcast.broadcast(self.root, "dev2", "Second")
        broadcasts = broadcast.get_latest_broadcasts(self.root, after=first_ts)
        self.assertEqual(len(broadcasts), 1)
        self.assertEqual(broadcasts[0]["sender"], "dev2")

    def test_notify_agent(self) -> None:
        bid = broadcast.notify_agent(self.root, "dev1", "dev2", "Hey dev2")
        self.assertIsNotNone(bid)
        broadcasts = broadcast.get_latest_broadcasts(self.root)
        self.assertEqual(len(broadcasts), 1)
        self.assertIn("dev2", broadcasts[0]["message"])

    def test_clear_broadcasts(self) -> None:
        broadcast.broadcast(self.root, "dev1", "Test")
        self.assertEqual(broadcast.get_broadcast_count(self.root), 1)
        broadcast.clear_broadcasts(self.root)
        self.assertEqual(broadcast.get_broadcast_count(self.root), 0)

    def test_get_broadcast_count(self) -> None:
        self.assertEqual(broadcast.get_broadcast_count(self.root), 0)
        broadcast.broadcast(self.root, "dev1", "First")
        self.assertEqual(broadcast.get_broadcast_count(self.root), 1)
        broadcast.broadcast(self.root, "dev2", "Second")
        self.assertEqual(broadcast.get_broadcast_count(self.root), 2)


class BroadcastDirTests(unittest.TestCase):
    def test_broadcast_dir_in_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bcast_dir = broadcast.get_broadcast_dir(root)
            self.assertTrue(str(bcast_dir).startswith(str(root)))
            self.assertIn(".oteam/runtime", str(bcast_dir))


if __name__ == "__main__":
    unittest.main()
