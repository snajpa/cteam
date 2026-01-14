# Tests for coord.activity module
import tempfile
import unittest
from pathlib import Path

import oteam
from oteam_pkg.coord import activity


class ActivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = oteam.build_state(
            self.root,
            "Activity Test",
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
        activity.clear_activity(self.root)
        self.tmp.cleanup()

    def test_get_activity_dir(self) -> None:
        act_dir = activity.get_activity_dir(self.root)
        self.assertTrue(str(act_dir).endswith(".oteam/runtime/activity"))

    def test_get_activity_log(self) -> None:
        act_log = activity.get_activity_log(self.root)
        self.assertTrue(str(act_log).endswith("activity.log"))

    def test_log_activity(self) -> None:
        activity.log_activity(self.root, "dev1", "grabbed ticket", "T001")
        act_log = activity.get_activity_log(self.root)
        self.assertTrue(act_log.exists())
        content = act_log.read_text()
        self.assertIn("dev1", content)
        self.assertIn("grabbed ticket", content)

    def test_get_activity(self) -> None:
        activity.log_activity(self.root, "dev1", "test action")
        activities = activity.get_activity(self.root)
        self.assertEqual(len(activities), 1)
        self.assertIn("dev1", activities[0])

    def test_get_activity_limit(self) -> None:
        for i in range(10):
            activity.log_activity(self.root, "dev1", f"action {i}")
        activities = activity.get_activity(self.root, limit=5)
        self.assertEqual(len(activities), 5)

    def test_get_activity_multiple(self) -> None:
        activity.log_activity(self.root, "dev1", "first")
        activity.log_activity(self.root, "dev2", "second")
        activities = activity.get_activity(self.root)
        self.assertEqual(len(activities), 2)

    def test_clear_activity(self) -> None:
        activity.log_activity(self.root, "dev1", "test")
        self.assertEqual(activity.activity_count(self.root), 1)
        activity.clear_activity(self.root)
        self.assertEqual(activity.activity_count(self.root), 0)

    def test_activity_count(self) -> None:
        self.assertEqual(activity.activity_count(self.root), 0)
        activity.log_activity(self.root, "dev1", "test")
        self.assertEqual(activity.activity_count(self.root), 1)

    def test_log_ticket_grabbed(self) -> None:
        activity.log_ticket_grabbed(self.root, "dev1", "3")
        activities = activity.get_activity(self.root)
        self.assertEqual(len(activities), 1)
        self.assertIn("grabbed ticket T3", activities[0])

    def test_log_ticket_pushed(self) -> None:
        activity.log_ticket_pushed(self.root, "dev1", "3")
        activities = activity.get_activity(self.root)
        self.assertIn("pushed agent/dev1/T3", activities[0])

    def test_log_ticket_merged_passed(self) -> None:
        activity.log_ticket_merged(self.root, "dev1", "3", passed=True)
        activities = activity.get_activity(self.root)
        self.assertIn("merged T3", activities[0])
        self.assertIn("CI passed", activities[0])

    def test_log_ticket_merged_failed(self) -> None:
        activity.log_ticket_merged(self.root, "dev1", "3", passed=False)
        activities = activity.get_activity(self.root)
        self.assertIn("CI failed", activities[0])


class ActivityDirTests(unittest.TestCase):
    def test_activity_dir_in_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            act_dir = activity.get_activity_dir(root)
            self.assertTrue(str(act_dir).startswith(str(root)))
            self.assertIn(".oteam/runtime", str(act_dir))


if __name__ == "__main__":
    unittest.main()
