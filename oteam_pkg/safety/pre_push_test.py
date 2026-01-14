# Tests for safety.pre_push module
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import oteam
from oteam_pkg.safety import pre_push


class PrePushTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = oteam.build_state(
            self.root,
            "Pre-push Test",
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
        self.repo = self.root / oteam.DIR_AGENTS / "dev1" / "proj"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @patch.object(oteam, "run_cmd")
    def test_is_clean_true(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        self.assertTrue(pre_push._is_clean(self.repo))

    @patch.object(oteam, "run_cmd")
    def test_is_clean_false(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "M file.py"
        mock_run.return_value = mock_result
        self.assertFalse(pre_push._is_clean(self.repo))

    def test_extract_ticket_id_valid(self) -> None:
        self.assertEqual(pre_push._extract_ticket_id("agent/dev1/T3"), "T3")
        self.assertEqual(pre_push._extract_ticket_id("agent/dev1/3"), "3")

    def test_extract_ticket_id_invalid(self) -> None:
        self.assertIsNone(pre_push._extract_ticket_id("main"))
        self.assertIsNone(pre_push._extract_ticket_id("feature-branch"))

    def test_extract_ticket_id_none(self) -> None:
        self.assertIsNone(pre_push._extract_ticket_id(None))

    @patch.object(oteam, "run_cmd")
    def test_has_upstream_true(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "origin/agent/dev1/T3"
        mock_run.return_value = mock_result
        self.assertTrue(pre_push._has_upstream(self.repo))

    @patch.object(oteam, "run_cmd")
    def test_has_upstream_false(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = Exception("no upstream")
        self.assertFalse(pre_push._has_upstream(self.repo))

    @patch.object(oteam, "run_cmd")
    def test_get_local_branch(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "agent/dev1/T3"
        mock_run.return_value = mock_result
        self.assertEqual(pre_push._get_local_branch(self.repo), "agent/dev1/T3")

    @patch.object(oteam, "run_cmd")
    def test_get_behind_ahead(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "1\t2"
        mock_run.return_value = mock_result
        behind, ahead = pre_push._get_behind_ahead(self.repo, "local", "remote")
        self.assertEqual(behind, 1)
        self.assertEqual(ahead, 2)

    @patch.object(oteam, "run_cmd")
    def test_get_behind_ahead_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = Exception("fail")
        behind, ahead = pre_push._get_behind_ahead(self.repo, "local", "remote")
        self.assertEqual(behind, 0)
        self.assertEqual(ahead, 0)

    @patch.object(oteam, "run_cmd")
    def test_has_untracked_files_true(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "newfile.py"
        mock_run.return_value = mock_result
        self.assertTrue(pre_push._has_untracked_files(self.repo))

    @patch.object(oteam, "run_cmd")
    def test_has_untracked_files_false(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        self.assertFalse(pre_push._has_untracked_files(self.repo))

    def test_check_pre_push_no_repo(self) -> None:
        all_passed, issues = pre_push.check_pre_push(self.root, "dev2")
        self.assertFalse(all_passed)
        self.assertIn("not found", issues[0].lower())


class CheckConflictsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @patch.object(oteam, "run_cmd")
    def test_check_conflicts_with_main(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "abc123 commit 1\ndef456 commit 2"
        mock_run.return_value = mock_result
        conflicts = pre_push.check_conflicts_with_main(self.repo)
        self.assertEqual(len(conflicts), 2)


if __name__ == "__main__":
    unittest.main()
