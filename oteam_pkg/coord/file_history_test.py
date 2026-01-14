# Tests for coord.file_history module
import tempfile
import unittest
from pathlib import Path

import oteam
from oteam_pkg.coord import file_history


class FileHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = oteam.build_state(
            self.root,
            "File History Test",
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
        oteam.git_init_bare(self.root / oteam.DIR_PROJECT_BARE)
        (self.root / oteam.DIR_PROJECT_CHECKOUT).mkdir(parents=True, exist_ok=True)
        oteam.run_cmd(
            [
                "git",
                "clone",
                str(self.root / oteam.DIR_PROJECT_BARE),
                str(self.root / oteam.DIR_PROJECT_CHECKOUT),
            ],
            cwd=self.root,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_get_file_history_no_files(self) -> None:
        result = file_history.get_file_history(self.root, [])
        self.assertEqual(result, "(No files specified)")

    def test_get_file_history_no_project(self) -> None:
        import shutil

        shutil.rmtree(self.root / oteam.DIR_PROJECT_CHECKOUT)
        result = file_history.get_file_history(self.root, ["some_file.py"])
        self.assertIn("No recent changes", result)

    def test_extract_files_from_ticket(self) -> None:
        ticket = {
            "id": 1,
            "title": "Test",
            "description": "File: auth.py\nFiles: test.py, main.py",
        }
        files = file_history._extract_files_from_ticket(ticket)
        self.assertIn("auth.py", files)
        self.assertIn("test.py", files)

    def test_suggest_test_command(self) -> None:
        ticket = {"description": "test_auth tests"}
        cmd = file_history.suggest_test_command(self.root, ticket)
        self.assertIn("test_auth", cmd)

    def test_suggest_test_command_make(self) -> None:
        ticket = {"description": "unit tests"}
        cmd = file_history.suggest_test_command(self.root, ticket)
        self.assertEqual(cmd, "make test")

    def test_get_verify_command_explicit(self) -> None:
        ticket = {"description": "Verify: pytest tests/"}
        cmd = file_history.get_verify_command(self.root, ticket)
        self.assertEqual(cmd, "pytest tests/")

    def test_get_verify_command_default(self) -> None:
        ticket = {"description": "Some work"}
        cmd = file_history.get_verify_command(self.root, ticket)
        self.assertEqual(cmd, "make test")

    def test_get_related_changes_no_ticket(self) -> None:
        result = file_history.get_related_changes(self.root, "dev1", "999")
        self.assertIn("not found", result.lower())

    def test_get_ticket_related_activity(self) -> None:
        from oteam_pkg.coord import activity

        activity.log_activity(self.root, "dev1", "grabbed T1")
        result = file_history.get_ticket_related_activity(self.root, "1")
        self.assertIn("grabbed T1", result)


class ExtractFilesTests(unittest.TestCase):
    def test_extract_files_python(self) -> None:
        ticket = {"description": "File: src/auth.py"}
        files = file_history._extract_files_from_ticket(ticket)
        self.assertEqual(files, ["src/auth.py"])

    def test_extract_files_multiple(self) -> None:
        ticket = {"description": "Files: foo.py, bar.py, baz.py"}
        files = file_history._extract_files_from_ticket(ticket)
        self.assertEqual(len(files), 3)

    def test_extract_files_none(self) -> None:
        ticket = {"description": "Just some work"}
        files = file_history._extract_files_from_ticket(ticket)
        self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
