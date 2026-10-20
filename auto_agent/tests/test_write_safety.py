"""Tests for guarded agent writes and timestamped backups."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import write_safety
from tools.file_tool import read, write


class WriteSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.root_patch = mock.patch.object(
            write_safety, "WORKSPACE_ROOT", self.root
        )
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp_dir.cleanup()

    def test_protected_existing_source_is_redirected(self):
        original = self.root / "main.py"
        original.write_text("original", encoding="utf-8")

        redirected = write_safety.prepare_write_path(original)

        self.assertEqual(redirected, self.root / "main.new.py")
        self.assertEqual(original.read_text(encoding="utf-8"), "original")

    def test_core_source_redirect_can_be_written(self):
        core_dir = self.root / "core"
        core_dir.mkdir()
        original = core_dir / "engine.py"
        original.write_text("original", encoding="utf-8")

        redirected = write_safety.prepare_write_path(original)

        self.assertEqual(redirected, core_dir / "engine.new.py")
        self.assertEqual(write_safety.prepare_write_path(redirected), redirected)

    def test_file_tool_redirects_without_overwriting_source(self):
        original = self.root / "main.py"
        original.write_text("original", encoding="utf-8")

        result = write({"path": str(original), "content": "replacement"})

        self.assertEqual(result, "Ghi file thành công")
        self.assertEqual(original.read_text(encoding="utf-8"), "original")
        self.assertEqual(
            (self.root / "main.new.py").read_text(encoding="utf-8"),
            "replacement",
        )

    def test_env_and_outside_workspace_writes_are_rejected(self):
        with self.assertRaises(write_safety.UnsafeWritePath):
            write_safety.prepare_write_path(self.root / ".env")
        with self.assertRaises(write_safety.UnsafeWritePath):
            write_safety.prepare_write_path(self.root.parent / "outside.txt")
        for relative in (
            "tokens/access.json",
            "cookies.json",
            "credentials/session.txt",
            "api_key.json",
            "private-key.pem",
        ):
            with self.subTest(path=relative), self.assertRaises(
                write_safety.UnsafeWritePath
            ):
                write_safety.prepare_read_path(self.root / relative)

    def test_file_tool_refuses_credential_like_content(self):
        destination = self.root / "draft.txt"

        result = write({
            "path": str(destination),
            "content": "api_key=unit-test-secret",
        })

        self.assertTrue(result.startswith("ERROR [SAFETY]:"))
        self.assertFalse(destination.exists())

    def test_read_paths_are_canonicalized_and_protected(self):
        allowed = self.root / "notes.txt"
        allowed.write_text("safe notes", encoding="utf-8")
        self.assertEqual(read({"path": str(allowed)}), "safe notes")
        with self.assertRaises(write_safety.UnsafeWritePath):
            write_safety.prepare_read_path(self.root / ".." / "outside.txt")
        with self.assertRaises(write_safety.UnsafeWritePath):
            write_safety.prepare_read_path(self.root / ".env")

    def test_read_rejects_symlink_escape(self):
        outside_dir = tempfile.TemporaryDirectory()
        self.addCleanup(outside_dir.cleanup)
        outside_file = Path(outside_dir.name) / "outside.txt"
        outside_file.write_text("not readable by the agent", encoding="utf-8")
        link = self.root / "escape.txt"
        try:
            link.symlink_to(outside_file)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is not available in this environment.")

        result = read({"path": str(link)})
        self.assertTrue(result.startswith("ERROR [SAFETY]:"))
        self.assertNotIn("not readable", result)

    def test_write_rejects_symlink_escape(self):
        outside_dir = tempfile.TemporaryDirectory()
        self.addCleanup(outside_dir.cleanup)
        outside_file = Path(outside_dir.name) / "outside.txt"
        outside_file.write_text("protected", encoding="utf-8")
        link = self.root / "escape.txt"
        try:
            link.symlink_to(outside_file)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is not available in this environment.")

        result = write({"path": str(link), "content": "replacement"})

        self.assertTrue(result.startswith("ERROR [SAFETY]:"))
        self.assertEqual(outside_file.read_text(encoding="utf-8"), "protected")

    def test_backup_rejects_paths_outside_workspace(self):
        outside_dir = tempfile.TemporaryDirectory()
        self.addCleanup(outside_dir.cleanup)
        outside = Path(outside_dir.name) / "outside-backup-target.txt"
        outside.write_text("protected", encoding="utf-8")

        with self.assertRaises(write_safety.UnsafeWritePath):
            write_safety.backup_existing_file(outside)

    def test_backup_retention_never_deletes_existing_copies(self):
        destination = self.root / "output.txt"
        for index in range(7):
            destination.write_text(str(index), encoding="utf-8")
            write_safety.backup_existing_file(destination)

        backups = list(self.root.glob("output.txt.bak_*"))
        self.assertEqual(len(backups), 7)

    def test_run_checkpoint_restores_only_its_own_write(self):
        destination = self.root / "draft.txt"
        destination.write_text("original", encoding="utf-8")
        checkpoint = write_safety.FileCheckpoint()
        token = write_safety.set_active_checkpoint(checkpoint)
        try:
            result = write({"path": str(destination), "content": "agent draft"})
        finally:
            write_safety.reset_active_checkpoint(token)

        self.assertEqual(result, "Ghi file thành công")
        report = checkpoint.rollback()
        self.assertEqual(destination.read_text(encoding="utf-8"), "original")
        self.assertEqual(report["files_restored"], 1)

    def test_checkpoint_preserves_external_changes_on_rollback(self):
        destination = self.root / "draft.txt"
        destination.write_text("original", encoding="utf-8")
        checkpoint = write_safety.FileCheckpoint()
        token = write_safety.set_active_checkpoint(checkpoint)
        try:
            write({"path": str(destination), "content": "agent draft"})
        finally:
            write_safety.reset_active_checkpoint(token)
        destination.write_text("external edit", encoding="utf-8")

        report = checkpoint.rollback()

        self.assertEqual(destination.read_text(encoding="utf-8"), "external edit")
        self.assertEqual(report["files_restored"], 0)
        self.assertEqual(report["files_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
