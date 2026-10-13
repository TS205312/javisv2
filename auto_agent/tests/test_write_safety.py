"""Tests for guarded agent writes and timestamped backups."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import write_safety
from tools.file_tool import write


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

    def test_backup_rotation_keeps_five_latest_copies(self):
        destination = self.root / "output.txt"
        for index in range(7):
            destination.write_text(str(index), encoding="utf-8")
            write_safety.backup_existing_file(destination)

        backups = list(self.root.glob("output.txt.bak_*"))
        self.assertEqual(len(backups), 5)


if __name__ == "__main__":
    unittest.main()
