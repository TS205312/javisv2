"""Tests for persistent, shared and redacted task experience memory."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import memory as memory_module
from tools.write_safety import UnsafeWritePath


class MemoryTests(unittest.TestCase):
    def test_default_storage_uses_designated_d_drive_path(self):
        memory = memory_module.Memory()
        self.assertEqual(
            memory.storage_path,
            Path("D:/JavisAgentMemory/chroma_db").resolve(strict=False),
        )

    def test_arbitrary_external_storage_on_allowed_drive_is_rejected(self):
        with self.assertRaises(UnsafeWritePath):
            memory_module.Memory(storage_path="D:/JavisAgentMemory/other_store")

    def test_experiences_are_saved_with_unique_ids_and_redacted(self):
        with tempfile.TemporaryDirectory(dir=memory_module._WORKSPACE_ROOT) as temp_dir:
            collection = mock.Mock()
            client = mock.Mock()
            client.get_or_create_collection.return_value = collection
            with mock.patch.object(
                memory_module,
                "chromadb",
                SimpleNamespace(PersistentClient=mock.Mock(return_value=client)),
            ):
                memory = memory_module.Memory(storage_path=temp_dir)
                memory.save_long_term(
                    "task-1",
                    "Use api_key=private-value for this task",
                    {"status": "success", "lesson": "Bearer private-token"},
                    success=True,
                    tool="browser_tool",
                    agent_id=2,
                )
                memory.save_long_term(
                    "task-1",
                    "Use api_key=private-value for this task",
                    {"status": "success", "lesson": "Bearer private-token"},
                    success=True,
                    tool="browser_tool",
                )

        first_call, second_call = collection.add.call_args_list
        self.assertNotEqual(first_call.kwargs["ids"], second_call.kwargs["ids"])
        document = first_call.kwargs["documents"][0]
        self.assertIn("api_key=[REDACTED]", document)
        self.assertIn("Bearer [REDACTED]", document)
        self.assertNotIn("private-value", document)
        self.assertEqual(first_call.kwargs["metadatas"][0]["success"], True)
        self.assertEqual(first_call.kwargs["metadatas"][0]["tool"], "browser_tool")
        self.assertEqual(first_call.kwargs["metadatas"][0]["agent_id"], 2)

    def test_desktop_operator_scores_count_only_saved_desktop_outcomes(self):
        collection = mock.Mock()
        collection.get.return_value = {"metadatas": [
            {"agent_id": 1, "success": True},
            {"agent_id": 1, "success": False},
            {"agent_id": 2, "success": True},
            {"success": True},
        ]}
        with tempfile.TemporaryDirectory(dir=memory_module._WORKSPACE_ROOT) as temp_dir:
            with mock.patch.object(
                memory_module,
                "chromadb",
                SimpleNamespace(PersistentClient=mock.Mock()),
            ) as chromadb_mock:
                chromadb_mock.PersistentClient.return_value.get_or_create_collection.return_value = collection
                memory = memory_module.Memory(storage_path=temp_dir)
                self.assertEqual(memory.desktop_operator_scores([1, 2]), {
                    1: {"verified_successes": 1, "attempts": 2},
                    2: {"verified_successes": 1, "attempts": 1},
                })

        collection.get.assert_called_once_with(
            where={"tool": "computer_tool"},
            include=["metadatas"],
            limit=500,
        )

    def test_similar_experiences_are_returned(self):
        collection = mock.Mock()
        collection.query.return_value = {
            "documents": [["related task experience"]]
        }
        with tempfile.TemporaryDirectory(dir=memory_module._WORKSPACE_ROOT) as temp_dir:
            with mock.patch.object(
                memory_module,
                "chromadb",
                SimpleNamespace(PersistentClient=mock.Mock()),
            ) as chromadb_mock:
                chromadb_mock.PersistentClient.return_value.get_or_create_collection.return_value = collection
                memory = memory_module.Memory(storage_path=temp_dir)
                self.assertEqual(
                    memory.search_similar("related task"),
                    ["related task experience"],
                )

        collection.query.assert_called_once_with(
            query_texts=["related task"], n_results=3, where={"success": True}
        )

    def test_similar_search_can_include_failed_experiences_explicitly(self):
        collection = mock.Mock()
        collection.query.return_value = {"documents": [[]]}
        with tempfile.TemporaryDirectory(dir=memory_module._WORKSPACE_ROOT) as temp_dir:
            with mock.patch.object(
                memory_module,
                "chromadb",
                SimpleNamespace(PersistentClient=mock.Mock()),
            ) as chromadb_mock:
                chromadb_mock.PersistentClient.return_value.get_or_create_collection.return_value = collection
                memory = memory_module.Memory(storage_path=temp_dir)
                self.assertEqual(
                    memory.search_similar("related task", success_only=False), []
                )

        collection.query.assert_called_once_with(
            query_texts=["related task"], n_results=3
        )

    def test_unavailable_store_fails_open_without_crashing_agent(self):
        with tempfile.TemporaryDirectory(dir=memory_module._WORKSPACE_ROOT) as temp_dir:
            with mock.patch.object(memory_module, "chromadb", None):
                memory = memory_module.Memory(storage_path=temp_dir)
                self.assertEqual(memory.search_similar("task"), [])
                self.assertFalse(
                    memory.save_long_term("task-1", "task", "result", success=True)
                )

    def test_external_storage_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(UnsafeWritePath):
                memory_module.Memory(storage_path=temp_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
