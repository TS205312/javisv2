"""Regression tests ensuring secrets are redacted before persistent logging."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import logger as logger_module


class SensitiveDataTests(unittest.TestCase):
    def test_logger_redacts_sensitive_fields_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(logger_module, "_LOG_DIR", Path(directory)):
                session_logger = logger_module.SessionLogger(
                    agent_id=1,
                    session_id="redaction-test",
                )
                session_logger.log("tool_result", {
                    "api_key": "unit-test-api-secret",
                    "authorization": "Bearer unit-test-bearer-secret",
                    "cookie": "unit-test-cookie-secret",
                })
                log_path = Path(session_logger.filename)
                stored = json.loads(log_path.read_text(encoding="utf-8"))

        serialized = json.dumps(stored)
        self.assertNotIn("unit-test-api-secret", serialized)
        self.assertNotIn("unit-test-bearer-secret", serialized)
        self.assertNotIn("unit-test-cookie-secret", serialized)
        self.assertIn("[REDACTED]", serialized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
