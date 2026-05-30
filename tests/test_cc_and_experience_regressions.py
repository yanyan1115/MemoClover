import os
import sys
import tempfile
import unittest
from pathlib import Path


_TMP = tempfile.TemporaryDirectory()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["IMPRINT_DATA_DIR"] = _TMP.name
os.environ["IMPRINT_DB"] = os.path.join(_TMP.name, "memory.db")
os.environ["EMBED_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = ""

from memo_clover import server  # noqa: E402
from memo_clover import tasks  # noqa: E402


class CcAndExperienceRegressionTests(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(_TMP.name)
        os.environ["IMPRINT_DATA_DIR"] = str(self.data_dir)
        os.environ["IMPRINT_DB"] = str(self.data_dir / "memory.db")
        self.experience_path = self.data_dir / "memory" / "bank" / "experience.md"
        if self.experience_path.exists():
            self.experience_path.unlink()

    def test_experience_append_uses_imprint_data_dir(self):
        result = server.experience_append(
            "test_experience_append_regression",
            "- test_experience_append_regression body",
        )

        self.assertEqual(result, "Added experience: test_experience_append_regression")
        content = self.experience_path.read_text(encoding="utf-8")
        self.assertIn("# Experience Log", content)
        self.assertIn("## test_experience_append_regression", content)
        self.assertIn("- test_experience_append_regression body", content)

    def test_claude_bin_env_override_wins(self):
        env = tasks._build_claude_env(
            {
                "HOME": str(self.data_dir),
                "PATH": "",
                "CLAUDE_BIN": "/opt/test/bin/claude",
                "CLAUDECODE": "1",
            }
        )

        self.assertEqual(tasks._resolve_claude_bin(env), "/opt/test/bin/claude")
        self.assertNotIn("CLAUDECODE", env)
        self.assertIn(str(Path("/opt/test/bin")), env["PATH"])

    def test_claude_bin_falls_back_to_nvm_path(self):
        fake_claude = self.data_dir / ".nvm" / "versions" / "node" / "v24.15.0" / "bin" / "claude"
        fake_claude.parent.mkdir(parents=True, exist_ok=True)
        fake_claude.write_text("#!/bin/sh\n", encoding="utf-8")

        env = tasks._build_claude_env({"HOME": str(self.data_dir), "PATH": ""})

        self.assertIn(str(fake_claude.parent), env["PATH"])
        self.assertEqual(Path(tasks._resolve_claude_bin(env)), fake_claude)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        _TMP.cleanup()
