import math
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

from memo_clover.db import _get_db, now_str  # noqa: E402
from memo_clover import memory_manager as mm  # noqa: E402


mm._embed = lambda _text: None


def _fetch_memory(content: str) -> dict:
    db = _get_db()
    try:
        row = db.execute(
            """SELECT id, content, category, valence, arousal, resolved, decay_rate
               FROM memories
               WHERE content = ?""",
            (content,),
        ).fetchone()
        if row is None:
            raise AssertionError(f"Memory not found: {content}")
        return dict(row)
    finally:
        db.close()


class Phase2Tests(unittest.TestCase):
    def test_schema_has_emotional_columns(self):
        db = _get_db()
        try:
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(memories)").fetchall()
            }
        finally:
            db.close()

        self.assertTrue(
            {"valence", "arousal", "resolved", "decay_rate"}.issubset(columns)
        )

    def test_deepseek_embedding_config_loads_from_env(self):
        old_env = {
            key: os.environ.get(key)
            for key in (
                "EMBED_PROVIDER",
                "EMBED_API_BASE",
                "EMBED_API_KEY",
                "DEEPSEEK_API_KEY",
                "OPENAI_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "EMBED_DIMENSIONS",
                "GOOGLE_EMBED_DIMENSIONS",
                "EMBED_MODEL",
            )
        }
        try:
            os.environ["EMBED_PROVIDER"] = "openai"
            os.environ["EMBED_API_BASE"] = "https://api.deepseek.com"
            os.environ["EMBED_API_KEY"] = "sk-test"
            os.environ["EMBED_MODEL"] = "deepseek-v4-flash"
            os.environ.pop("DEEPSEEK_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)

            config = mm._embedding_config_from_env()

            self.assertEqual(config["provider"], "openai")
            self.assertEqual(config["api_base"], "https://api.deepseek.com")
            self.assertEqual(config["api_key"], "sk-test")
            self.assertEqual(config["model"], "deepseek-v4-flash")
            self.assertEqual(
                mm._openai_embeddings_url(config["api_base"], config["api_path"]),
                "https://api.deepseek.com/embeddings",
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            mm._embed = lambda _text: None

    def test_google_embedding_config_loads_from_env(self):
        old_env = {
            key: os.environ.get(key)
            for key in (
                "EMBED_PROVIDER",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "EMBED_DIMENSIONS",
                "GOOGLE_EMBED_DIMENSIONS",
                "EMBED_MODEL",
            )
        }
        try:
            os.environ["EMBED_PROVIDER"] = "google"
            os.environ["GOOGLE_API_KEY"] = "google-test"
            os.environ["EMBED_MODEL"] = "gemini-embedding-2"
            os.environ["EMBED_DIMENSIONS"] = "1536"
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_EMBED_DIMENSIONS", None)

            config = mm._embedding_config_from_env()

            self.assertEqual(config["provider"], "google")
            self.assertEqual(config["google_api_key"], "google-test")
            self.assertEqual(config["model"], "gemini-embedding-2")
            self.assertEqual(config["dimensions"], "1536")
            self.assertEqual(
                mm._google_embeddings_url(config["model"]),
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent",
            )
            self.assertEqual(
                mm._google_embedding_values({"embedding": {"values": [0.1, 0.2]}}),
                [0.1, 0.2],
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            mm._embed = lambda _text: None

    def test_remember_writes_emotional_fields_and_decay_rate(self):
        mm.remember("phase2 facts memory", category="facts")
        facts = _fetch_memory("phase2 facts memory")
        self.assertEqual(facts["decay_rate"], 0.0)

        mm.remember(
            "phase2 event unresolved",
            category="events",
            arousal=0.9,
            resolved=False,
        )
        event = _fetch_memory("phase2 event unresolved")
        self.assertEqual(event["decay_rate"], 0.05)
        self.assertEqual(event["resolved"], 0)
        self.assertEqual(event["arousal"], 0.9)

        mm.remember("phase2 default memory")
        default = _fetch_memory("phase2 default memory")
        self.assertEqual(default["valence"], 0.5)
        self.assertEqual(default["arousal"], 0.3)
        self.assertEqual(default["resolved"], 1)

    def test_rerank_emotional_weight_boosts_unresolved_high_arousal(self):
        base = {
            "importance": 5,
            "recalled_count": 0,
            "created_at": now_str(),
            "last_accessed_at": None,
            "pinned": 0,
            "decay_rate": 0.05,
        }
        high_emotion = {**base, "arousal": 0.9, "resolved": 0}
        calm_resolved = {**base, "arousal": 0.3, "resolved": 1}

        self.assertGreater(
            mm._rerank_memory(0.05, high_emotion),
            mm._rerank_memory(0.05, calm_resolved),
        )

    def test_zero_decay_rate_removes_time_decay(self):
        base = {
            "importance": 5,
            "recalled_count": 0,
            "last_accessed_at": None,
            "pinned": 0,
            "decay_rate": 0.0,
            "arousal": 0.3,
            "resolved": 1,
        }
        old_row = {**base, "created_at": "2000-01-01 00:00:00"}
        recent_row = {**base, "created_at": now_str()}

        self.assertTrue(math.isclose(
            mm._rerank_memory(0.05, old_row),
            mm._rerank_memory(0.05, recent_row),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ))

    def test_update_memory_can_mark_resolved(self):
        mm.remember(
            "phase2 update resolved memory",
            category="events",
            arousal=0.8,
            resolved=False,
        )
        row = _fetch_memory("phase2 update resolved memory")
        self.assertEqual(row["resolved"], 0)

        result = mm.update_memory(row["id"], resolved=1)
        self.assertTrue(result["ok"])

        updated = _fetch_memory("phase2 update resolved memory")
        self.assertEqual(updated["resolved"], 1)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        _TMP.cleanup()
