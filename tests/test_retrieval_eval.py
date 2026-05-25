import importlib
import os
import sqlite3
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

from memo_clover import db as db_mod  # noqa: E402
from memo_clover import memory_manager as mm  # noqa: E402


VECTORS = {
    "query:SQLite FTS5 检索": [1.0, 0.0, 0.0],
    "query:vector-only climbing": [0.0, 1.0, 0.0],
    "query:bank retrieval playbook": [0.0, 0.0, 1.0],
    "query:telegram incident": [0.0, 0.5, 0.5],
    "exact-memory": [0.95, 0.05, 0.0],
    "weak-semantic": [0.0, 0.0, 0.1],
    "vector-only": [0.0, 0.95, 0.0],
    "bank": [0.0, 0.0, 0.95],
}

EMBEDDINGS = {
    "SQLite FTS5 检索": VECTORS["query:SQLite FTS5 检索"],
    "vector-only climbing": VECTORS["query:vector-only climbing"],
    "bank retrieval playbook": VECTORS["query:bank retrieval playbook"],
    "telegram incident": VECTORS["query:telegram incident"],
    "SQLite FTS5 检索 exact memory baseline": VECTORS["exact-memory"],
    "semantically weak but unrelated vector noise": VECTORS["weak-semantic"],
    "vector only climbing beta movement": VECTORS["vector-only"],
    "## bank retrieval playbook\nBank retrieval playbook for SQLite FTS5 检索 tuning.": VECTORS["bank"],
}


def _mock_embed(text: str):
    return EMBEDDINGS.get(text)


mm._embed = _mock_embed


def _reset_database():
    db = db_mod._get_db()
    try:
        for table in (
            "memory_edges",
            "memory_vectors",
            "bank_chunks",
            "conversation_log",
            "memories",
        ):
            db.execute(f"DELETE FROM {table}")
        db.commit()
    finally:
        db.close()


def _insert_memory(content: str, *, category: str = "facts", importance: int = 5) -> int:
    result = mm.remember(content, category=category, importance=importance)
    if result.startswith("Duplicate") or result.startswith("Semantically similar"):
        raise AssertionError(result)
    db = db_mod._get_db()
    try:
        return db.execute(
            "SELECT id FROM memories WHERE content = ?",
            (content,),
        ).fetchone()["id"]
    finally:
        db.close()


def _insert_conversation(content: str, *, platform: str = "telegram") -> int:
    db = db_mod._get_db()
    try:
        cur = db.execute(
            """INSERT INTO conversation_log
               (platform, direction, speaker, content, session_id, entrypoint, created_at, summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (platform, "in", "User", content, "", "eval", db_mod.now_str(), ""),
        )
        db.commit()
        return cur.lastrowid
    finally:
        db.close()


def _ids_by_content(results: list[dict]) -> dict[str, dict]:
    return {r["content"]: r for r in results}


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        mm._embed = _mock_embed
        _reset_database()
        mm.BANK_DIR.mkdir(parents=True, exist_ok=True)
        for path in mm.BANK_DIR.glob("*.md"):
            path.unlink()

        self.exact_id = _insert_memory("SQLite FTS5 检索 exact memory baseline")
        self.weak_id = _insert_memory("semantically weak but unrelated vector noise")
        self.vector_only_id = _insert_memory("vector only climbing beta movement")
        self.conversation_id = _insert_conversation(
            "telegram incident included SQLite FTS5 检索 diagnostics"
        )

        self.bank_file = mm.BANK_DIR / "retrieval.md"
        self.bank_file.write_text(
            "# Retrieval\n\n## bank retrieval playbook\n"
            "Bank retrieval playbook for SQLite FTS5 检索 tuning.\n",
            encoding="utf-8",
        )

    def test_exact_mixed_language_memory_ranks_above_weak_vector_noise(self):
        results = mm.unified_search(
            "SQLite FTS5 检索",
            pools=["memory"],
            limit=5,
            _internal=True,
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], self.exact_id)

        by_content = _ids_by_content(results)
        self.assertIn("SQLite FTS5 检索 exact memory baseline", by_content)
        self.assertNotIn("semantically weak but unrelated vector noise", by_content)

    def test_low_similarity_vector_is_filtered_before_rrf(self):
        db = db_mod._get_db()
        try:
            _, vec_ranking, _, details = mm._search_memory_channels(
                "SQLite FTS5 检索",
                VECTORS["query:SQLite FTS5 检索"],
                db,
            )
        finally:
            db.close()

        ranked_ids = {details[key]["id"] for key, _ in vec_ranking}
        self.assertIn(self.exact_id, ranked_ids)
        self.assertNotIn(self.weak_id, ranked_ids)

    def test_vector_only_memory_can_surface_when_similarity_is_strong(self):
        results = mm.unified_search(
            "vector-only climbing",
            pools=["memory"],
            limit=5,
            _internal=True,
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], self.vector_only_id)
        self.assertEqual(results[0]["pool"], "memory")
        self.assertGreaterEqual(results[0]["vec_similarity"], mm.VEC_PRE_FILTER)

    def test_bank_pool_returns_indexed_markdown_chunk(self):
        results = mm.unified_search(
            "bank retrieval playbook",
            pools=["bank"],
            limit=5,
            _internal=True,
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["pool"], "bank")
        self.assertIn("Bank retrieval playbook", results[0]["content"])

    def test_conversation_pool_returns_matching_log_entry(self):
        results = mm.unified_search(
            "telegram incident",
            pools=["conversation"],
            limit=5,
            _internal=True,
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["pool"], "conversation")
        self.assertEqual(results[0]["id"], self.conversation_id)
        self.assertIn("telegram incident", results[0]["content"])

    def test_internal_search_does_not_increment_edge_surface_count(self):
        source_id = _insert_memory("edge source recall token")
        target_id = _insert_memory("neighbor-only context")
        edge = mm.add_edge(source_id, target_id, "related", "test edge")
        self.assertTrue(edge["ok"])

        internal_results = mm.unified_search(
            "edge source recall token",
            pools=["memory"],
            limit=5,
            _internal=True,
        )
        self.assertIn(source_id, [r["id"] for r in internal_results])
        self.assertNotIn(target_id, [r["id"] for r in internal_results if r.get("edge_relation")])

        db = db_mod._get_db()
        try:
            surfaced = db.execute(
                "SELECT surfaced_count FROM memory_edges WHERE id = ?",
                (edge["edge_id"],),
            ).fetchone()["surfaced_count"]
        finally:
            db.close()
        self.assertEqual(surfaced, 0)

        normal_results = mm.unified_search(
            "edge source recall token",
            pools=["memory"],
            limit=5,
            _internal=False,
        )
        self.assertIn(target_id, [r["id"] for r in normal_results if r.get("edge_relation")])

        db = db_mod._get_db()
        try:
            surfaced = db.execute(
                "SELECT surfaced_count FROM memory_edges WHERE id = ?",
                (edge["edge_id"],),
            ).fetchone()["surfaced_count"]
        finally:
            db.close()
        self.assertEqual(surfaced, 1)

    def test_single_cjk_word_query_recalls_memory_in_text_fallback(self):
        target_id = _insert_memory(
            "衿衿喜欢在安静的早晨整理长期记忆。",
            category="core_profile",
        )

        results = mm.unified_search(
            "衿衿",
            pools=["memory"],
            limit=5,
            _internal=True,
        )

        self.assertIn(target_id, [r["id"] for r in results])
        self.assertTrue(all(r["search_mode"] == "fts5_fallback" for r in results))

    def test_multi_term_query_uses_fts5_or_recall(self):
        like_id = _insert_memory("衿衿喜欢温柔、稳定、可持续的协作节奏。")
        habit_id = _insert_memory("衿衿的习惯是先确认上下文，再推进实现。")

        fts_query = db_mod.build_fts_match_query("喜欢 偏好 习惯")
        self.assertIn(" OR ", fts_query)
        self.assertNotIn(" AND ", fts_query)

        results = mm.unified_search(
            "喜欢 偏好 习惯",
            pools=["memory"],
            limit=10,
            _internal=True,
        )
        result_ids = {r["id"] for r in results}

        self.assertIn(like_id, result_ids)
        self.assertIn(habit_id, result_ids)
        self.assertTrue(all(r["search_mode"] == "fts5_fallback" for r in results))

    def test_long_query_recalls_memory_when_embedding_provider_fails(self):
        target_id = _insert_memory(
            "bge-m3 service outage caused memory_search to silently degrade into keyword retrieval",
            category="experience",
        )

        long_query = (
            "Can you help me investigate why a very long memory_search prompt about "
            "the bge-m3 service outage and silent keyword fallback has terrible recall "
            "even though the relevant memory clearly mentions provider failure diagnostics?"
        )

        old_embed = mm._embed

        def _failing_provider(text: str):
            raise RuntimeError("simulated embedding provider outage")

        try:
            importlib.reload(mm)
            old_openai = mm._embed_openai
            mm._embed_openai = _failing_provider
            with self.assertLogs("memo_clover.memory_manager", level="WARNING") as logs:
                results = mm.unified_search(
                    long_query,
                    pools=["memory"],
                    limit=5,
                    _internal=True,
                )
        finally:
            mm._embed = old_embed
            if "old_openai" in locals():
                mm._embed_openai = old_openai

        self.assertIn(target_id, [r["id"] for r in results])
        self.assertTrue(all(r["search_mode"] == "fts5_fallback" for r in results))
        self.assertIn(
            "Vector retrieval will fall back to text-only search",
            "\n".join(logs.output),
        )


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        _TMP.cleanup()
