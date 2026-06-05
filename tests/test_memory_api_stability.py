import os
import re
import sys
import json
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
from memo_clover import server  # noqa: E402


def _use_temp_database():
    data_dir = Path(_TMP.name)
    db_mod.DATA_DIR = data_dir
    db_mod.DB_PATH = data_dir / "memory.db"
    db_mod.DAILY_LOG_DIR = data_dir / "memory"
    db_mod.BANK_DIR = data_dir / "memory" / "bank"
    db_mod.MEMORY_INDEX = data_dir / "MEMORY.md"
    mm.DATA_DIR = db_mod.DATA_DIR
    mm.DB_PATH = db_mod.DB_PATH
    mm.DAILY_LOG_DIR = db_mod.DAILY_LOG_DIR
    mm.BANK_DIR = db_mod.BANK_DIR
    mm.MEMORY_INDEX = db_mod.MEMORY_INDEX


def _reset_database():
    _use_temp_database()
    db = db_mod._get_db()
    try:
        for table in (
            "memory_review_suggestions",
            "memory_tags",
            "memory_vectors",
            "bank_chunks",
            "daily_logs",
            "conversation_log",
            "memories",
        ):
            db.execute(f"DELETE FROM {table}")
        db.commit()
    finally:
        db.close()
    for path in (Path(_TMP.name) / "memory").glob("*.md"):
        path.unlink()


def _memory_id_from_response(response: str) -> int:
    match = re.search(r"#(\d+)", response)
    if not match:
        raise AssertionError(f"memory id not found in response: {response}")
    return int(match.group(1))


def _memory_row(memory_id: int) -> dict:
    db = db_mod._get_db()
    try:
        row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise AssertionError(f"memory not found: {memory_id}")
        return dict(row)
    finally:
        db.close()


def _set_created_at(memory_id: int, created_at: str) -> None:
    db = db_mod._get_db()
    try:
        db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (created_at, created_at, memory_id),
        )
        db.commit()
    finally:
        db.close()


class MemoryApiStabilityTests(unittest.TestCase):
    def setUp(self):
        mm._embed = lambda _text: None
        _reset_database()

    def test_remember_returns_id_and_id_can_update_memory(self):
        response = server.memory_remember(
            "stable api remember returns id",
            category="facts",
            source="test",
        )

        memory_id = _memory_id_from_response(response)
        self.assertIn("Remembered", response)
        self.assertIn(f"#{memory_id}", response)

        update = server.memory_update(memory_id, content="stable api update by returned id")
        self.assertIn(f"Updated memory #{memory_id}", update)
        self.assertEqual(_memory_row(memory_id)["content"], "stable api update by returned id")

    def test_search_text_results_include_updateable_memory_id(self):
        response = server.memory_remember("stableid searchable memory", category="facts")
        memory_id = _memory_id_from_response(response)

        search = server.memory_search("stableid", limit=5)

        self.assertIn(f"#{memory_id}", search)
        self.assertIn("stableid searchable memory", search)

    def test_search_result_id_can_be_passed_to_update(self):
        response = server.memory_remember("api id handoff original keyword", category="facts")
        memory_id = _memory_id_from_response(response)
        search = server.memory_search("handoff original", limit=5)
        found_id = _memory_id_from_response(search)

        self.assertEqual(found_id, memory_id)
        result = mm.update_memory(found_id, content="api id handoff updated keyword")

        self.assertTrue(result["ok"])
        self.assertEqual(result["id"], memory_id)
        self.assertEqual(_memory_row(memory_id)["content"], "api id handoff updated keyword")

    def test_memory_list_iso_time_filters(self):
        old_id = _memory_id_from_response(server.memory_remember("old iso filter memory"))
        middle_id = _memory_id_from_response(server.memory_remember("middle iso filter memory"))
        new_id = _memory_id_from_response(server.memory_remember("new iso filter memory"))
        _set_created_at(old_id, "2026-05-20 08:00:00")
        _set_created_at(middle_id, "2026-05-21 12:34:56")
        _set_created_at(new_id, "2026-05-22 09:00:00")

        date_only = server.memory_list(after="2026-05-21", before="2026-05-21")
        self.assertIn("middle iso filter memory", date_only)
        self.assertNotIn("old iso filter memory", date_only)
        self.assertNotIn("new iso filter memory", date_only)

        z_suffix = server.memory_list(after="2026-05-21T12:34:56Z")
        self.assertIn("middle iso filter memory", z_suffix)
        self.assertIn("new iso filter memory", z_suffix)
        self.assertNotIn("old iso filter memory", z_suffix)

        offset = server.memory_list(before="2026-05-21T13:34:56+01:00")
        self.assertIn("middle iso filter memory", offset)
        self.assertIn("old iso filter memory", offset)
        self.assertNotIn("new iso filter memory", offset)

    def test_invalid_iso_filter_returns_clear_error(self):
        server.memory_remember("invalid iso guard memory")

        response = server.memory_list(after="not-a-date")

        self.assertIn("Error: invalid ISO 8601 timestamp", response)

    def test_daily_log_read_reports_paths_and_content_without_writing(self):
        write = server.memory_daily_log("test_daily_log_read marker")
        today = mm.now_local().strftime("%Y-%m-%d")

        self.assertIn(f"Logged to {today}", write)
        self.assertIn(f"DB path: {db_mod.DB_PATH.resolve()}", write)
        self.assertIn(f"Markdown file path: {(db_mod.DAILY_LOG_DIR / f'{today}.md').resolve()}", write)

        read_default = server.memory_daily_log_read()
        read_explicit = server.memory_daily_log_read(today)
        invalid = server.memory_daily_log_read("2026-99-99")

        for response in (read_default, read_explicit):
            self.assertIn("Daily log read-only result", response)
            self.assertIn(f"Date: {today}", response)
            self.assertIn(f"DB path: {db_mod.DB_PATH.resolve()}", response)
            self.assertIn(f"Markdown file path: {(db_mod.DAILY_LOG_DIR / f'{today}.md').resolve()}", response)
            self.assertIn("DB row exists: True", response)
            self.assertIn("DB row empty: False", response)
            self.assertIn("Markdown file exists: True", response)
            self.assertIn("Markdown file empty: False", response)
            self.assertIn("Empty: False", response)
            self.assertIn("test_daily_log_read marker", response)
        self.assertIn("Error: date must be a valid calendar date", invalid)

    def test_daily_log_read_missing_storage_does_not_create_files(self):
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(db_mod.DB_PATH) + suffix)
            if path.exists():
                path.unlink()

        response = server.memory_daily_log_read("2026-01-02")

        self.assertIn("Date: 2026-01-02", response)
        self.assertIn("DB exists: False", response)
        self.assertIn("DB row exists: False", response)
        self.assertIn("Markdown file exists: False", response)
        self.assertIn("Empty: True", response)
        self.assertFalse(db_mod.DB_PATH.exists())
        self.assertFalse((db_mod.DAILY_LOG_DIR / "2026-01-02.md").exists())

    def test_memory_find_duplicates_defaults_to_quieter_threshold_and_stays_read_only(self):
        original = server.find_duplicates
        calls = []

        def fake_find_duplicates(threshold=0.85):
            calls.append(threshold)
            return []

        try:
            server.find_duplicates = fake_find_duplicates
            default_response = server.memory_find_duplicates()
            custom_response = server.memory_find_duplicates(threshold=0.95)
        finally:
            server.find_duplicates = original

        self.assertEqual(calls, [0.92, 0.95])
        self.assertIn("above threshold 0.92", default_response)
        self.assertIn("read-only", default_response)
        self.assertIn("never merges or deletes", default_response)
        self.assertIn("above threshold 0.95", custom_response)

    def test_update_content_updated_at_tags_and_search_index(self):
        memory_id = _memory_id_from_response(
            server.memory_remember("oldkeyword update index source", category="facts")
        )
        before = _memory_row(memory_id)["updated_at"]

        result = mm.update_memory(
            memory_id,
            content="newkeyword update index target",
            tags=["api", "stable"],
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["id"], memory_id)
        self.assertIsNotNone(result["updated_at"])
        self.assertNotEqual(result["updated_at"], before)
        self.assertEqual(result["embedding_status"], "pending_reindex")

        row = _memory_row(memory_id)
        self.assertEqual(row["content"], "newkeyword update index target")
        self.assertEqual(row["tags"], '["api", "stable"]')

        new_search = server.memory_search("newkeyword", limit=5)
        old_search = server.memory_search("oldkeyword", limit=5)
        self.assertIn(f"#{memory_id}", new_search)
        self.assertIn("newkeyword update index target", new_search)
        self.assertNotIn("oldkeyword update index source", old_search)

    def test_update_handles_nonexistent_and_empty_updates(self):
        missing = mm.update_memory(999999, content="nope")
        empty = mm.update_memory(999999)

        self.assertFalse(missing["ok"])
        self.assertIn("not found", missing["error"])
        self.assertFalse(empty["ok"])

        memory_id = _memory_id_from_response(server.memory_remember("empty update target"))
        empty_existing = mm.update_memory(memory_id)
        self.assertFalse(empty_existing["ok"])
        self.assertIn("No update fields provided", empty_existing["error"])

    def test_schema_has_layer_column_and_idempotent_migration(self):
        db = db_mod._get_db()
        try:
            db_mod._init_tables(db)
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(memories)").fetchall()
            }
            indexes = {
                row["name"]
                for row in db.execute("PRAGMA index_list(memories)").fetchall()
            }
        finally:
            db.close()

        self.assertIn("layer", columns)
        self.assertIn("idx_memories_layer", indexes)

    def test_layer_remember_list_search_and_update(self):
        legacy_id = _memory_id_from_response(
            server.memory_remember("layercompat shared legacy memory", category="facts")
        )
        project_id = _memory_id_from_response(
            server.memory_remember(
                "layercompat shared project memory",
                category="facts",
                layer="project_memory",
            )
        )
        temp_id = _memory_id_from_response(
            server.memory_remember(
                "layercompat shared temporary memory",
                category="facts",
                layer="temporary_summaries",
            )
        )

        self.assertIsNone(_memory_row(legacy_id)["layer"])
        self.assertEqual(_memory_row(project_id)["layer"], "project_memory")
        self.assertEqual(_memory_row(temp_id)["layer"], "temporary_summaries")

        all_results = server.memory_search("layercompat shared", limit=10)
        self.assertIn("layercompat shared legacy memory", all_results)
        self.assertIn("layercompat shared project memory", all_results)
        self.assertIn("layercompat shared temporary memory", all_results)

        project_results = server.memory_search(
            "layercompat shared",
            limit=10,
            layer="project_memory",
        )
        self.assertIn(f"#{project_id}", project_results)
        self.assertIn("layercompat shared project memory", project_results)
        self.assertNotIn("layercompat shared legacy memory", project_results)
        self.assertNotIn("layercompat shared temporary memory", project_results)

        project_list = server.memory_list(layer="project_memory")
        self.assertIn("layercompat shared project memory", project_list)
        self.assertNotIn("layercompat shared legacy memory", project_list)

        no_clear = server.memory_update(
            project_id,
            content="layercompat project renamed memory",
            layer="",
        )
        self.assertIn(f"Updated memory #{project_id}", no_clear)
        self.assertEqual(_memory_row(project_id)["layer"], "project_memory")

        changed = server.memory_update(project_id, layer="long_term_preferences")
        self.assertIn(f"Updated memory #{project_id}", changed)
        self.assertEqual(_memory_row(project_id)["layer"], "long_term_preferences")

    def test_layer_filter_searches_only_memory_pool(self):
        memory_id = _memory_id_from_response(
            server.memory_remember(
                "poolscope project memory needle",
                layer="project_memory",
            )
        )
        db = db_mod._get_db()
        try:
            db.execute(
                """INSERT INTO conversation_log
                   (platform, direction, speaker, content, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    "test",
                    "in",
                    "tester",
                    "poolscope conversation needle should not appear",
                    "2026-05-25 10:00:00",
                ),
            )
            db.commit()
        finally:
            db.close()

        filtered = server.memory_search("poolscope needle", layer="project_memory")

        self.assertIn(f"#{memory_id}", filtered)
        self.assertIn("poolscope project memory needle", filtered)
        self.assertNotIn("poolscope conversation needle should not appear", filtered)
        self.assertNotIn("Conversation", filtered)

    def test_invalid_layer_returns_clear_errors(self):
        remember = server.memory_remember("bad layer remember", layer="not_a_layer")
        search = server.memory_search("anything", layer="not_a_layer")
        listing = server.memory_list(layer="not_a_layer")

        memory_id = _memory_id_from_response(server.memory_remember("bad layer update target"))
        update = server.memory_update(memory_id, layer="not_a_layer")

        for response in (remember, search, listing, update):
            self.assertIn("Error: invalid memory layer", response)

    def test_normalize_layer_treats_empty_as_legacy(self):
        self.assertIsNone(mm.normalize_layer(None))
        self.assertIsNone(mm.normalize_layer(""))
        self.assertEqual(mm.normalize_layer("project_memory"), "project_memory")
        with self.assertRaises(ValueError):
            mm.normalize_layer("unknown")

    def test_memory_review_deepseek_payload_disables_thinking_by_default(self):
        original_urlopen = mm.urllib.request.urlopen
        original_env = {
            key: os.environ.get(key)
            for key in (
                "MEMORY_REVIEW_API_KEY",
                "MEMORY_REVIEW_THINKING",
                "MEMORY_REVIEW_REASONING_EFFORT",
            )
        }
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps({"suggestions": []})},
                        }
                    ]
                }).encode()

        def fake_urlopen(req, timeout):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        try:
            os.environ["MEMORY_REVIEW_API_KEY"] = "test-review-key"
            os.environ.pop("MEMORY_REVIEW_THINKING", None)
            os.environ.pop("MEMORY_REVIEW_REASONING_EFFORT", None)
            mm.urllib.request.urlopen = fake_urlopen

            result = mm._deepseek_chat_json([{"role": "user", "content": "json"}])
        finally:
            mm.urllib.request.urlopen = original_urlopen
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(result, {"suggestions": []})
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", captured["payload"])

    def test_memory_review_deepseek_payload_can_enable_thinking(self):
        original_urlopen = mm.urllib.request.urlopen
        original_env = {
            key: os.environ.get(key)
            for key in (
                "MEMORY_REVIEW_API_KEY",
                "MEMORY_REVIEW_THINKING",
                "MEMORY_REVIEW_REASONING_EFFORT",
            )
        }
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps({"suggestions": []})},
                        }
                    ]
                }).encode()

        def fake_urlopen(req, timeout):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return FakeResponse()

        try:
            os.environ["MEMORY_REVIEW_API_KEY"] = "test-review-key"
            os.environ["MEMORY_REVIEW_THINKING"] = "enabled"
            os.environ["MEMORY_REVIEW_REASONING_EFFORT"] = "max"
            mm.urllib.request.urlopen = fake_urlopen

            mm._deepseek_chat_json([{"role": "user", "content": "json"}])
        finally:
            mm.urllib.request.urlopen = original_urlopen
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(captured["payload"]["thinking"], {"type": "enabled"})
        self.assertEqual(captured["payload"]["reasoning_effort"], "max")

    def test_memory_review_prompt_requests_chinese_reasons(self):
        messages = mm._memory_review_messages([
            {
                "id": 1,
                "content": "中文记忆审计测试",
                "category": "general",
                "layer": None,
                "source": "test",
                "tags": "[]",
                "importance": 5,
                "created_at": "2026-05-26 12:00:00",
            }
        ])

        self.assertIn("reason 和 merge_suggestion 的值必须是简体中文句子", messages[0]["content"])
        self.assertIn("不要在 reason 或 merge_suggestion 里写英文说明", messages[0]["content"])
        self.assertIn("简短中文判断理由", messages[0]["content"])
        self.assertIn("不要翻译 memory content 本身", messages[0]["content"])

    def test_memory_review_validation_replaces_english_explanations(self):
        suggestions = mm._validate_memory_review_payload(
            {
                "suggestions": [
                    {
                        "memory_id": 1,
                        "suggested_layer": "project_memory",
                        "confidence": 0.82,
                        "duplicate_candidates": [2],
                        "merge_suggestion": "Could be merged with the other project note.",
                        "temporary_summary_like": False,
                        "reason": "Project-scoped operational note.",
                    }
                ]
            },
            {1, 2},
        )

        self.assertIn("项目", suggestions[0]["reason"])
        self.assertIn("可能存在重复", suggestions[0]["merge_suggestion"])
        self.assertNotIn("Project-scoped", suggestions[0]["reason"])
        self.assertNotIn("Could be merged", suggestions[0]["merge_suggestion"])

    def test_memory_review_layers_returns_suggestions_without_writing(self):
        legacy_id = _memory_id_from_response(server.memory_remember("review layer legacy target"))
        project_id = _memory_id_from_response(
            server.memory_remember("review layer existing project", layer="project_memory")
        )
        original_call = mm._deepseek_chat_json

        def fake_deepseek(_messages):
            return {
                "suggestions": [
                    {
                        "memory_id": legacy_id,
                        "suggested_layer": "project_memory",
                        "confidence": 0.82,
                        "duplicate_candidates": [],
                        "merge_suggestion": "",
                        "temporary_summary_like": False,
                        "reason": "project-scoped wording",
                    }
                ]
            }

        try:
            mm._deepseek_chat_json = fake_deepseek
            response = json.loads(server.memory_review_layers(limit=10))
        finally:
            mm._deepseek_chat_json = original_call

        self.assertTrue(response["ok"])
        self.assertTrue(response["dry_run"])
        self.assertFalse(response["wrote"])
        self.assertEqual(response["scanned"], 1)
        self.assertEqual(response["suggestions"][0]["memory_id"], legacy_id)
        self.assertEqual(response["suggestions"][0]["suggested_layer"], "project_memory")
        self.assertIsNone(_memory_row(legacy_id)["layer"])
        self.assertEqual(_memory_row(project_id)["layer"], "project_memory")

    def test_memory_review_layers_fails_closed_on_bad_deepseek_json(self):
        memory_id = _memory_id_from_response(server.memory_remember("review layer invalid json guard"))
        original_call = mm._deepseek_chat_json

        try:
            mm._deepseek_chat_json = lambda _messages: (_ for _ in ()).throw(ValueError("DeepSeek returned invalid JSON"))
            response = json.loads(server.memory_review_layers(limit=5, dry_run=True))
        finally:
            mm._deepseek_chat_json = original_call

        self.assertFalse(response["ok"])
        self.assertFalse(response["wrote"])
        self.assertEqual(response["suggestions"], [])
        self.assertIn("DeepSeek returned invalid JSON", response["errors"][0])
        self.assertIsNone(_memory_row(memory_id)["layer"])

    def test_memory_review_layers_does_not_filter_intimate_content(self):
        content = "review layer intimate romantic context should stay available for audit"
        memory_id = _memory_id_from_response(server.memory_remember(content))
        captured = {}
        original_call = mm._deepseek_chat_json

        def fake_deepseek(messages):
            captured["prompt"] = messages[1]["content"]
            return {
                "suggestions": [
                    {
                        "memory_id": memory_id,
                        "suggested_layer": None,
                        "confidence": 0.2,
                        "duplicate_candidates": [],
                        "merge_suggestion": "",
                        "temporary_summary_like": False,
                        "reason": "uncertain layer",
                    }
                ]
            }

        try:
            mm._deepseek_chat_json = fake_deepseek
            response = json.loads(server.memory_review_layers(limit=5))
        finally:
            mm._deepseek_chat_json = original_call

        self.assertTrue(response["ok"])
        self.assertIn(content, captured["prompt"])
        self.assertEqual(response["suggestions"][0]["memory_id"], memory_id)
        self.assertIsNone(_memory_row(memory_id)["layer"])

    def test_memory_review_layers_rejects_apply_mode_without_writing(self):
        memory_id = _memory_id_from_response(server.memory_remember("review layer apply guard"))

        response = json.loads(server.memory_review_layers(limit=5, dry_run=False))

        self.assertFalse(response["ok"])
        self.assertFalse(response["wrote"])
        self.assertIn("read-only", response["error"])
        self.assertIsNone(_memory_row(memory_id)["layer"])

    def test_memory_review_suggestions_schema_exists(self):
        db = db_mod._get_db()
        try:
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(memory_review_suggestions)").fetchall()
            }
            indexes = {
                row["name"]
                for row in db.execute("PRAGMA index_list(memory_review_suggestions)").fetchall()
            }
        finally:
            db.close()

        self.assertIn("memory_id", columns)
        self.assertIn("suggested_layer", columns)
        self.assertIn("status", columns)
        self.assertIn("idx_memory_review_status", indexes)

    def test_memory_review_layers_can_persist_suggestions_without_changing_memories(self):
        legacy_id = _memory_id_from_response(server.memory_remember("review persist legacy target"))
        original_call = mm._deepseek_chat_json

        def fake_deepseek(_messages):
            return {
                "suggestions": [
                    {
                        "memory_id": legacy_id,
                        "suggested_layer": "project_memory",
                        "confidence": 0.91,
                        "duplicate_candidates": [],
                        "merge_suggestion": "",
                        "temporary_summary_like": False,
                        "reason": "project-scoped test",
                    }
                ]
            }

        try:
            mm._deepseek_chat_json = fake_deepseek
            response = json.loads(server.memory_review_layers(limit=5, persist_suggestions=True))
        finally:
            mm._deepseek_chat_json = original_call

        self.assertTrue(response["ok"])
        self.assertFalse(response["wrote"])
        self.assertEqual(response["persisted_suggestions"], 1)
        self.assertIsNone(_memory_row(legacy_id)["layer"])
        suggestion_id = response["suggestions"][0]["suggestion_id"]
        queue = json.loads(server.memory_review_queue(status="pending"))
        self.assertEqual(queue["suggestions"][0]["id"], suggestion_id)
        self.assertEqual(queue["suggestions"][0]["memory_id"], legacy_id)
        self.assertEqual(queue["suggestions"][0]["suggested_layer"], "project_memory")

    def test_memory_review_apply_layer_only_updates_layer_once(self):
        memory_id = _memory_id_from_response(server.memory_remember("review apply layer target"))
        persisted = mm._persist_memory_review_suggestions(
            [
                {
                    "memory_id": memory_id,
                    "suggested_layer": "temporary_summaries",
                    "confidence": 0.88,
                    "duplicate_candidates": [],
                    "merge_suggestion": "do not execute merge",
                    "temporary_summary_like": True,
                    "reason": "summary-like",
                }
            ],
            model="test-model",
        )
        suggestion_id = persisted[0]["suggestion_id"]

        applied = json.loads(server.memory_review_apply_layer(suggestion_id))
        repeated = json.loads(server.memory_review_apply_layer(suggestion_id))
        row = _memory_row(memory_id)

        self.assertTrue(applied["ok"])
        self.assertEqual(applied["applied_layer"], "temporary_summaries")
        self.assertEqual(row["layer"], "temporary_summaries")
        self.assertEqual(row["content"], "review apply layer target")
        self.assertFalse(repeated["ok"])
        self.assertIn("already applied", repeated["error"])

    def test_memory_review_persist_supersedes_older_pending_suggestion(self):
        memory_id = _memory_id_from_response(server.memory_remember("review duplicate pending target"))
        first = mm._persist_memory_review_suggestions(
            [
                {
                    "memory_id": memory_id,
                    "suggested_layer": "project_memory",
                    "confidence": 0.7,
                    "duplicate_candidates": [],
                    "merge_suggestion": "",
                    "temporary_summary_like": False,
                    "reason": "first suggestion",
                }
            ],
            model="test-model",
        )[0]
        second = mm._persist_memory_review_suggestions(
            [
                {
                    "memory_id": memory_id,
                    "suggested_layer": "long_term_preferences",
                    "confidence": 0.9,
                    "duplicate_candidates": [],
                    "merge_suggestion": "",
                    "temporary_summary_like": False,
                    "reason": "second suggestion",
                }
            ],
            model="test-model",
        )[0]

        pending = json.loads(server.memory_review_queue(status="pending"))
        superseded = json.loads(server.memory_review_queue(status="superseded"))

        self.assertEqual([item["id"] for item in pending["suggestions"]], [second["suggestion_id"]])
        self.assertEqual(pending["suggestions"][0]["suggested_layer"], "long_term_preferences")
        self.assertEqual([item["id"] for item in superseded["suggestions"]], [first["suggestion_id"]])

    def test_memory_review_dismiss_does_not_change_memory(self):
        memory_id = _memory_id_from_response(server.memory_remember("review dismiss target"))
        persisted = mm._persist_memory_review_suggestions(
            [
                {
                    "memory_id": memory_id,
                    "suggested_layer": "project_memory",
                    "confidence": 0.7,
                    "duplicate_candidates": [],
                    "merge_suggestion": "",
                    "temporary_summary_like": False,
                    "reason": "dismiss test",
                }
            ],
            model="test-model",
        )
        suggestion_id = persisted[0]["suggestion_id"]

        dismissed = json.loads(server.memory_review_dismiss(suggestion_id))
        applied = json.loads(server.memory_review_apply_layer(suggestion_id))

        self.assertTrue(dismissed["ok"])
        self.assertFalse(applied["ok"])
        self.assertIn("already dismissed", applied["error"])
        self.assertIsNone(_memory_row(memory_id)["layer"])


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        _TMP.cleanup()
