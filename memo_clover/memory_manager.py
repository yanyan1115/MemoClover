"""
MemoClover — Memory System
Pure memory operations: CRUD, hybrid search (FTS5 + bge-m3), bank indexing, daily log.
Includes RRF unified retrieval across memory, bank, and conversation pools.
"""

import json
import logging
import math
import os
import re
import sqlite3
import struct
import urllib.request
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

from .db import (
    _get_db, now_local, now_str,
    DATA_DIR, DB_PATH, DAILY_LOG_DIR, BANK_DIR, MEMORY_INDEX, LOCAL_TZ,
    build_fts_match_query, like_search_terms, segment_cjk,
)

logger = logging.getLogger(__name__)

# ─── Embedding Config ────────────────────────────────────
_DEFAULT_MODELS = {"ollama": "bge-m3", "openai": "text-embedding-3-small"}


def _embedding_config_from_env() -> dict[str, str]:
    """Resolve embedding settings while preserving legacy OPENAI_API_KEY support."""
    provider = (os.environ.get("EMBED_PROVIDER") or "ollama").strip().lower()
    return {
        "provider": provider,
        "ollama_url": (os.environ.get("OLLAMA_URL") or "http://localhost:11434").strip(),
        "api_key": (
            os.environ.get("EMBED_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip(),
        "api_base": (os.environ.get("EMBED_API_BASE") or "https://api.openai.com").strip(),
        "api_path": (os.environ.get("EMBED_API_PATH") or "").strip(),
        "model": (os.environ.get("EMBED_MODEL") or _DEFAULT_MODELS.get(provider, "bge-m3")).strip(),
    }


def _openai_embeddings_url(base: str | None = None, path: str | None = None) -> str:
    """Build the embeddings URL for OpenAI-compatible providers."""
    resolved_base = (base or EMBED_API_BASE or "https://api.openai.com").rstrip("/")
    resolved_path = EMBED_API_PATH if path is None else path
    resolved_path = (resolved_path or "").strip()
    if resolved_path:
        return f"{resolved_base}/{resolved_path.lstrip('/')}"
    if resolved_base.lower().endswith("/v1"):
        return f"{resolved_base}/embeddings"
    if "api.deepseek.com" in resolved_base.lower():
        return f"{resolved_base}/embeddings"
    return f"{resolved_base}/v1/embeddings"


_EMBED_CONFIG = _embedding_config_from_env()
EMBED_PROVIDER = _EMBED_CONFIG["provider"]  # "ollama" or "openai"
OLLAMA_URL = _EMBED_CONFIG["ollama_url"]
EMBED_API_KEY = _EMBED_CONFIG["api_key"]
OPENAI_API_KEY = EMBED_API_KEY
EMBED_API_BASE = _EMBED_CONFIG["api_base"]
EMBED_API_PATH = _EMBED_CONFIG["api_path"]
EMBED_MODEL = _EMBED_CONFIG["model"]

BANK_INDEX_VERSION = 2

# Hybrid search weights
WEIGHT_VECTOR = 0.4
WEIGHT_FTS = 0.4
WEIGHT_RECENCY = 0.2


# ─── Vector Embeddings ───────────────────────────────────

def _embed_ollama(text: str) -> Optional[list[float]]:
    """Generate embedding via Ollama (local)."""
    try:
        payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings[0]) > 0:
                return embeddings[0]
            logger.warning(
                "Ollama embedding returned an empty vector payload; provider=%s model=%s url=%s. "
                "Vector retrieval will fall back to text-only search.",
                EMBED_PROVIDER,
                EMBED_MODEL,
                OLLAMA_URL,
            )
    except Exception as exc:
        logger.warning(
            "Ollama embedding failed; provider=%s model=%s url=%s error=%s. "
            "Vector retrieval will fall back to text-only search.",
            EMBED_PROVIDER,
            EMBED_MODEL,
            OLLAMA_URL,
            exc,
            exc_info=True,
        )
    return None


def _embed_openai(text: str) -> Optional[list[float]]:
    """Generate embedding via OpenAI-compatible API.
    Works with: OpenAI, DeepSeek-compatible gateways, Voyage AI, Azure OpenAI,
    or any service exposing the OpenAI embeddings shape.
    Set EMBED_API_BASE to point to your provider."""
    if not OPENAI_API_KEY:
        logger.warning(
            "OpenAI-compatible embedding is configured but EMBED_API_KEY/DEEPSEEK_API_KEY/OPENAI_API_KEY is empty; "
            "provider=%s model=%s base=%s. Vector retrieval will fall back to text-only search.",
            EMBED_PROVIDER,
            EMBED_MODEL,
            EMBED_API_BASE,
        )
        return None
    try:
        url = _openai_embeddings_url()
        payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            items = data.get("data", [])
            if items and "embedding" in items[0]:
                return items[0]["embedding"]
            logger.warning(
                "OpenAI-compatible embedding returned an empty payload; provider=%s model=%s url=%s. "
                "Vector retrieval will fall back to text-only search.",
                EMBED_PROVIDER,
                EMBED_MODEL,
                url,
            )
    except Exception as exc:
        logger.warning(
            "OpenAI-compatible embedding failed; provider=%s model=%s url=%s error=%s. "
            "Vector retrieval will fall back to text-only search.",
            EMBED_PROVIDER,
            EMBED_MODEL,
            url,
            exc,
            exc_info=True,
        )
    return None


def _embed(text: str) -> Optional[list[float]]:
    """Generate embedding vector using configured provider.
    Returns None on failure (search falls back to FTS5 keyword only)."""
    try:
        if EMBED_PROVIDER == "openai":
            vec = _embed_openai(text)
        elif EMBED_PROVIDER == "ollama":
            vec = _embed_ollama(text)
        else:
            logger.warning(
                "Unknown embedding provider '%s'; model=%s. "
                "Vector retrieval will fall back to text-only search.",
                EMBED_PROVIDER,
                EMBED_MODEL,
            )
            return None
    except Exception as exc:
        logger.warning(
            "Embedding provider raised unexpectedly; provider=%s model=%s error=%s. "
            "Vector retrieval will fall back to text-only search.",
            EMBED_PROVIDER,
            EMBED_MODEL,
            exc,
            exc_info=True,
        )
        return None

    if not vec:
        logger.warning(
            "Embedding unavailable; provider=%s model=%s. "
            "Search is running in text-only fallback mode.",
            EMBED_PROVIDER,
            EMBED_MODEL,
        )
    return vec


def _search_mode(query_vec: Optional[list[float]]) -> str:
    return "vector" if query_vec else "fts5_fallback"


def _annotate_search_mode(results: list[dict], search_mode: str) -> list[dict]:
    for result in results:
        result["search_mode"] = search_mode
    return results


def _mark_vector_failed(diagnostics: Optional[dict]) -> None:
    if diagnostics is not None:
        diagnostics["vector_failed"] = True


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0  # Different embedding dimensions — incomparable
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recency_score(created_at: str) -> float:
    """Time decay score: more recent = higher (0-1). 30-day half-life."""
    try:
        t = datetime_strptime(created_at)
        days_ago = (now_local() - t).total_seconds() / 86400
        return math.exp(-days_ago / 30)
    except (ValueError, TypeError):
        return 0.5


def datetime_strptime(s: str):
    text = str(s or "").strip()
    for fmt, size in (("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d %H:%M", 16)):
        try:
            return datetime.strptime(text[:size], fmt).replace(tzinfo=LOCAL_TZ)
        except ValueError:
            continue
    raise ValueError(f"invalid database timestamp: {s!r}")


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_iso_bound(value: Optional[str], *, end_of_day: bool = False) -> Optional[str]:
    """Convert accepted ISO 8601 date/datetime bounds to DB timestamp text."""
    if value is None or value == "":
        return None
    raw = str(value).strip()
    if not raw:
        return None

    try:
        if _DATE_ONLY_RE.match(raw):
            parsed_date = datetime.strptime(raw, "%Y-%m-%d").date()
            parsed = datetime.combine(
                parsed_date,
                time.max.replace(microsecond=0) if end_of_day else time.min,
                tzinfo=LOCAL_TZ,
            )
        else:
            normalized = raw.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=LOCAL_TZ)
            else:
                parsed = parsed.astimezone(LOCAL_TZ)
    except ValueError as exc:
        raise ValueError(
            f"invalid ISO 8601 timestamp: {value!r}; expected YYYY-MM-DD or ISO datetime"
        ) from exc

    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_time_bounds(
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    return (
        _normalize_iso_bound(after, end_of_day=False),
        _normalize_iso_bound(before, end_of_day=True),
    )


# ─── Core API ────────────────────────────────────────────

def _clamp01(value: float, default: float) -> float:
    """Clamp a numeric score to the 0..1 range."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _decay_rate_for_category(category: str) -> float:
    rates = {
        "facts": 0.0,
        "core": 0.0,
        "core_profile": 0.0,
        "tasks": 0.02,
        "events": 0.05,
        "experience": 0.10,
        "general": 0.05,
    }
    return rates.get((category or "general").strip().lower(), 0.05)


def remember(content: str, category: str = "general", source: str = "cc",
             tags: Optional[list[str]] = None, importance: int = 5,
             valence: float = 0.5, arousal: float = 0.3,
             resolved: bool = True) -> str:
    """Store a memory with automatic dedup and conflict detection.
    - Exact duplicate content → skip
    - Semantic similarity ≥ 0.92 → skip (nearly identical)
    - Semantic similarity 0.85~0.92 → supersede: old memory marked historical, new one stored
    - Semantic similarity < 0.85 → new memory, stored directly
    """
    db = _get_db()

    existing = db.execute(
        "SELECT id FROM memories WHERE content = ?", (content,)
    ).fetchone()
    if existing:
        db.close()
        return f"Duplicate memory, skipped (existing id #{existing['id']})"

    # Generate embedding early (reused for semantic dedup + storage)
    vec = _embed(content)

    # Semantic dedup: check active memories in same category
    DUPLICATE_THRESHOLD = 0.92   # Nearly identical, skip
    SUPERSEDE_THRESHOLD = 0.85   # Similar but updated, supersede old
    supersede_ids = []

    if vec:
        cat_rows = db.execute(
            """SELECT m.id, m.content, v.embedding FROM memories m
               JOIN memory_vectors v ON m.id = v.memory_id
               WHERE m.category = ? AND m.superseded_by IS NULL""",
            (category,),
        ).fetchall()
        for r in cat_rows:
            existing_vec = _blob_to_vec(r["embedding"])
            sim = _cosine_similarity(vec, existing_vec)
            if sim >= DUPLICATE_THRESHOLD:
                db.close()
                return f"Semantically similar memory exists (ID {r['id']}, similarity {sim:.3f}). Use update_memory to update it."
            elif sim >= SUPERSEDE_THRESHOLD:
                supersede_ids.append((r["id"], r["content"][:40], sim))

    tags_json = json.dumps(tags or [], ensure_ascii=False)
    valence = _clamp01(valence, 0.5)
    arousal = _clamp01(arousal, 0.3)
    resolved_int = 1 if resolved else 0
    decay_rate = _decay_rate_for_category(category)
    now = now_str()

    cursor = db.execute(
        """INSERT INTO memories (
               content, category, source, tags, importance,
               valence, arousal, resolved, decay_rate, created_at
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            content, category, source, tags_json, importance,
            valence, arousal, resolved_int, decay_rate, now,
        ),
    )
    memory_id = cursor.lastrowid

    if vec:
        db.execute(
            "INSERT INTO memory_vectors (memory_id, embedding, model) VALUES (?, ?, ?)",
            (memory_id, _vec_to_blob(vec), EMBED_MODEL),
        )

    # Mark old memories as historical (not deleted, just superseded)
    supersede_notes = []
    for old_id, old_preview, sim in supersede_ids:
        db.execute(
            "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE id = ?",
            (memory_id, now, old_id),
        )
        supersede_notes.append(f"  ↳ Superseded #{old_id} ({old_preview}… sim {sim:.3f})")

    db.commit()
    db.close()
    _rebuild_index()

    result = (
        f"Remembered #{memory_id} [{category}]: {content[:50]}...\n"
        f"Created: {now}\n"
        f"Tags: {tags_json}"
    )
    if supersede_notes:
        result += "\n" + "\n".join(supersede_notes)
    return result


def _normalize_tags_input(tags) -> Optional[list[str]]:
    if tags is None:
        return None
    if isinstance(tags, str):
        raw = tags.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                tags = parsed
            else:
                tags = raw.split(",")
        except json.JSONDecodeError:
            tags = raw.split(",")
    normalized = []
    seen = set()
    for tag in tags:
        item = str(tag).strip()
        if item and item not in seen:
            normalized.append(item)
            seen.add(item)
    return normalized


def _sync_memory_tags(db: sqlite3.Connection, memory_id: int, tags: list[str]) -> None:
    db.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
    for tag in tags:
        db.execute(
            "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
            (memory_id, tag),
        )


def forget(keyword: str) -> str:
    """Delete memories containing keyword."""
    db = _get_db()
    rows = db.execute(
        "SELECT id, content FROM memories WHERE content LIKE ?",
        (f"%{keyword}%",),
    ).fetchall()

    if not rows:
        db.close()
        return f"No memories found containing '{keyword}'"

    for row in rows:
        db.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (row["id"],))
        db.execute("DELETE FROM memories WHERE id = ?", (row["id"],))

    db.commit()
    db.close()
    _rebuild_index()
    return f"Deleted {len(rows)} memories containing '{keyword}'"


def delete_memory(memory_id: int) -> dict:
    """Delete a single memory by ID."""
    db = _get_db()
    row = db.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}

    db.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))
    db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    db.close()
    _rebuild_index()
    return {"ok": True}


def update_memory(
    memory_id: int,
    content: str = "",
    category: str = "",
    importance: int = 0,
    resolved: int = -1,
    tags=None,
) -> dict:
    """Update a single memory by ID. Only non-empty/non-zero fields are changed."""
    db = _get_db()
    row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}

    tags_list = _normalize_tags_input(tags)
    has_content = bool(content and content.strip())
    has_category = bool(category and category.strip())
    has_importance = importance > 0
    has_resolved = resolved in (0, 1)
    has_tags = tags_list is not None
    if not any((has_content, has_category, has_importance, has_resolved, has_tags)):
        db.close()
        return {"ok": False, "error": "No update fields provided"}

    new_content = content.strip() if has_content else row["content"]
    new_category = category.strip() if has_category else row["category"]
    new_importance = importance if has_importance else row["importance"]
    new_resolved = resolved if has_resolved else row["resolved"]
    new_tags_json = (
        json.dumps(tags_list, ensure_ascii=False)
        if has_tags
        else row["tags"]
    )
    new_decay_rate = (
        _decay_rate_for_category(new_category)
        if new_category != row["category"]
        else row["decay_rate"]
    )
    updated_at = now_str()

    db.execute(
        """UPDATE memories
           SET content = ?, category = ?, tags = ?, importance = ?,
               resolved = ?, decay_rate = ?, updated_at = ?
           WHERE id = ?""",
        (
            new_content, new_category, new_tags_json, new_importance,
            new_resolved, new_decay_rate, updated_at, memory_id,
        ),
    )
    if has_tags:
        _sync_memory_tags(db, memory_id, tags_list)

    # Only refresh embedding if content changed
    vec_refreshed = False
    if new_content != row["content"]:
        db.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))
        vec = _embed(new_content)
        if vec:
            db.execute(
                "INSERT INTO memory_vectors (memory_id, embedding, model) VALUES (?, ?, ?)",
                (memory_id, _vec_to_blob(vec), EMBED_MODEL),
            )
            vec_refreshed = True
    embedding_status = "unchanged"
    if new_content != row["content"]:
        embedding_status = "refreshed" if vec_refreshed else "pending_reindex"

    db.commit()
    updated = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    db.close()
    _rebuild_index()
    return {
        "ok": True,
        "id": memory_id,
        "updated_at": updated_at,
        "memory": dict(updated) if updated else None,
        "embedding_refreshed": vec_refreshed,
        "embedding_status": embedding_status,
    }


def search(query: str, limit: int = 10, category: Optional[str] = None) -> list[dict]:
    """Hybrid search: vector semantic + FTS5 keyword + time decay."""
    db = _get_db()
    results = {}

    # 1. FTS5 keyword search
    try:
        fts_query = build_fts_match_query(query)
        if fts_query:
            cat_filter = "AND m.category = ?" if category else ""
            params = [fts_query, category] if category else [fts_query]
            fts_rows = db.execute(f"""
                SELECT m.id, m.content, m.category, m.source, m.importance,
                       m.created_at, m.recalled_count, rank
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.id
                WHERE memories_fts MATCH ? AND m.superseded_by IS NULL {cat_filter}
                ORDER BY rank LIMIT {limit * 2}
            """, params).fetchall()

            if fts_rows:
                max_rank = max(abs(r["rank"]) for r in fts_rows) or 1
                for r in fts_rows:
                    mid = r["id"]
                    fts_score = abs(r["rank"]) / max_rank
                    results[mid] = {
                        "id": mid, "content": r["content"], "category": r["category"],
                        "source": r["source"], "importance": r["importance"],
                        "created_at": r["created_at"], "recalled_count": r["recalled_count"],
                        "fts_score": fts_score, "vec_score": 0.0,
                    }
    except Exception as exc:
        logger.warning(
            "[FTS5Search] Failed: query=%r fts_query=%r error=%s. Continuing with vector/empty results.",
            query,
            locals().get("fts_query", ""),
            exc,
            exc_info=True,
        )

    # 2. Vector semantic search
    query_vec = _embed(query)
    search_mode = _search_mode(query_vec)
    if not query_vec:
        logger.warning(
            "[VectorSearch] Failed: embedding unavailable for provider=%s model=%s. "
            "Falling back to FTS5 text retrieval.",
            EMBED_PROVIDER,
            EMBED_MODEL,
        )
    if query_vec:
        try:
            cat_filter = "AND m.category = ?" if category else ""
            params = [category] if category else []
            vec_rows = db.execute(f"""
                SELECT m.id, m.content, m.category, m.source, m.importance,
                       m.created_at, m.recalled_count, v.embedding
                FROM memories m
                JOIN memory_vectors v ON m.id = v.memory_id
                WHERE m.superseded_by IS NULL {cat_filter}
            """, params).fetchall()

            scored = []
            for r in vec_rows:
                mem_vec = _blob_to_vec(r["embedding"])
                sim = _cosine_similarity(query_vec, mem_vec)
                scored.append((r, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            for r, sim in scored[:limit * 2]:
                mid = r["id"]
                if mid in results:
                    results[mid]["vec_score"] = sim
                else:
                    results[mid] = {
                        "id": mid, "content": r["content"], "category": r["category"],
                        "source": r["source"], "importance": r["importance"],
                        "created_at": r["created_at"], "recalled_count": r["recalled_count"],
                        "fts_score": 0.0, "vec_score": sim,
                    }
        except Exception as exc:
            search_mode = "fts5_fallback"
            logger.warning(
                "[VectorSearch] Failed: vector memory channel error=%s. Falling back to FTS5 text retrieval.",
                exc,
                exc_info=True,
            )

    # 3. Combined scoring
    for mid, info in results.items():
        recency = _recency_score(info["created_at"])
        # recalled_count as tiny tiebreaker (max 0.05, prevents snowball)
        recall_bonus = min(0.05, 0.01 * math.log1p(info.get("recalled_count", 0)))
        info["final_score"] = (
            WEIGHT_VECTOR * info["vec_score"]
            + WEIGHT_FTS * info["fts_score"]
            + WEIGHT_RECENCY * recency
            + recall_bonus
        )

    MIN_SCORE = 0.40
    ranked = [r for r in results.values() if r["final_score"] >= MIN_SCORE]
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    ranked = ranked[:limit]

    for r in ranked:
        if "id" in r:
            db.execute(
                "UPDATE memories SET recalled_count = recalled_count + 1 WHERE id = ?",
                (r["id"],),
            )
    db.commit()
    db.close()

    bank_results = _search_bank(query_vec, query, limit=5)
    ranked.extend(bank_results)
    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    return _annotate_search_mode(ranked[:limit], search_mode)


def search_text(query: str, limit: int = 10) -> str:
    """Search and return formatted text. Adds staleness warning for old memories."""
    results = search(query, limit)
    if not results:
        return "No matching memories found"
    lines = []
    now = now_local()
    for r in results:
        score = f"{r['final_score']:.2f}"
        created = r.get('created_at', '')
        line = f"[{r['category']}|{r['source']}|{created}] (relevance:{score}) {r['content'][:1000]}"
        # Staleness warning for old memories
        if created:
            try:
                from datetime import datetime
                created_dt = datetime.strptime(created[:10], "%Y-%m-%d")
                days_old = (now.replace(tzinfo=None) - created_dt).days
                if days_old > 14:
                    line += f"\n  ⚠ {days_old}天前的记忆，涉及代码/配置/状态请先验证再使用"
            except (ValueError, TypeError):
                pass
        lines.append(line)
    return "\n".join(lines)


def get_all(category: Optional[str] = None, limit: int = 50, after: Optional[str] = None, before: Optional[str] = None) -> list[dict]:
    """Get all active memories (by time desc). Excludes superseded memories.
    after: ISO date string, only memories created on or after this date (e.g. '2026-04-01').
    before: ISO date string, only memories created on or before this date."""
    after_bound, before_bound = _normalize_time_bounds(after=after, before=before)
    db = _get_db()
    filters = []
    params: list = []
    if category:
        filters.append("AND category = ?")
        params.append(category)
    if after_bound:
        filters.append("AND created_at >= ?")
        params.append(after_bound)
    if before_bound:
        filters.append("AND created_at <= ?")
        params.append(before_bound)
    filter_sql = " ".join(filters)
    rows = db.execute(
        f"SELECT * FROM memories WHERE superseded_by IS NULL {filter_sql} ORDER BY created_at DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


# ─── Daily Log ───────────────────────────────────────────

def daily_log(text: str) -> str:
    """Append to today's daily log."""
    today = now_local().strftime("%Y-%m-%d")
    DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DAILY_LOG_DIR / f"{today}.md"

    now_time = now_local().strftime("%H:%M")
    entry = f"- [{now_time}] {text}\n"

    needs_header = not log_file.exists() or log_file.stat().st_size == 0
    with open(log_file, "a", encoding="utf-8") as f:
        if needs_header:
            f.write(f"# {today} Log\n\n")
        f.write(entry)

    db = _get_db()
    existing = db.execute("SELECT content FROM daily_logs WHERE date = ?", (today,)).fetchone()
    if existing:
        new_content = existing["content"] + entry
        db.execute("UPDATE daily_logs SET content = ? WHERE date = ?", (new_content, today))
    else:
        db.execute("INSERT INTO daily_logs (date, content) VALUES (?, ?)", (today, entry))
    db.commit()
    db.close()

    return f"Logged to {today}"


# ─── Notification Dedup ──────────────────────────────────

def was_notified(content_key: str, hours: int = 24) -> bool:
    """Check if already notified in the past N hours."""
    db = _get_db()
    cutoff = (now_local() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M")
    row = db.execute(
        "SELECT 1 FROM notifications WHERE content LIKE ? AND created_at > ? LIMIT 1",
        (f"%{content_key}%", cutoff),
    ).fetchone()
    db.close()
    return row is not None


def record_notification(content: str):
    """Record a sent notification."""
    db = _get_db()
    db.execute(
        "INSERT INTO notifications (content, created_at) VALUES (?, ?)",
        (content, now_str()),
    )
    db.commit()
    db.close()


# ─── Memory Health Tools ────────────────────────────────

def find_duplicates(threshold: float = 0.85) -> list[dict]:
    """Find memory pairs with cosine similarity above threshold. Read-only."""
    db = _get_db()
    rows = db.execute("""
        SELECT m.id, m.content, m.category, v.embedding
        FROM memories m
        JOIN memory_vectors v ON m.id = v.memory_id
    """).fetchall()
    db.close()

    pairs = []
    for i in range(len(rows)):
        vec_i = _blob_to_vec(rows[i]["embedding"])
        for j in range(i + 1, len(rows)):
            vec_j = _blob_to_vec(rows[j]["embedding"])
            sim = _cosine_similarity(vec_i, vec_j)
            if sim >= threshold:
                pairs.append({
                    "id_a": rows[i]["id"],
                    "content_a": rows[i]["content"][:100],
                    "category_a": rows[i]["category"],
                    "id_b": rows[j]["id"],
                    "content_b": rows[j]["content"][:100],
                    "category_b": rows[j]["category"],
                    "similarity": round(sim, 4),
                })
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs


def _format_reindex_report(result: dict) -> str:
    """Format reindex status for MCP users and logs."""
    lines = [
        f"memory_reindex completed: {result['status']}",
        f"Database: {result['database']}",
        f"Provider: {result['provider']}, model: {result['model']}",
        f"Started: {result['started_at']}",
        f"Finished: {result['finished_at']}",
    ]
    for item in result["targets"]:
        if item["target"] == "memory_vectors":
            lines.append(
                "- memory_vectors: {status}, rebuilt {rebuilt}/{total} memories, "
                "{failed} failed".format(**item)
            )
        elif item["target"] in {"memories_fts", "conversation_log_fts"}:
            lines.append(
                f"- {item['target']}: {item['status']}, rebuilt {item['rebuilt']} rows"
            )
        elif item["target"] == "bank_chunks":
            lines.append(
                "- bank_chunks: {status}, cleared {cleared} rows, indexed {files_indexed} "
                "files, wrote {chunks_written} chunks, skipped {files_skipped} files".format(**item)
            )
        else:
            lines.append(f"- {item['target']}: {item['status']}")
        if item.get("error"):
            lines.append(f"  error: {item['error']}")
    return "\n".join(lines)


def _rebuild_memory_vectors(db: sqlite3.Connection) -> dict:
    rows = db.execute("SELECT id, content FROM memories").fetchall()
    total = len(rows)
    rebuilt = 0
    failed = 0

    for r in rows:
        vec = _embed(r["content"])
        db.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (r["id"],))
        if vec:
            db.execute(
                "INSERT INTO memory_vectors (memory_id, embedding, model) VALUES (?, ?, ?)",
                (r["id"], _vec_to_blob(vec), EMBED_MODEL),
            )
            rebuilt += 1
        else:
            failed += 1

    return {
        "target": "memory_vectors",
        "status": "ok",
        "total": total,
        "rebuilt": rebuilt,
        "failed": failed,
    }


def _rebuild_fts_table(
    db: sqlite3.Connection,
    *,
    target: str,
    source_table: str,
    columns: tuple[str, ...],
) -> dict:
    column_sql = ", ".join(columns)
    db.execute(f"DROP TABLE IF EXISTS {target}")
    db.execute(
        f"""CREATE VIRTUAL TABLE {target}
            USING fts5({column_sql}, content={source_table}, content_rowid=id)"""
    )
    rows = db.execute(
        f"SELECT id, {column_sql} FROM {source_table} ORDER BY id"
    ).fetchall()
    for row in rows:
        values = [segment_cjk(row[columns[0]] or "")]
        values.extend(row[col] or "" for col in columns[1:])
        placeholders = ", ".join("?" for _ in columns)
        db.execute(
            f"INSERT INTO {target}(rowid, {column_sql}) VALUES (?, {placeholders})",
            (row["id"], *values),
        )
    return {
        "target": target,
        "status": "ok",
        "rebuilt": len(rows),
    }


def _rebuild_bank_chunks(db: sqlite3.Connection) -> dict:
    cleared = db.execute("SELECT COUNT(*) AS c FROM bank_chunks").fetchone()["c"]
    db.execute("DELETE FROM bank_chunks")

    files_indexed = 0
    files_skipped = 0
    chunks_written = 0

    if not BANK_DIR.exists():
        return {
            "target": "bank_chunks",
            "status": "ok",
            "cleared": cleared,
            "files_indexed": 0,
            "files_skipped": 0,
            "chunks_written": 0,
        }

    for md_file in BANK_DIR.glob("*.md"):
        if md_file.name in _BANK_EXCLUDE:
            files_skipped += 1
            continue
        md_file = md_file.resolve()
        mtime = md_file.stat().st_mtime
        text = md_file.read_text(encoding="utf-8")
        chunks = _split_into_chunks(text)
        wrote_for_file = 0

        for chunk in chunks:
            cleaned_chunk = _clean_bank_chunk(chunk)
            if not cleaned_chunk or len(cleaned_chunk) < 10:
                continue
            vec = _embed(cleaned_chunk)
            blob = _vec_to_blob(vec) if vec else None
            db.execute(
                """INSERT INTO bank_chunks
                   (file_path, chunk_text, embedding, file_mtime, index_version)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(md_file), cleaned_chunk, blob, mtime, BANK_INDEX_VERSION),
            )
            chunks_written += 1
            wrote_for_file += 1

        if wrote_for_file:
            files_indexed += 1
        else:
            files_skipped += 1

    return {
        "target": "bank_chunks",
        "status": "ok",
        "cleared": cleared,
        "files_indexed": files_indexed,
        "files_skipped": files_skipped,
        "chunks_written": chunks_written,
    }


def reindex_all() -> dict:
    """Rebuild derived retrieval indexes and return structured status."""
    started = now_str()
    db = _get_db()
    targets = []

    rebuild_steps = [
        ("memory_vectors", lambda: _rebuild_memory_vectors(db)),
        (
            "memories_fts",
            lambda: _rebuild_fts_table(
                db,
                target="memories_fts",
                source_table="memories",
                columns=("content", "category", "tags"),
            ),
        ),
        (
            "conversation_log_fts",
            lambda: _rebuild_fts_table(
                db,
                target="conversation_log_fts",
                source_table="conversation_log",
                columns=("content", "platform", "speaker"),
            ),
        ),
        ("bank_chunks", lambda: _rebuild_bank_chunks(db)),
    ]

    try:
        for target, step in rebuild_steps:
            try:
                targets.append(step())
                db.commit()
            except Exception as exc:
                db.rollback()
                targets.append({
                    "target": target,
                    "status": "error",
                    "error": str(exc),
                })
    finally:
        db.close()

    status = "success" if all(t["status"] == "ok" for t in targets) else "partial_failure"
    return {
        "status": status,
        "database": str(DB_PATH),
        "provider": EMBED_PROVIDER,
        "model": EMBED_MODEL,
        "started_at": started,
        "finished_at": now_str(),
        "targets": targets,
    }


def reindex_embeddings() -> str:
    """Rebuild all recoverable retrieval indexes.
    Useful after switching embedding providers or repairing FTS/bank indexes."""
    return _format_reindex_report(reindex_all())


def find_stale(days: int = 14) -> list[dict]:
    """Find potentially stale memories: older than N days, importance < 7, recalled < 3. Read-only."""
    db = _get_db()
    cutoff = (now_local() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
    rows = db.execute("""
        SELECT id, content, category, importance, recalled_count, created_at
        FROM memories
        WHERE created_at < ? AND importance < 7 AND recalled_count < 3
            AND superseded_by IS NULL
        ORDER BY created_at ASC
    """, (cutoff,)).fetchall()
    db.close()
    return [dict(r) for r in rows]


def calculate_memory_score(memory: dict) -> float:
    """Calculate memory activity score using an Ebbinghaus-style forgetting curve.

    Score = importance x (activation ^ 0.3) x e^(-lambda x days) x emotion_weight
    """
    importance = max(memory.get("importance", 5), 1) / 10.0
    activation = max(memory.get("recalled_count", 0) + 1, 1)

    ref = memory.get("last_accessed_at") or memory.get("created_at", "")
    days = _days_since(ref, default=30)
    raw_decay_rate = memory.get("decay_rate", 0.05)
    decay_rate = 0.05 if raw_decay_rate is None else float(raw_decay_rate)

    if decay_rate == 0:
        time_decay = 1.0
    else:
        time_decay = math.exp(-decay_rate * days)

    arousal = float(memory.get("arousal", 0.3) or 0.3)
    base_emotion = 0.5
    arousal_boost = 1.0
    emotion_weight = base_emotion + arousal * arousal_boost

    resolved = bool(memory.get("resolved", 1))
    resolved_penalty = 1.0 if not resolved else 0.05
    if arousal <= 0.7:
        resolved_penalty = 1.0

    score = importance * (activation ** 0.3) * time_decay * emotion_weight * resolved_penalty
    return round(score, 4)


def decay_memories(days: int = 30, dry_run: bool = True, threshold: float = 0.3) -> dict:
    """Decay memories using the emotional forgetting curve.

    Computes an activity score for each dynamic memory:
    - Score >= threshold: keep active
    - Score < threshold: archive by setting importance=0 and superseded_by=-1
    - pinned memories and decay_rate=0 memories are skipped

    dry_run=True: preview only. dry_run=False: apply archive changes.
    """
    db = _get_db()
    now = now_str()

    rows = db.execute("""
        SELECT id, content, category, importance, recalled_count,
               created_at, last_accessed_at, valence, arousal, resolved,
               decay_rate, pinned
        FROM memories
        WHERE COALESCE(pinned, 0) = 0
            AND COALESCE(decay_rate, 0.05) > 0
            AND importance > 0
            AND superseded_by IS NULL
        ORDER BY created_at ASC
    """).fetchall()

    archived: list[dict] = []
    for r in rows:
        memory = dict(r)
        score = calculate_memory_score(memory)
        if score >= threshold:
            continue

        entry = {"id": r["id"], "category": r["category"],
                 "content": r["content"][:100],
                 "importance": f"{r['importance']} -> 0",
                 "score": score}
        archived.append(entry)
        if not dry_run:
            db.execute(
                "UPDATE memories SET importance = 0, superseded_by = -1, updated_at = ? WHERE id = ?",
                (now, r["id"]),
            )

    if not dry_run:
        db.commit()
    db.close()

    if not dry_run:
        _rebuild_index()

    return {
        "dry_run": dry_run,
        "scanned": len(rows),
        "threshold": threshold,
        "decayed": 0,
        "archived": len(archived),
        "details_decayed": [],
        "details_archived": archived[:20],
    }


def decay(days: int = 30, dry_run: bool = True, threshold: float = 0.3) -> dict:
    """Backward-compatible wrapper for the Phase 3 emotional decay engine."""
    return decay_memories(days=days, dry_run=dry_run, threshold=threshold)


def get_surfacing_memories(arousal_threshold: float = 0.7, limit: int = 3) -> list[dict]:
    """Get unresolved high-arousal memories that should be proactively surfaced."""
    db = _get_db()
    rows = db.execute("""
        SELECT id, content, category, arousal, valence, created_at
        FROM memories
        WHERE resolved = 0
            AND arousal > ?
            AND importance > 0
            AND COALESCE(pinned, 0) = 0
            AND superseded_by IS NULL
        ORDER BY arousal DESC, created_at DESC
        LIMIT ?
    """, (arousal_threshold, limit)).fetchall()
    db.close()
    return [dict(r) for r in rows]


# ─── Memory Context ──────────────────────────────────────

def get_context(query: Optional[str] = None, max_chars: int = 3000) -> str:
    """Generate memory context summary."""
    if query:
        return search_text(query, limit=10)

    db = _get_db()
    rows = db.execute("""
        SELECT content, category, source, created_at, importance
        FROM memories
        ORDER BY
            CASE WHEN importance >= 7 THEN 0 ELSE 1 END,
            created_at DESC
        LIMIT 20
    """).fetchall()
    db.close()

    if not rows:
        return "(No memories yet)"

    lines = ["# Memory Summary\n"]
    total = 0
    for r in rows:
        line = f"- [{r['category']}] {r['content']}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines)


# ─── Bank File Index ─────────────────────────────────────

def _clean_bank_chunk(chunk: str) -> Optional[str]:
    """Remove template comments from a bank chunk."""
    cleaned_lines = []
    substantive_lines = []
    in_comment = False

    for line in chunk.split("\n"):
        stripped = line.strip()
        if in_comment:
            if "-->" in stripped:
                in_comment = False
            continue
        if stripped.startswith("<!--"):
            if "-->" not in stripped:
                in_comment = True
            continue

        cleaned_lines.append(line.rstrip())
        if stripped and not stripped.startswith("#"):
            substantive_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned or not substantive_lines:
        return None
    return cleaned


_BANK_EXCLUDE = set(
    f.strip() for f in os.environ.get("IMPRINT_BANK_EXCLUDE", "").split(",") if f.strip()
)

def _index_bank_files():
    """Index markdown files in bank/ directory. Skip unchanged files."""
    if not BANK_DIR.exists():
        return
    db = _get_db()
    for md_file in BANK_DIR.glob("*.md"):
        if md_file.name in _BANK_EXCLUDE:
            continue
        md_file = md_file.resolve()
        mtime = md_file.stat().st_mtime
        existing = db.execute(
            "SELECT file_mtime, index_version FROM bank_chunks WHERE file_path = ? LIMIT 1",
            (str(md_file),),
        ).fetchone()
        if (
            existing
            and abs(existing["file_mtime"] - mtime) < 1
            and existing["index_version"] == BANK_INDEX_VERSION
        ):
            continue

        db.execute("DELETE FROM bank_chunks WHERE file_path = ?", (str(md_file),))

        text = md_file.read_text(encoding="utf-8")
        chunks = _split_into_chunks(text)

        for chunk in chunks:
            cleaned_chunk = _clean_bank_chunk(chunk)
            if not cleaned_chunk or len(cleaned_chunk) < 10:
                continue
            vec = _embed(cleaned_chunk)
            blob = _vec_to_blob(vec) if vec else None
            db.execute(
                """INSERT INTO bank_chunks
                   (file_path, chunk_text, embedding, file_mtime, index_version)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(md_file), cleaned_chunk, blob, mtime, BANK_INDEX_VERSION),
            )
    db.commit()
    db.close()


def _split_into_chunks(text: str) -> list[str]:
    """Split by markdown ## headings."""
    chunks = []
    current = []
    for line in text.split("\n"):
        if line.startswith("## ") and current:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _search_bank(query_vec, query_text: str, limit: int = 5) -> list[dict]:
    """Search bank/ file chunks."""
    _index_bank_files()
    db = _get_db()
    results = []

    if query_vec:
        rows = db.execute(
            "SELECT chunk_text, file_path, embedding FROM bank_chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        for r in rows:
            vec = _blob_to_vec(r["embedding"])
            sim = _cosine_similarity(query_vec, vec)
            if sim > 0.3:
                results.append({
                    "content": r["chunk_text"],
                    "source": Path(r["file_path"]).stem,
                    "category": "bank",
                    "final_score": sim,
                })

    # Keyword search — score no longer hardcoded, merges with vector results
    KEYWORD_BASE = 0.5
    KEYWORD_BONUS = 0.15
    DUAL_HIT_BONUS = 0.1
    query_lower = query_text.lower()
    rows = db.execute("SELECT chunk_text, file_path FROM bank_chunks").fetchall()
    for r in rows:
        if query_lower in r["chunk_text"].lower():
            kw_score = KEYWORD_BASE + KEYWORD_BONUS  # 0.65
            existing = next((x for x in results if x["content"] == r["chunk_text"]), None)
            if existing:
                existing["final_score"] = max(existing["final_score"], kw_score) + DUAL_HIT_BONUS
            else:
                results.append({
                    "content": r["chunk_text"],
                    "source": Path(r["file_path"]).stem,
                    "category": "bank",
                    "final_score": kw_score,
                })

    db.close()
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:limit]


# ─── MEMORY.md Index Rebuild ─────────────────────────────

def _summarize_for_index(content, max_len=50):
    """Truncate memory content to a short index pointer."""
    text = content.strip()
    for sep in ("：", "——", "—", "。", "，", "；"):
        idx = text.find(sep)
        if 0 < idx <= max_len:
            return text[:idx]
    for sep in (":", ", "):
        idx = text.find(sep)
        if 10 < idx <= max_len:
            return text[:idx]
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def _rebuild_index():
    """Rebuild MEMORY.md as a lightweight index (date + keyword per line).
    Full content is available via memory_search."""
    db = _get_db()
    lines = ["# Memory Index\n", f"*Last updated: {now_str()}*\n"]

    total = db.execute("SELECT COUNT(*) as c FROM memories WHERE superseded_by IS NULL").fetchone()["c"]
    lines.append(f"*{total} memories — use memory_search for details*\n")

    categories = db.execute(
        "SELECT DISTINCT category FROM memories WHERE superseded_by IS NULL ORDER BY category"
    ).fetchall()

    for cat_row in categories:
        cat = cat_row["category"]
        rows = db.execute(
            """SELECT content, source, created_at, importance
               FROM memories WHERE category = ? AND superseded_by IS NULL
               ORDER BY importance DESC, created_at DESC""",
            (cat,),
        ).fetchall()
        if not rows:
            continue

        section = [f"\n## {cat}"]
        for r in rows:
            date = r["created_at"][:10] if r["created_at"] else ""
            short_date = date[5:].replace("-", "/") if date else ""
            summary = _summarize_for_index(r["content"])
            section.append(f"- [{short_date}] {summary}")

        lines.extend(section)

    db.close()

    with open(MEMORY_INDEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════
# RRF Unified Retrieval — fusion across memory, bank, conversation
# ═══════════════════════════════════════════════════════════════

RRF_K = 60              # RRF ranking constant (standard value)
VEC_PRE_FILTER = 0.3    # Vector similarity pre-filter threshold
MIN_FINAL_SCORE = 0.003 # Drop results below this after reranking
RERANK_BLEND = 0.3      # How much rerank factors affect final score
LIKE_LIMIT = 50         # Max results from LIKE exact-match channel per pool


def _days_since(time_str: str, default: float = 30.0) -> float:
    """Days elapsed since a timestamp string."""
    if not time_str:
        return default
    try:
        t = datetime.strptime(time_str[:16], "%Y-%m-%d %H:%M").replace(tzinfo=LOCAL_TZ)
        return max(0.0, (now_local() - t).total_seconds() / 86400)
    except (ValueError, TypeError):
        return default


def _sanitize_fts(query: str) -> str:
    """Build a sanitized, recall-friendly FTS5 MATCH expression."""
    return build_fts_match_query(query)


def _rank_like_rows(
    rows,
    *,
    key_prefix: str,
    content_field: str,
    details_factory,
) -> tuple[list[tuple[str, int]], dict[str, dict]]:
    """Rank LIKE fallback rows by keyword coverage instead of whole-query match."""
    scored = []
    for r in rows:
        matched = int(r["matched_terms"])
        if matched <= 0:
            continue
        scored.append((r, matched, len((r[content_field] or ""))))

    scored.sort(key=lambda item: (-item[1], item[2]))

    ranking: list[tuple[str, int]] = []
    details: dict[str, dict] = {}
    for idx, (r, _, _) in enumerate(scored[:LIKE_LIMIT]):
        key = f"{key_prefix}_{r['id']}"
        ranking.append((key, idx + 1))
        details[key] = details_factory(r)
    return ranking, details


# ─── RRF Core ───────────────────────────────────────────

def _rrf_fuse(channel_rankings: list[list[tuple[str, int]]]) -> dict[str, float]:
    """Reciprocal Rank Fusion over N ranked lists."""
    scores: dict[str, float] = {}
    for ranking in channel_rankings:
        for key, rank in ranking:
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    return scores


_RANK_BASELINE = 10

def _inject_default_ranks(
    fts_ranking: list[tuple[str, int]],
    vec_ranking: list[tuple[str, int]],
) -> None:
    """Give absent paired channel a default low rank so single-channel
    results aren't unfairly penalised. Mirrors rankings when one channel
    is completely empty (e.g. FTS can't tokenize the query)."""
    if not fts_ranking and not vec_ranking:
        return
    if not fts_ranking and vec_ranking:
        fts_ranking.extend(vec_ranking)
        return
    if not vec_ranking and fts_ranking:
        vec_ranking.extend(fts_ranking)
        return

    fts_keys = {k for k, _ in fts_ranking}
    vec_keys = {k for k, _ in vec_ranking}

    default_fts = max(len(fts_ranking), _RANK_BASELINE) + 1
    default_vec = max(len(vec_ranking), _RANK_BASELINE) + 1

    for k in vec_keys - fts_keys:
        fts_ranking.append((k, default_fts))
    for k in fts_keys - vec_keys:
        vec_ranking.append((k, default_vec))


# ─── Rerank Functions ───────────────────────────────────

def _rerank_memory(rrf_score: float, row: dict) -> float:
    """Memory rerank: time x activation x importance x emotion, blended with RRF."""
    if row.get("pinned"):
        return rrf_score

    importance = max(row.get("importance", 5), 1)
    recalled = row.get("recalled_count", 0)
    arousal = _clamp01(row.get("arousal", 0.3), 0.3)
    resolved = bool(row.get("resolved", 1))
    decay_rate = max(float(row.get("decay_rate", 0.05) or 0.0), 0.0)

    ref = row.get("last_accessed_at") or row.get("created_at", "")
    days = _days_since(ref, default=30)
    lam = decay_rate / (importance / 5) if decay_rate > 0 else 0.0
    time_factor = 0.4 + 0.6 * math.exp(-lam * days)

    activation_factor = 0.8 + 0.2 * (math.log(recalled + 1) / math.log(51))
    importance_factor = 0.7 + 0.3 * (importance / 10)
    emotion_factor = 0.9 + 0.2 * arousal
    if not resolved and arousal > 0.7:
        emotion_factor *= 1.5

    factor = time_factor * activation_factor * importance_factor * emotion_factor
    return rrf_score * (1 - RERANK_BLEND + RERANK_BLEND * factor)


def _rerank_bank(rrf_score: float, row: dict) -> float:
    """Bank rerank: gentle file freshness (tiebreaker only)."""
    mtime = row.get("file_mtime")
    if mtime is not None:
        try:
            dt = datetime.fromtimestamp(float(mtime), tz=LOCAL_TZ)
            days = max(0.0, (now_local() - dt).total_seconds() / 86400)
        except (ValueError, TypeError, OSError):
            days = 7.0
    else:
        days = 7.0
    freshness = 0.90 + 0.10 * math.exp(-days / 90)
    return rrf_score * (1 - RERANK_BLEND + RERANK_BLEND * freshness)


def _rerank_conv(rrf_score: float, row: dict) -> float:
    """Conversation rerank: recency (7-day half-life)."""
    days = _days_since(row.get("created_at", ""), default=30)
    recency = 0.3 + 0.7 * math.exp(-days / 7)
    return rrf_score * (1 - RERANK_BLEND + RERANK_BLEND * recency)


# ─── Per-Pool Channel Search ────────────────────────────

def _search_memory_channels(query, query_vec, db, *, category=None, limit=50, diagnostics=None):
    """Return (fts_ranking, vec_ranking, like_ranking, details) for memory pool."""
    details = {}
    fts_ranking = []
    vec_ranking = []

    safe_q = _sanitize_fts(query)
    if safe_q:
        try:
            cat_sql = "AND m.category = ?" if category else ""
            params = [safe_q] + ([category] if category else []) + [limit]
            fts_rows = db.execute(
                f"""SELECT m.id, m.content, m.category, m.source, m.importance,
                           m.valence, m.arousal, m.resolved, m.decay_rate,
                           m.created_at, m.recalled_count,
                           m.last_accessed_at, m.pinned
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.id
                    WHERE memories_fts MATCH ? AND m.superseded_by IS NULL {cat_sql}
                    ORDER BY f.rank
                    LIMIT ?""",
                params,
            ).fetchall()
            for idx, r in enumerate(fts_rows):
                key = f"mem_{r['id']}"
                fts_ranking.append((key, idx + 1))
                details[key] = dict(r)
        except Exception as exc:
            logger.warning(
                "Memory FTS search failed; query=%r fts_query=%r error=%s",
                query,
                safe_q,
                exc,
                exc_info=True,
            )

    if query_vec:
        try:
            cat_sql = "AND m.category = ?" if category else ""
            params = [category] if category else []
            vec_rows = db.execute(
                f"""SELECT m.id, m.content, m.category, m.source, m.importance,
                           m.valence, m.arousal, m.resolved, m.decay_rate,
                           m.created_at, m.recalled_count,
                           m.last_accessed_at, m.pinned,
                           v.embedding
                    FROM memories m
                    JOIN memory_vectors v ON m.id = v.memory_id
                    WHERE m.superseded_by IS NULL {cat_sql}""",
                params,
            ).fetchall()

            scored = []
            for r in vec_rows:
                sim = _cosine_similarity(query_vec, _blob_to_vec(r["embedding"]))
                if sim >= VEC_PRE_FILTER:
                    scored.append((r, sim))
            scored.sort(key=lambda x: x[1], reverse=True)

            for idx, (r, sim) in enumerate(scored[:limit]):
                key = f"mem_{r['id']}"
                vec_ranking.append((key, idx + 1))
                if key not in details:
                    details[key] = dict(r)
                details[key]["vec_similarity"] = sim
        except Exception as exc:
            _mark_vector_failed(diagnostics)
            logger.warning(
                "[VectorSearch] Failed: memory vector channel query=%r error=%s. "
                "Falling back to FTS5/LIKE text retrieval.",
                query,
                exc,
                exc_info=True,
            )

    like_ranking = []
    like_terms = like_search_terms(query)
    if like_terms:
        cat_sql = "AND category = ?" if category else ""
        where_sql = " OR ".join("LOWER(content) LIKE ?" for _ in like_terms)
        score_sql = " + ".join(
            "CASE WHEN LOWER(content) LIKE ? THEN 1 ELSE 0 END"
            for _ in like_terms
        )
        like_params = [f"%{term.lower()}%" for term in like_terms]
        params = like_params + like_params + ([category] if category else []) + [LIKE_LIMIT]
        like_rows = db.execute(
            f"""SELECT id, content, category, source, importance,
                       valence, arousal, resolved, decay_rate,
                       created_at, recalled_count,
                       last_accessed_at, pinned,
                       ({score_sql}) AS matched_terms
                FROM memories
                WHERE ({where_sql}) AND superseded_by IS NULL {cat_sql}
                ORDER BY matched_terms DESC, created_at DESC
                LIMIT ?""",
            params,
        ).fetchall()
        like_ranking, like_details = _rank_like_rows(
            like_rows,
            key_prefix="mem",
            content_field="content",
            details_factory=lambda r: dict(r),
        )
        for key, item in like_details.items():
            details.setdefault(key, item)

    return fts_ranking, vec_ranking, like_ranking, details


def _search_bank_channels(query, query_vec, db, *, limit=50, diagnostics=None):
    """Return (fts_ranking, vec_ranking, like_ranking, details) for bank pool."""
    _index_bank_files()
    details = {}
    fts_ranking = []
    vec_ranking = []

    like_terms = like_search_terms(query)
    if like_terms:
        where_sql = " OR ".join("LOWER(chunk_text) LIKE ?" for _ in like_terms)
        score_sql = " + ".join(
            "CASE WHEN LOWER(chunk_text) LIKE ? THEN 1 ELSE 0 END"
            for _ in like_terms
        )
        like_params = [f"%{term.lower()}%" for term in like_terms]
        kw_rows = db.execute(
            f"""SELECT id, chunk_text, file_path, file_mtime,
                       ({score_sql}) AS matched_terms
                FROM bank_chunks
                WHERE ({where_sql})
                ORDER BY matched_terms DESC, file_mtime DESC
                LIMIT ?""",
            like_params + like_params + [limit],
        ).fetchall()
        fts_ranking, bank_details = _rank_like_rows(
            kw_rows,
            key_prefix="bank",
            content_field="chunk_text",
            details_factory=lambda r: {
                "id": r["id"],
                "content": r["chunk_text"],
                "source": Path(r["file_path"]).stem,
                "file_path": r["file_path"],
                "file_mtime": r["file_mtime"],
                "category": "bank",
            },
        )
        details.update(bank_details)

    if query_vec:
        try:
            v_rows = db.execute(
                "SELECT id, chunk_text, file_path, file_mtime, embedding "
                "FROM bank_chunks WHERE embedding IS NOT NULL"
            ).fetchall()
            scored = []
            for r in v_rows:
                sim = _cosine_similarity(query_vec, _blob_to_vec(r["embedding"]))
                if sim >= VEC_PRE_FILTER:
                    scored.append((r, sim))
            scored.sort(key=lambda x: x[1], reverse=True)

            for idx, (r, sim) in enumerate(scored[:limit]):
                key = f"bank_{r['id']}"
                vec_ranking.append((key, idx + 1))
                if key not in details:
                    details[key] = {
                        "id": r["id"],
                        "content": r["chunk_text"],
                        "source": Path(r["file_path"]).stem,
                        "file_path": r["file_path"],
                        "file_mtime": r["file_mtime"],
                        "category": "bank",
                    }
                details[key]["vec_similarity"] = sim
        except Exception as exc:
            _mark_vector_failed(diagnostics)
            logger.warning(
                "[VectorSearch] Failed: bank vector channel query=%r error=%s. "
                "Falling back to keyword text retrieval.",
                query,
                exc,
                exc_info=True,
            )

    like_ranking = []
    return fts_ranking, vec_ranking, like_ranking, details


def _search_conv_channels(query, query_vec, db, *, platform="", limit=50):
    """Return (fts_ranking, vec_ranking, like_ranking, details) for conversation pool."""
    details = {}
    fts_ranking = []
    vec_ranking = []

    safe_q = _sanitize_fts(query)
    if safe_q:
        try:
            if platform:
                fts_rows = db.execute(
                    """SELECT c.id, c.platform, c.direction, c.speaker, c.content, c.created_at
                       FROM conversation_log_fts f
                       JOIN conversation_log c ON c.id = f.rowid
                       WHERE conversation_log_fts MATCH ? AND c.platform = ?
                       ORDER BY f.rank
                       LIMIT ?""",
                    (safe_q, platform, limit),
                ).fetchall()
            else:
                fts_rows = db.execute(
                    """SELECT c.id, c.platform, c.direction, c.speaker, c.content, c.created_at
                       FROM conversation_log_fts f
                       JOIN conversation_log c ON c.id = f.rowid
                       WHERE conversation_log_fts MATCH ?
                       ORDER BY f.rank
                       LIMIT ?""",
                    (safe_q, limit),
                ).fetchall()

            for idx, r in enumerate(fts_rows):
                key = f"conv_{r['id']}"
                fts_ranking.append((key, idx + 1))
                details[key] = dict(r)
        except Exception as exc:
            logger.warning(
                "Conversation FTS search failed; query=%r fts_query=%r error=%s",
                query,
                safe_q,
                exc,
                exc_info=True,
            )

    like_ranking = []
    like_terms = like_search_terms(query)
    if like_terms:
        where_sql = " OR ".join("LOWER(content) LIKE ?" for _ in like_terms)
        score_sql = " + ".join(
            "CASE WHEN LOWER(content) LIKE ? THEN 1 ELSE 0 END"
            for _ in like_terms
        )
        like_params = [f"%{term.lower()}%" for term in like_terms]
        if platform:
            like_rows = db.execute(
                f"""SELECT id, platform, direction, speaker, content, created_at,
                          ({score_sql}) AS matched_terms
                   FROM conversation_log
                   WHERE ({where_sql}) AND platform = ?
                   ORDER BY matched_terms DESC, created_at DESC
                   LIMIT ?""",
                like_params + like_params + [platform, LIKE_LIMIT],
            ).fetchall()
        else:
            like_rows = db.execute(
                f"""SELECT id, platform, direction, speaker, content, created_at,
                          ({score_sql}) AS matched_terms
                   FROM conversation_log
                   WHERE ({where_sql})
                   ORDER BY matched_terms DESC, created_at DESC
                   LIMIT ?""",
                like_params + like_params + [LIKE_LIMIT],
            ).fetchall()
        like_ranking, like_details = _rank_like_rows(
            like_rows,
            key_prefix="conv",
            content_field="content",
            details_factory=lambda r: dict(r),
        )
        for key, item in like_details.items():
            details.setdefault(key, item)

    return fts_ranking, vec_ranking, like_ranking, details


# ─── Graph Expansion ───────────────────────────────────

def _expand_via_edges(results: list[dict], db, max_expand: int = 3) -> list[dict]:
    """Append edge-connected memories to search results."""
    existing_ids = {r["id"] for r in results if r.get("pool") == "memory"}
    expanded = []

    for r in results:
        if r.get("pool") != "memory" or len(expanded) >= max_expand:
            break
        rid = r.get("id")
        if not rid:
            continue

        try:
            edges = db.execute("""
                SELECT e.id, e.relation, e.context,
                       CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as neighbor_id
                FROM memory_edges e
                WHERE (e.source_id = ? OR e.target_id = ?)
            """, (rid, rid, rid)).fetchall()
        except Exception:
            continue

        for edge in edges:
            nid = edge["neighbor_id"]
            if nid in existing_ids or len(expanded) >= max_expand:
                continue
            neighbor = db.execute(
                "SELECT * FROM memories WHERE id = ? AND superseded_by IS NULL", (nid,)
            ).fetchone()
            if neighbor:
                existing_ids.add(nid)
                expanded.append({
                    "pool": "memory", "score": 0, "rrf_raw": 0,
                    "source": "edge",
                    "edge_relation": edge["relation"],
                    "edge_context": edge["context"],
                    **dict(neighbor),
                })
                db.execute(
                    "UPDATE memory_edges SET surfaced_count = surfaced_count + 1 WHERE id = ?",
                    (edge["id"],),
                )

    if expanded:
        db.commit()

    return results + expanded


# ─── Unified Search ─────────────────────────────────────

def unified_search(
    query: str,
    limit: int = 10,
    pools: list[str] | None = None,
    category: str | None = None,
    platform: str = "",
    after: str | None = None,
    before: str | None = None,
    _internal: bool = False,
) -> list[dict]:
    """Search across all memory pools with RRF fusion and per-pool reranking.

    Args:
        query:    natural-language search query
        limit:    max results to return
        pools:    subset of ["memory", "bank", "conversation"]; None = all
        category: filter memory pool by category
        platform: filter conversation pool by platform
        after/before: ISO date strings to filter by time range
        _internal: skip side-effects (recalled_count, last_accessed_at) — for edge expansion

    Returns list of dicts sorted by final score, each containing:
        pool, score, rrf_raw, id, content, + pool-specific fields
    """
    if pools is None:
        pools = ["memory", "bank", "conversation"]

    after_bound, before_bound = _normalize_time_bounds(after=after, before=before)

    if (after_bound or before_bound) and "bank" in pools:
        pools = [p for p in pools if p != "bank"]

    db = _get_db()
    query_vec = _embed(query)
    diagnostics = {"vector_failed": not bool(query_vec)}
    search_mode = _search_mode(query_vec)
    if not query_vec:
        logger.warning(
            "[VectorSearch] Failed: embedding unavailable for provider=%s model=%s. "
            "Falling back to FTS5/LIKE text retrieval.",
            EMBED_PROVIDER,
            EMBED_MODEL,
        )
    all_rankings: list[list[tuple[str, int]]] = []
    all_details: dict[str, dict] = {}

    if "memory" in pools:
        m_fts, m_vec, m_like, m_det = _search_memory_channels(
            query, query_vec, db, category=category, diagnostics=diagnostics
        )
        _inject_default_ranks(m_fts, m_vec)
        all_rankings += [m_fts, m_vec, m_like]
        all_details.update(m_det)

    if "bank" in pools:
        b_fts, b_vec, b_like, b_det = _search_bank_channels(
            query, query_vec, db, diagnostics=diagnostics
        )
        _inject_default_ranks(b_fts, b_vec)
        all_rankings += [b_fts, b_vec, b_like]
        all_details.update(b_det)

    if "conversation" in pools:
        c_fts, c_vec, c_like, c_det = _search_conv_channels(
            query, query_vec, db, platform=platform
        )
        if c_vec:
            _inject_default_ranks(c_fts, c_vec)
        all_rankings += [c_fts, c_vec, c_like]
        all_details.update(c_det)

    if diagnostics.get("vector_failed"):
        search_mode = "fts5_fallback"

    rrf_scores = _rrf_fuse(all_rankings)

    # Per-pool rerank + within-pool normalisation
    pool_items: dict[str, list[dict]] = {"memory": [], "bank": [], "conversation": []}

    for key, rrf in rrf_scores.items():
        detail = all_details.get(key, {})

        if key.startswith("mem_"):
            pool = "memory"
            reranked = _rerank_memory(rrf, detail)
        elif key.startswith("bank_"):
            pool = "bank"
            reranked = _rerank_bank(rrf, detail)
        elif key.startswith("conv_"):
            pool = "conversation"
            reranked = _rerank_conv(rrf, detail)
        else:
            continue

        if reranked < MIN_FINAL_SCORE:
            continue

        detail.pop("embedding", None)
        pool_items[pool].append({
            "pool": pool, "score": reranked, "rrf_raw": rrf, **detail
        })

    # Normalise within each pool so pools compete on equal footing
    results: list[dict] = []
    for pool, items in pool_items.items():
        if not items:
            continue
        max_score = max(r["score"] for r in items)
        for r in items:
            r["score"] = r["score"] / max_score if max_score > 0 else 0
        results.extend(items)

    # Keyword presence boost: results containing query terms get a bonus.
    # This prevents semantically vague matches from outranking exact keyword hits.
    _KEYWORD_BOOST = 0.3
    query_terms = query.split() if " " in query else [query]
    for r in results:
        content = r.get("content", "")
        matched = sum(1 for t in query_terms if t in content)
        if matched:
            r["score"] += _KEYWORD_BOOST * (matched / len(query_terms))

    results.sort(key=lambda x: x["score"], reverse=True)

    # Time range filtering (after/before)
    if after_bound or before_bound:
        def _in_time_range(r):
            ts = r.get("created_at", "")
            if not ts:
                return True
            if after_bound and ts < after_bound:
                return False
            if before_bound and ts > before_bound:
                return False
            return True
        results = [r for r in results if _in_time_range(r)]

    results = results[:limit]

    # Graph expansion: append edge-connected memories
    if "memory" in pools:
        results = _expand_via_edges(results, db, max_expand=3)

    # Side-effect: update last_accessed_at + recalled_count
    if not _internal:
        mem_ids = [r["id"] for r in results if r.get("pool") == "memory"]
        if mem_ids:
            now = now_str()
            for mid in mem_ids:
                db.execute(
                    "UPDATE memories SET recalled_count = recalled_count + 1, "
                    "last_accessed_at = ? WHERE id = ?",
                    (now, mid),
                )
            db.commit()

    db.close()
    return _annotate_search_mode(results, search_mode)


_LOCALE_LABELS = {
    "en": {"memory": "Memory", "bank": "Bank", "conversation": "Conversation",
           "empty": "No matching results found"},
    "zh": {"memory": "记忆", "bank": "知识库", "conversation": "对话",
           "empty": "没有找到匹配的结果"},
}

def unified_search_text(
    query: str,
    limit: int = 10,
    pools: list[str] | None = None,
    platform: str = "",
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Format unified search results as readable text.
    Set IMPRINT_LOCALE=zh for Chinese labels, default English.
    after/before: ISO date strings to filter by time range."""
    results = unified_search(query, limit=limit, pools=pools, platform=platform, after=after, before=before)
    locale = os.environ.get("IMPRINT_LOCALE", "en")
    loc = _LOCALE_LABELS.get(locale, _LOCALE_LABELS["en"])
    if not results:
        return loc["empty"]

    _labels = {k: loc[k] for k in ("memory", "bank", "conversation")}
    lines: list[str] = []

    for r in results:
        label = _labels.get(r["pool"], r["pool"])
        score = f"{r['score']:.4f}"
        content = r.get("content", "")[:1000]

        if r["pool"] == "memory":
            cat = r.get("category", "")
            ts = r.get("created_at", "")
            memory_id = r.get("id", "")
            pin = " [pinned]" if r.get("pinned") else ""
            if r.get("source") == "edge":
                rel = r.get("edge_relation", "")
                lines.append(f"[{label}|edge|{rel}] #{memory_id} {content}")
            else:
                lines.append(f"[{label}|{cat}|{ts}]{pin} #{memory_id} ({score}) {content}")

        elif r["pool"] == "bank":
            src = r.get("source", "")
            bank_id = r.get("id", "")
            lines.append(f"[{label}|{src}|id=bank:{bank_id}] ({score}) {content}")

        elif r["pool"] == "conversation":
            plat = r.get("platform", "")
            dire = "<-" if r.get("direction") == "in" else "->"
            ts = r.get("created_at", "")
            conv_id = r.get("id", "")
            lines.append(f"[{label}|{plat}{dire}|{ts}|id=conversation:{conv_id}] ({score}) {content}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Pin / Tag / Edge operations
# ═══════════════════════════════════════════════════════════════

def pin_memory(memory_id: int) -> dict:
    """Pin a memory. Pinned memories bypass all time-decay in search."""
    db = _get_db()
    row = db.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}
    pinned_count = db.execute("SELECT COUNT(*) as c FROM memories WHERE pinned = 1").fetchone()["c"]
    db.execute("UPDATE memories SET pinned = 1, updated_at = ? WHERE id = ?", (now_str(), memory_id))
    db.commit()
    db.close()
    result = {"ok": True, "pinned": memory_id}
    if pinned_count >= 20:
        result["warning"] = f"Already {pinned_count} pinned memories — consider keeping under 20"
    return result


def unpin_memory(memory_id: int) -> dict:
    """Unpin a memory, restoring normal time-decay."""
    db = _get_db()
    row = db.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}
    db.execute("UPDATE memories SET pinned = 0, updated_at = ? WHERE id = ?", (now_str(), memory_id))
    db.commit()
    db.close()
    return {"ok": True, "unpinned": memory_id}


def add_tags(memory_id: int, tags: list[str]) -> dict:
    """Add tags to a memory (writes to memory_tags table and updates memories.tags JSON)."""
    db = _get_db()
    row = db.execute("SELECT id, tags FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}

    added = []
    for tag in tags:
        t = tag.strip()
        if t:
            try:
                db.execute("INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)", (memory_id, t))
                added.append(t)
            except sqlite3.IntegrityError:
                pass

    if added:
        existing_tags = json.loads(row["tags"] or "[]")
        merged = list(dict.fromkeys(existing_tags + added))
        db.execute("UPDATE memories SET tags = ? WHERE id = ?",
                   (json.dumps(merged, ensure_ascii=False), memory_id))

    db.commit()
    db.close()
    return {"ok": True, "memory_id": memory_id, "added": added}


def get_tags(memory_id: int) -> list[str]:
    """Get all tags for a memory."""
    db = _get_db()
    rows = db.execute("SELECT tag FROM memory_tags WHERE memory_id = ?", (memory_id,)).fetchall()
    db.close()
    return [r["tag"] for r in rows]


def add_edge(source_id: int, target_id: int, relation: str, context: str) -> dict:
    """Create a bidirectional edge between two memories."""
    if source_id == target_id:
        return {"ok": False, "error": "Cannot create edge to self"}

    db = _get_db()

    for mid in (source_id, target_id):
        row = db.execute("SELECT id FROM memories WHERE id = ?", (mid,)).fetchone()
        if not row:
            db.close()
            return {"ok": False, "error": f"Memory {mid} not found"}

    existing = db.execute("""
        SELECT id FROM memory_edges
        WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)
    """, (source_id, target_id, target_id, source_id)).fetchone()
    if existing:
        db.close()
        return {"ok": False, "error": f"Edge already exists (edge #{existing['id']})"}

    cursor = db.execute("""
        INSERT INTO memory_edges (source_id, target_id, relation, context, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (source_id, target_id, relation.strip(), context.strip(), now_str()))
    db.commit()
    db.close()
    return {"ok": True, "edge_id": cursor.lastrowid}


def get_edges(memory_id: int) -> list[dict]:
    """Get all edges for a memory, including neighbor previews."""
    db = _get_db()
    rows = db.execute("""
        SELECT e.id, e.source_id, e.target_id, e.relation, e.context,
               e.surfaced_count, e.used_count, e.created_at,
               CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as neighbor_id
        FROM memory_edges e
        WHERE e.source_id = ? OR e.target_id = ?
    """, (memory_id, memory_id, memory_id)).fetchall()

    edges = []
    for r in rows:
        neighbor = db.execute(
            "SELECT content, category FROM memories WHERE id = ?", (r["neighbor_id"],)
        ).fetchone()
        edges.append({
            "edge_id": r["id"],
            "source_id": r["source_id"],
            "target_id": r["target_id"],
            "neighbor_id": r["neighbor_id"],
            "neighbor_preview": neighbor["content"][:80] if neighbor else "(deleted)",
            "neighbor_category": neighbor["category"] if neighbor else "",
            "relation": r["relation"],
            "context": r["context"],
            "surfaced_count": r["surfaced_count"],
            "used_count": r["used_count"],
            "created_at": r["created_at"],
        })
    db.close()
    return edges


def get_relationship_snapshot() -> str:
    """Read CLAUDE.md relationship snapshot from data directory."""
    snapshot_path = DATA_DIR / "CLAUDE.md"
    try:
        return snapshot_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "No relationship snapshot found. Create CLAUDE.md in ~/.imprint/ directory."


def save_summary(content: str, turn_count: int = 0, platform: str = "unknown") -> dict:
    """Save a conversation summary. Truncates to 1500 chars max."""
    content = content.strip()[:1500]
    if not content:
        return {"error": "Empty summary content"}
    platform = (platform or "unknown").strip() or "unknown"
    try:
        turn_count = max(int(turn_count), 0)
    except (TypeError, ValueError):
        turn_count = 0
    db = _get_db()
    try:
        cursor = db.execute(
            "INSERT INTO summaries (content, turn_count, platform, created_at) VALUES (?, ?, ?, ?)",
            (content, turn_count, platform, now_str()),
        )
        db.commit()
        summary_id = cursor.lastrowid
    finally:
        db.close()
    return {"id": summary_id, "status": "saved"}


def get_recent_summaries(limit: int = 3) -> list[dict]:
    """Get recent conversation summaries, newest first."""
    db = _get_db()
    rows = db.execute(
        "SELECT id, content, turn_count, platform, created_at FROM summaries ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def update_summary(
    summary_id: int,
    content: str,
    turn_count: int = 0,
    platform: str = "unknown",
) -> dict:
    """Update a conversation summary by ID. Truncates content to 1500 chars max."""
    try:
        summary_id = int(summary_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid summary id"}

    content = (content or "").strip()[:1500]
    if not content:
        return {"ok": False, "error": "content is required"}

    platform = (platform or "unknown").strip() or "unknown"
    try:
        turn_count = max(int(turn_count), 0)
    except (TypeError, ValueError):
        turn_count = 0

    db = _get_db()
    try:
        cursor = db.execute(
            "UPDATE summaries SET content = ?, platform = ?, turn_count = ? WHERE id = ?",
            (content, platform, turn_count, summary_id),
        )
        db.commit()
        if cursor.rowcount == 0:
            return {"ok": False, "error": "summary not found"}
    finally:
        db.close()
    return {"ok": True}


def delete_summary(summary_id: int) -> dict:
    """Delete a conversation summary by ID."""
    try:
        summary_id = int(summary_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid summary id"}

    db = _get_db()
    try:
        cursor = db.execute("DELETE FROM summaries WHERE id = ?", (summary_id,))
        db.commit()
        if cursor.rowcount == 0:
            return {"ok": False, "error": "summary not found"}
    finally:
        db.close()
    return {"ok": True}


def build_context(query: str = "") -> str:
    """Build a full context document for conversation start.
    Assembles up to five layers, skipping empty ones.
    Total output capped at 3000 chars.
    """
    sections = []

    # Layer 1: Continuity rules (always present)
    sections.append(
        "=== 连续性规则 ===\n"
        "你正在继续一段持续的关系，不是第一次见面。请根据以下上下文自然地延续对话。"
    )

    # Layer 2: Relationship snapshot
    snapshot = get_relationship_snapshot()
    if not snapshot.startswith("No relationship snapshot"):
        sections.append(f"=== 关系快照 ===\n{snapshot}")

    # Layer 3: Recent summaries
    summaries = get_recent_summaries(limit=3)
    if summaries:
        summary_lines = []
        for s in summaries:
            summary_lines.append(f"[{s['created_at']}] {s['content']}")
        sections.append("=== 最近摘要 ===\n" + "\n".join(summary_lines))

    # Layer 4: Surfacing memories
    surfacing = get_surfacing_memories(limit=3)
    if surfacing:
        surf_lines = []
        for m in surfacing:
            surf_lines.append(
                f"[#{m['id']}|arousal={m['arousal']:.1f}] {m['content'][:300]}"
            )
        sections.append("=== 主动浮现记忆 ===\n" + "\n".join(surf_lines))

    # Layer 5: Relevant memories (only if query provided)
    relevant_text = ""
    if query.strip():
        relevant_text = unified_search_text(query.strip(), limit=5)
        if relevant_text and relevant_text not in ("No matching results found", "没有找到匹配的结果"):
            sections.append(f"=== 相关记忆 ===\n{relevant_text}")

    # If nothing beyond continuity rules
    if len(sections) <= 1:
        return "No context available yet. This appears to be a fresh start."

    # Length control: cap at 3000 chars
    result = "\n\n".join(sections)
    if len(result) > 3000:
        # Remove relevant memories first
        sections = [s for s in sections if not s.startswith("=== 相关记忆")]
        result = "\n\n".join(sections)
    if len(result) > 3000:
        # Then remove summaries
        sections = [s for s in sections if not s.startswith("=== 最近摘要")]
        result = "\n\n".join(sections)
    if len(result) > 3000:
        result = result[:2997] + "..."

    return result

