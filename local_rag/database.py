from __future__ import annotations

from array import array
from pathlib import Path
import json
import math
import re
import sqlite3
from functools import wraps
from threading import RLock
from typing import Any

from local_rag.chunking import cosine_similarity


def synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class Database:
    def __init__(self, config_manager) -> None:
        self.config_manager = config_manager
        self.config = config_manager.load_merged()
        self.db_path = config_manager.resolve_path(self.config["app"]["database_path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.sqlite_vec = self._load_sqlite_vec()
        self.fts5 = self._probe_fts5()

    def _load_sqlite_vec(self):
        try:
            import sqlite_vec
        except ImportError:
            return None

        if not hasattr(self.connection, "enable_load_extension"):
            return None

        try:
            self.connection.enable_load_extension(True)
            sqlite_vec.load(self.connection)
            self.connection.enable_load_extension(False)
            return sqlite_vec
        except (AttributeError, sqlite3.DatabaseError, sqlite3.OperationalError):
            return None

    def _probe_fts5(self) -> bool:
        try:
            self.connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS app_fts_probe USING fts5(probe)")
            self.connection.execute("DROP TABLE IF EXISTS app_fts_probe")
            self.connection.commit()
            return True
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            return False

    @synchronized
    def initialize(self) -> None:
        embedding_dimensions = int(self.config["ollama"]["embedding_dimensions"])
        self.connection.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                canonical_uri TEXT,
                external_ref TEXT,
                content_hash TEXT NOT NULL UNIQUE,
                summary TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                raw_text_path TEXT,
                raw_binary_path TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_canonical_uri
            ON sources(canonical_uri)
            WHERE canonical_uri IS NOT NULL;

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ingestion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                content_hash TEXT,
                source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
                status TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        if self.sqlite_vec:
            self.connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_index USING vec0(embedding float[{embedding_dimensions}])"
            )

        if self.fts5:
            self.connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(text, content='')")
            fts_count = int(self.connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0])
            chunk_count = int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            if fts_count != chunk_count:
                self.rebuild_fts_index()
        self.connection.commit()

    @synchronized
    def find_duplicate(self, canonical_uri: str | None, content_hash: str) -> sqlite3.Row | None:
        if canonical_uri:
            row = self.connection.execute(
                "SELECT * FROM sources WHERE canonical_uri = ?",
                (canonical_uri,),
            ).fetchone()
            if row:
                return row
        return self.connection.execute(
            "SELECT * FROM sources WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()

    @synchronized
    def find_source_by_uri(self, canonical_uri: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM sources WHERE canonical_uri = ?",
            (canonical_uri,),
        ).fetchone()

    @synchronized
    def insert_source(
        self,
        *,
        source_type: str,
        title: str,
        canonical_uri: str | None,
        external_ref: str | None,
        content_hash: str,
        summary: str,
        tags: list[str],
        metadata: dict[str, Any],
        raw_text_path: str | None,
        raw_binary_path: str | None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO sources (
                source_type, title, canonical_uri, external_ref, content_hash,
                summary, tags_json, metadata_json, raw_text_path, raw_binary_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_type,
                title,
                canonical_uri,
                external_ref,
                content_hash,
                summary,
                json.dumps(tags),
                json.dumps(metadata),
                raw_text_path,
                raw_binary_path,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    _ISO_DATE_RE = re.compile(r"(\d{4})[-/](\d{2})[-/](\d{2})")

    @classmethod
    def _normalize_dates(cls, text: str) -> str:
        """Collapse ISO dates (2026-08-21) into a single token (20260821) so a full
        date in a query matches exactly, instead of degrading to '2026' OR 'meeting'."""
        return cls._ISO_DATE_RE.sub(r"\1\2\3", text)

    def _fts_index_text(self, title: str, text: str) -> str:
        """Indexed string: source title + chunk body, so titles (meeting dates) are matchable."""
        text = self._normalize_dates(text)
        return f"{self._normalize_dates(title)}\n{text}" if title else text

    @synchronized
    def replace_chunks(self, source_id: int, chunks: list, embeddings: list[list[float]]) -> None:
        source_row = self.connection.execute("SELECT title FROM sources WHERE id = ?", (source_id,)).fetchone()
        source_title = source_row["title"] if source_row else ""
        existing = self.connection.execute(
            "SELECT id, text FROM chunks WHERE source_id = ?", (source_id,)
        ).fetchall()
        for row in existing:
            chunk_id = int(row["id"])
            self.connection.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
            if self.sqlite_vec:
                self.connection.execute("DELETE FROM chunk_index WHERE rowid = ?", (chunk_id,))
            if self.fts5:
                self.connection.execute(
                    "INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', ?, ?)",
                    (chunk_id, self._fts_index_text(source_title, row["text"])),
                )
        self.connection.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cursor = self.connection.execute(
                "INSERT INTO chunks (source_id, chunk_index, text, char_count, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (source_id, index, chunk.text, chunk.char_count, "{}"),
            )
            chunk_id = int(cursor.lastrowid)
            self.connection.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding, dimension) VALUES (?, ?, ?)",
                (chunk_id, self._pack_embedding(embedding), len(embedding)),
            )
            if self.sqlite_vec:
                self.connection.execute(
                    "INSERT INTO chunk_index (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, self.sqlite_vec.serialize_float32(embedding)),
                )
            if self.fts5:
                self.connection.execute(
                    "INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
                    (chunk_id, self._fts_index_text(source_title, chunk.text)),
                )

        self.connection.commit()

    @synchronized
    def search_candidates(self, query_embedding: list[float], limit: int) -> list[tuple[int, float]]:
        if self.sqlite_vec:
            try:
                rows = self.connection.execute(
                    "SELECT rowid, distance FROM chunk_index WHERE embedding MATCH ? AND k = ?",
                    (self.sqlite_vec.serialize_float32(query_embedding), limit),
                ).fetchall()
                candidate_ids = [int(row["rowid"]) for row in rows]
                return self._rerank_candidates(candidate_ids, query_embedding)
            except sqlite3.DatabaseError:
                pass

        rows = self.connection.execute("SELECT chunk_id, embedding FROM chunk_vectors").fetchall()
        scored = []
        for row in rows:
            embedding = self._unpack_embedding(row["embedding"])
            scored.append((int(row["chunk_id"]), cosine_similarity(query_embedding, embedding)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    @synchronized
    def _rerank_candidates(self, candidate_ids: list[int], query_embedding: list[float]) -> list[tuple[int, float]]:
        if not candidate_ids:
            return []
        placeholders = ",".join(["?"] * len(candidate_ids))
        rows = self.connection.execute(
            f"SELECT chunk_id, embedding FROM chunk_vectors WHERE chunk_id IN ({placeholders})",
            tuple(candidate_ids),
        ).fetchall()
        scores = []
        for row in rows:
            embedding = self._unpack_embedding(row["embedding"])
            scores.append((int(row["chunk_id"]), cosine_similarity(query_embedding, embedding)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    @classmethod
    def _build_fts_match(cls, query: str) -> str:
        """Turn free text into a safe FTS5 MATCH expression (quoted tokens, OR-joined).

        Keeps tokens of at least three characters to avoid noisy two-char fragments
        like the month/day pieces of a date; bm25() still rewards chunks that hit
        more of the tokens.
        """
        tokens: list[str] = []
        for token in re.findall(r"\w+", cls._normalize_dates(query), flags=re.UNICODE):
            if len(token) < 3:
                continue
            normalized = token.casefold()
            if normalized not in tokens:
                tokens.append(normalized)
            if len(tokens) == 16:
                break
        if not tokens:
            return ""
        return " OR ".join(f'"{token}"' for token in tokens)

    @synchronized
    def search_lexical(self, query: str, limit: int) -> list[tuple[int, float]]:
        """BM25-ranked chunks for a free-text query via FTS5.

        Returns (chunk_id, score) ordered best-first; higher score is better
        (bm25() is negative, so it is flipped).
        """
        if not self.fts5:
            return []
        match = self._build_fts_match(query)
        if not match:
            return []
        try:
            rows = self.connection.execute(
                """
                SELECT rowid, bm25(chunk_fts) AS score
                FROM chunk_fts
                WHERE chunk_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match, limit),
            ).fetchall()
        except sqlite3.DatabaseError:
            return []
        return [(int(row["rowid"]), -float(row["score"])) for row in rows]

    @synchronized
    def get_chunk_neighborhood(self, source_id: int, chunk_index: int, radius: int) -> list[dict]:
        """Adjacent chunks of a source (small-to-big context expansion)."""
        rows = self.connection.execute(
            """
            SELECT id, chunk_index, text
            FROM chunks
            WHERE source_id = ? AND chunk_index BETWEEN ? AND ?
            ORDER BY chunk_index
            """,
            (source_id, chunk_index - radius, chunk_index + radius),
        ).fetchall()
        return [
            {"id": int(row["id"]), "chunk_index": int(row["chunk_index"]), "text": row["text"]}
            for row in rows
        ]

    @synchronized
    def rebuild_fts_index(self) -> None:
        if not self.fts5:
            return
        self.connection.execute("DROP TABLE IF EXISTS chunk_fts")
        self.connection.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(text, content='')")
        rows = self.connection.execute(
            """
            SELECT chunks.id, sources.title, chunks.text
            FROM chunks
            JOIN sources ON sources.id = chunks.source_id
            ORDER BY chunks.id
            """
        ).fetchall()
        self.connection.executemany(
            "INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
            [(int(row["id"]), self._fts_index_text(row["title"] or "", row["text"] or "")) for row in rows],
        )
        self.connection.commit()

    @synchronized
    def get_chunks(self, chunk_ids: list[int]) -> list[dict]:
        if not chunk_ids:
            return []
        placeholders = ",".join(["?"] * len(chunk_ids))
        rows = self.connection.execute(
            f"""
            SELECT
                chunks.id,
                chunks.chunk_index,
                chunks.text,
                sources.id AS source_id,
                sources.title,
                sources.source_type,
                sources.canonical_uri,
                sources.tags_json,
                sources.metadata_json,
                sources.created_at
            FROM chunks
            JOIN sources ON sources.id = chunks.source_id
            WHERE chunks.id IN ({placeholders})
            """,
            tuple(chunk_ids),
        ).fetchall()
        items = []
        for row in rows:
            items.append(
                {
                    "id": int(row["id"]),
                    "source_id": int(row["source_id"]),
                    "chunk_index": int(row["chunk_index"]),
                    "text": row["text"],
                    "title": row["title"],
                    "source_type": row["source_type"],
                    "canonical_uri": row["canonical_uri"],
                    "tags": json.loads(row["tags_json"]),
                    "metadata": json.loads(row["metadata_json"]),
                    "created_at": row["created_at"],
                }
            )
        return items

    @synchronized
    def get_stats(self) -> dict[str, int]:
        stats = {
            "source_count": int(self.connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]),
            "chunk_count": int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
            "embedding_count": int(self.connection.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]),
        }
        if self.fts5:
            stats["fts_count"] = int(self.connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0])
        return stats

    @synchronized
    def list_sources(self, limit: int = 20) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM sources ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "title": row["title"],
                "source_type": row["source_type"],
                "canonical_uri": row["canonical_uri"],
                "summary": row["summary"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    @synchronized
    def get_state(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    @synchronized
    def set_state(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO app_state(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.connection.commit()

    @synchronized
    def get_ingestion_log_entry(self, file_path: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM ingestion_log WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "file_path": row["file_path"],
            "filename": row["filename"],
            "content_hash": row["content_hash"],
            "source_id": int(row["source_id"]) if row["source_id"] is not None else None,
            "status": row["status"],
            "message": row["message"],
            "created_at": row["created_at"],
        }

    @synchronized
    def record_ingestion_log(
        self,
        *,
        file_path: str,
        filename: str,
        content_hash: str | None,
        source_id: int | None,
        status: str,
        message: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO ingestion_log(file_path, filename, content_hash, source_id, status, message)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                filename = excluded.filename,
                content_hash = excluded.content_hash,
                source_id = excluded.source_id,
                status = excluded.status,
                message = excluded.message,
                created_at = CURRENT_TIMESTAMP
            """,
            (file_path, filename, content_hash, source_id, status, message),
        )
        self.connection.commit()

    @synchronized
    def list_ingestion_log(self, limit: int = 100) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM ingestion_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "file_path": row["file_path"],
                "filename": row["filename"],
                "source_id": int(row["source_id"]) if row["source_id"] is not None else None,
                "status": row["status"],
                "message": row["message"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    @synchronized
    def rebuild_vector_index(self) -> None:
        if not self.sqlite_vec:
            return
        self.connection.execute("DROP TABLE IF EXISTS chunk_index")
        dimensions = int(self.config["ollama"]["embedding_dimensions"])
        self.connection.execute(
            f"CREATE VIRTUAL TABLE chunk_index USING vec0(embedding float[{dimensions}])"
        )
        rows = self.connection.execute("SELECT chunk_id, embedding FROM chunk_vectors ORDER BY chunk_id").fetchall()
        for row in rows:
            self.connection.execute(
                "INSERT INTO chunk_index (rowid, embedding) VALUES (?, ?)",
                (int(row["chunk_id"]), self.sqlite_vec.serialize_float32(self._unpack_embedding(row["embedding"]))),
            )
        self.connection.commit()

    def _pack_embedding(self, embedding: list[float]) -> bytes:
        return array("f", embedding).tobytes()

    def _unpack_embedding(self, raw: bytes) -> list[float]:
        values = array("f")
        values.frombytes(raw)
        return list(values)
