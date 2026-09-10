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
from local_rag.extractors import extract_html_text


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

        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(sources)")}
        for name in ("deleted_at", "updated_at", "full_text"):
            if name not in columns:
                self.connection.execute(f"ALTER TABLE sources ADD COLUMN {name} TEXT")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, messages_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS operation_runs (
            id INTEGER PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
            details_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS notion_calendar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notion_page_id TEXT NOT NULL UNIQUE,
            source_id INTEGER NOT NULL UNIQUE REFERENCES sources(id) ON DELETE CASCADE,
            event TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', start_date TEXT NOT NULL,
            end_date TEXT, timezone TEXT, attendees_json TEXT NOT NULL DEFAULT '[]', location TEXT,
            tags_json TEXT NOT NULL DEFAULT '[]', completed INTEGER NOT NULL DEFAULT 0,
            blocked_by_json TEXT NOT NULL DEFAULT '[]', blocking_json TEXT NOT NULL DEFAULT '[]',
            teams_link TEXT, created_time TEXT, last_edited_time TEXT,
            created_by_json TEXT, last_edited_by_json TEXT, rag_id TEXT,
            raw_json TEXT NOT NULL DEFAULT '{}', synced_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        calendar_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(notion_calendar)")}
        if "rag_id" not in calendar_columns:
            self.connection.execute("ALTER TABLE notion_calendar ADD COLUMN rag_id TEXT")

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

    @staticmethod
    def hash_text(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
        full_text: str | None = None,
        tags: list[str],
        metadata: dict[str, Any],
        raw_text_path: str | None,
        raw_binary_path: str | None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO sources (
                source_type, title, canonical_uri, external_ref, content_hash,
                summary, full_text, tags_json, metadata_json, raw_text_path, raw_binary_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_type,
                title,
                canonical_uri,
                external_ref,
                content_hash,
                summary,
                full_text,
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
    def replace_chunks(self, source_id: int, chunks: list, embeddings: list[list[float]], *, commit: bool = True) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have an embedding.")
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

        if commit:
            self.connection.commit()

    @synchronized
    def search_candidates(self, query_embedding: list[float], limit: int, source_ids: list[int] | None = None) -> list[tuple[int, float]]:
        if self.sqlite_vec and source_ids is None:
            try:
                rows = self.connection.execute(
                    "SELECT rowid, distance FROM chunk_index WHERE embedding MATCH ? AND k = ?",
                    (self.sqlite_vec.serialize_float32(query_embedding), limit + int(self.connection.execute("SELECT COUNT(*) FROM chunks JOIN sources ON sources.id = chunks.source_id WHERE sources.deleted_at IS NOT NULL").fetchone()[0])),
                ).fetchall()
                candidate_ids = [int(row["rowid"]) for row in rows]
                return self._rerank_candidates(candidate_ids, query_embedding)
            except sqlite3.DatabaseError:
                pass

        scope = "" if source_ids is None else " AND sources.id IN (" + ",".join("?" for _ in source_ids) + ")"
        rows = self.connection.execute("SELECT chunk_id, embedding FROM chunk_vectors JOIN chunks ON chunks.id = chunk_id JOIN sources ON sources.id = chunks.source_id WHERE sources.deleted_at IS NULL" + scope, tuple(source_ids or [])).fetchall()
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
            f"SELECT chunk_id, embedding FROM chunk_vectors JOIN chunks ON chunks.id = chunk_id JOIN sources ON sources.id = chunks.source_id WHERE sources.deleted_at IS NULL AND chunk_id IN ({placeholders})",
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
    def search_lexical(self, query: str, limit: int, source_ids: list[int] | None = None) -> list[tuple[int, float]]:
        """BM25-ranked chunks for a free-text query via FTS5.

        Returns (chunk_id, score) ordered best-first; higher score is better
        (bm25() is negative, so it is flipped).
        """
        if not self.fts5:
            return []
        match = self._build_fts_match(query)
        if not match:
            return []
        scope = "" if source_ids is None else " AND sources.id IN (" + ",".join("?" for _ in source_ids) + ")"
        try:
            rows = self.connection.execute(
                f"""
                SELECT chunk_fts.rowid, bm25(chunk_fts) AS score
                FROM chunk_fts JOIN chunks ON chunks.id = chunk_fts.rowid JOIN sources ON sources.id = chunks.source_id
                WHERE sources.deleted_at IS NULL AND chunk_fts MATCH ? {scope}
                ORDER BY score
                LIMIT ?
                """,
                (match, *(source_ids or []), limit),
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
            WHERE sources.deleted_at IS NULL AND chunks.id IN ({placeholders})
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
            "source_count": int(self.connection.execute("SELECT COUNT(*) FROM sources WHERE deleted_at IS NULL").fetchone()[0]),
            "chunk_count": int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
            "embedding_count": int(self.connection.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]),
        }
        if self.fts5:
            stats["fts_count"] = int(self.connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0])
        return stats

    @synchronized
    def list_sources(self, limit: int = 20) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM sources WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT ?",
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

    @synchronized
    def library_sources(self, *, removed: bool = False, include_calendar: bool = False) -> list[dict]:
        calendar_filter = "" if include_calendar else " AND source_type != 'notion_calendar'"
        rows = self.connection.execute(
            "SELECT id, title, source_type, tags_json, created_at, updated_at, canonical_uri "
            "FROM sources WHERE deleted_at IS " + ("NOT NULL" if removed else "NULL") + calendar_filter +
            " ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [{**dict(row), "tags": json.loads(row["tags_json"])} for row in rows]

    @synchronized
    def get_source(self, source_id: int) -> dict | None:
        row = self.connection.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            return None
        source = dict(row)
        source["tags"] = json.loads(source["tags_json"])
        source["metadata"] = json.loads(source["metadata_json"])
        text = source["full_text"]
        if text is None and source["raw_text_path"]:
            try:
                text = Path(source["raw_text_path"]).read_text(encoding="utf-8")
            except OSError:
                pass
            if text and source["source_type"] == "article":
                extracted_text = extract_html_text(text)
                if extracted_text:
                    text = extracted_text
                    self.connection.execute(
                        "UPDATE sources SET full_text=? WHERE id=?",
                        (text, source_id),
                    )
                    self.connection.commit()
        if text is None:
            text = "\n\n".join(r[0] for r in self.connection.execute(
                "SELECT text FROM chunks WHERE source_id = ? ORDER BY chunk_index", (source_id,)))
        source["full_text"] = text
        return source

    @synchronized
    def set_source_removed(self, source_id: int, removed: bool) -> None:
        with self.connection:
            self.connection.execute("UPDATE sources SET deleted_at = " +
                                    ("CURRENT_TIMESTAMP" if removed else "NULL") + " WHERE id = ?", (source_id,))

    @synchronized
    def update_document(self, source_id: int, *, title: str, text: str, tags: list[str],
                        content_hash: str, chunks: list, embeddings: list) -> None:
        with self.connection:
            row = self.connection.execute("SELECT id FROM sources WHERE id = ? AND deleted_at IS NULL", (source_id,)).fetchone()
            if row is None:
                raise ValueError("This document is no longer in the library.")
            # Delete old FTS entries while the original title is still available.
            self.replace_chunks(source_id, [], [], commit=False)
            self.connection.execute(
                "UPDATE sources SET title=?, full_text=?, tags_json=?, content_hash=?, summary=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (title, text, json.dumps(tags), content_hash, text[:280], source_id))
            self.replace_chunks(source_id, chunks, embeddings, commit=False)

    @synchronized
    def update_source_tags(self, source_id: int, tags: list[str]) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE sources SET tags_json=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND deleted_at IS NULL",
                (json.dumps(tags), source_id),
            )

    @synchronized
    def update_source_metadata(self, source_id: int, metadata: dict) -> None:
        with self.connection:
            self.connection.execute("UPDATE sources SET metadata_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                    (json.dumps(metadata), source_id))

    @synchronized
    def upsert_calendar_event(self, event, source_id: int) -> None:
        values = (event.page_id, source_id, event.event, event.description, event.start_date, event.end_date,
                  event.timezone, json.dumps(event.attendees), event.location, json.dumps(event.tags), int(event.completed),
                  json.dumps(event.blocked_by), json.dumps(event.blocking), event.teams_link, event.created_time,
                  event.last_edited_time, json.dumps(event.created_by), json.dumps(event.last_edited_by), event.rag_id, json.dumps(event.raw_json))
        with self.connection:
            self.connection.execute("""INSERT INTO notion_calendar(
                notion_page_id, source_id, event, description, start_date, end_date, timezone,
                attendees_json, location, tags_json, completed, blocked_by_json, blocking_json,
                teams_link, created_time, last_edited_time, created_by_json, last_edited_by_json, rag_id, raw_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(notion_page_id) DO UPDATE SET source_id=excluded.source_id, event=excluded.event,
                description=excluded.description, start_date=excluded.start_date, end_date=excluded.end_date,
                timezone=excluded.timezone, attendees_json=excluded.attendees_json, location=excluded.location,
                tags_json=excluded.tags_json, completed=excluded.completed, blocked_by_json=excluded.blocked_by_json,
                blocking_json=excluded.blocking_json, teams_link=excluded.teams_link, created_time=excluded.created_time,
                last_edited_time=excluded.last_edited_time, created_by_json=excluded.created_by_json,
                last_edited_by_json=excluded.last_edited_by_json, rag_id=excluded.rag_id, raw_json=excluded.raw_json, synced_at=CURRENT_TIMESTAMP""", values)

    @synchronized
    def calendar_page_ids(self) -> set[str]:
        return {row[0] for row in self.connection.execute("SELECT notion_page_id FROM notion_calendar")}

    @synchronized
    def calendar_source_ids(self) -> list[int]:
        return [int(row[0]) for row in self.connection.execute(
            "SELECT source_id FROM notion_calendar JOIN sources ON sources.id=source_id WHERE sources.deleted_at IS NULL")]

    @synchronized
    def active_source_ids(self, include_calendar: bool = True) -> list[int]:
        where = "" if include_calendar else " AND source_type != 'notion_calendar'"
        return [int(row[0]) for row in self.connection.execute(
            "SELECT id FROM sources WHERE deleted_at IS NULL" + where)]

    @synchronized
    def list_calendar_events(self, source_ids: list[int] | None = None) -> list[dict]:
        scope = "" if source_ids is None else " WHERE source_id IN (" + ",".join("?" for _ in source_ids) + ")"
        rows = self.connection.execute("SELECT * FROM notion_calendar" + scope + " ORDER BY start_date", tuple(source_ids or [])).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for field in ("attendees_json", "tags_json", "blocked_by_json", "blocking_json", "created_by_json", "last_edited_by_json", "raw_json"):
                item[field.removesuffix("_json")] = json.loads(item.pop(field) or ("{}" if field in {"created_by_json", "last_edited_by_json", "raw_json"} else "[]"))
            result.append(item)
        return result

    @synchronized
    def delete_calendar_event(self, notion_page_id: str) -> None:
        with self.connection:
            row = self.connection.execute("SELECT source_id FROM notion_calendar WHERE notion_page_id=?", (notion_page_id,)).fetchone()
            if not row:
                return
            source_id = int(row[0])
            chunk_ids = [r[0] for r in self.connection.execute("SELECT id FROM chunks WHERE source_id=?", (source_id,))]
            if self.sqlite_vec:
                for chunk_id in chunk_ids:
                    self.connection.execute("DELETE FROM chunk_index WHERE rowid=?", (chunk_id,))
            if self.fts5:
                fts_rows = self.connection.execute(
                    "SELECT chunks.id, sources.title, chunks.text FROM chunks JOIN sources ON sources.id=chunks.source_id WHERE chunks.source_id=?",
                    (source_id,),
                ).fetchall()
                for fts_row in fts_rows:
                    self.connection.execute(
                        "INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', ?, ?)",
                        (fts_row["id"], self._fts_index_text(fts_row["title"] or "", fts_row["text"] or "")),
                    )
            self.connection.execute("DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id=?)", (source_id,))
            self.connection.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
            self.connection.execute("DELETE FROM notion_calendar WHERE notion_page_id=?", (notion_page_id,))
            self.connection.execute("DELETE FROM sources WHERE id=?", (source_id,))

    @synchronized
    def save_conversation(self, conversation_id: str, messages: list[dict]) -> None:
        title = next((m["content"][:80] for m in messages if m["role"] == "user"), "New conversation")
        with self.connection:
            self.connection.execute("""INSERT INTO conversations(id, title, messages_json) VALUES(?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title, messages_json=excluded.messages_json,
                updated_at=CURRENT_TIMESTAMP""", (conversation_id, title, json.dumps(messages)))

    @synchronized
    def list_conversations(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC, rowid DESC")]

    @synchronized
    def load_conversation(self, conversation_id: str) -> list[dict]:
        row = self.connection.execute("SELECT messages_json FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return json.loads(row[0]) if row else []

    @synchronized
    def record_operation(self, kind: str, result: dict) -> None:
        failed = result.get("status") == "error" or bool(result.get("errors")) or any(
            item.get("status") == "error" for item in result.get("items", []))
        status = "error" if failed else result.get("status", "ok")
        with self.connection:
            self.connection.execute("INSERT INTO operation_runs(kind,status,details_json) VALUES(?,?,?)",
                                    (kind, status, json.dumps(result, default=str)))

    @synchronized
    def operation_history(self, limit: int = 50) -> list[dict]:
        return [{**dict(row), "details": json.loads(row["details_json"])} for row in self.connection.execute(
            "SELECT * FROM operation_runs ORDER BY id DESC LIMIT ?", (limit,))]

    @synchronized
    def last_successful_operation(self, kind: str) -> dict | None:
        row = self.connection.execute(
            "SELECT created_at, details_json FROM operation_runs WHERE kind=? AND status IN ('ok','ingested','duplicate') ORDER BY id DESC LIMIT 1",
            (kind,)).fetchone()
        return {"created_at": row[0], "details": json.loads(row[1])} if row else None
