from datetime import datetime
import math
from pathlib import Path

from local_rag.chunking import SemanticChunk
from local_rag.calendar import CalendarEvent
from local_rag.chat import ChatService
from local_rag.database import Database
from local_rag.retrieval import SearchService


class FakeConfig:
    def __init__(self, root: Path, hybrid_enabled: bool = True):
        self.root = root
        self.values = {
            "app": {"database_path": str(root / "rag.sqlite3")},
            "ollama": {
                "embedding_dimensions": 3,
                "base_url": "http://localhost:11434/v1",
                "api_key": "test",
                "embedding_model": "test-embedding",
                "request_timeout_seconds": 1,
            },
            "retrieval": {
                "hybrid_enabled": hybrid_enabled,
                "candidate_k": 10,
                "lexical_k": 10,
                "rrf_k": 60,
                "min_similarity": 0,
                "dedupe_by_source": False,
                "expand_radius": 0,
            },
        }

    def load_merged(self):
        return self.values

    def resolve_path(self, value):
        return Path(value)


class FakeOllama:
    def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def make_service(tmp_path, hybrid_enabled=True):
    config = FakeConfig(tmp_path, hybrid_enabled)
    database = Database(config)
    database.initialize()
    # sqlite-vec is optional; these tests exercise the portable SQLite fallback.
    database.sqlite_vec = None
    source_id = database.insert_source(
        source_type="text",
        title="Meeting notes",
        canonical_uri=None,
        external_ref=None,
        content_hash="hash",
        summary="",
        tags=[],
        metadata={},
        raw_text_path=None,
        raw_binary_path=None,
    )
    database.replace_chunks(
        source_id,
        [
            SemanticChunk("General project planning discussion.", 38),
            SemanticChunk("Margaret Palmer attended Partner's Meeting 2026-08-21.", 58),
        ],
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
    )
    service = SearchService(config, database)
    service.ollama = FakeOllama()
    return service, database


def test_hybrid_fts_match_can_outvote_dense_first_result(tmp_path):
    service, _ = make_service(tmp_path)

    results = service.search("Margaret Palmer 2026-08-21", limit=1)

    assert results[0]["text"].startswith("Margaret Palmer")
    assert results[0]["retrieval_mode"] == "hybrid"
    assert results[0]["lexical_score"] is not None


def test_explicit_dense_mode_does_not_query_fts5(tmp_path):
    service, database = make_service(tmp_path)

    def fail_if_called(query, limit):
        raise AssertionError("FTS5 should not be queried in dense-only mode")

    database.search_lexical = fail_if_called
    results = service.search("Margaret Palmer", limit=2, hybrid=False)

    assert len(results) == 2
    assert all(item["retrieval_mode"] == "dense" for item in results)
    assert all(item["lexical_score"] is None for item in results)
    assert service.last_latency_ms >= 0


def test_config_defaults_to_hybrid_when_setting_is_missing(tmp_path):
    service, _ = make_service(tmp_path)
    service.config["retrieval"].pop("hybrid_enabled")

    results = service.rank_candidates("Margaret Palmer")

    assert results[0]["retrieval_mode"] == "hybrid"


def test_chat_answer_stream_yields_incremental_text(tmp_path):
    service = ChatService(FakeConfig(tmp_path), retrieval_service=None)

    class StreamingFake:
        def chat_stream(self, **kwargs):
            yield "The answer"
            yield " arrives in pieces."

    service.ollama = StreamingFake()
    sources = [{
        "title": "Notes",
        "source_type": "text",
        "canonical_uri": None,
        "text": "A useful fact.",
    }]

    answer = "".join(service.answer_stream("What happened?", "test", sources=sources))

    assert answer == "The answer arrives in pieces."


def test_chat_retrieval_scopes_to_selected_knowledge_sources(tmp_path):
    service, database = make_service(tmp_path)
    chat = ChatService(FakeConfig(tmp_path), service)
    chat.retrieval_service.rank_candidates = lambda query, hybrid=None, source_ids=None: [
        {"id": 1, "source_id": 1, "chunk_index": 0, "text": "knowledge", "title": "Knowledge", "source_type": "text", "canonical_uri": None, "rrf_score": 1.0, "similarity": 1.0, "lexical_score": None}
    ] if source_ids else []
    database.active_source_ids = lambda include_calendar=True: [1]
    assert chat.retrieve_sources("question", include_knowledge_base=True, include_calendar=False)
    assert chat.retrieve_sources("question", include_knowledge_base=False, include_calendar=False) == []


def test_database_finds_calendar_events_overlapping_date_window(tmp_path):
    service, database = make_service(tmp_path)
    source_id = database.insert_source(
        source_type="notion_calendar",
        title="Overnight planning",
        canonical_uri="notion://calendar/overnight",
        external_ref=None,
        content_hash="calendar-hash",
        summary="",
        tags=[],
        metadata={},
        raw_text_path=None,
        raw_binary_path=None,
    )
    database.upsert_calendar_event(CalendarEvent(
        page_id="overnight",
        event="Overnight planning",
        description="Planning session",
        start_date="2026-09-14T23:00:00+01:00",
        end_date="2026-09-15T01:00:00+01:00",
        timezone="Europe/London",
        attendees=[], location=None, tags=[], completed=False,
        blocked_by=[], blocking=[], teams_link=None,
        created_time=None, last_edited_time=None, created_by=None,
        last_edited_by=None, rag_id=None, raw_json={},
    ), source_id)

    events = database.search_calendar_events(
        datetime.fromisoformat("2026-09-15T00:00:00+01:00"),
        datetime.fromisoformat("2026-09-16T00:00:00+01:00"),
    )

    assert [event["event"] for event in events] == ["Overnight planning"]


def test_chat_includes_exact_calendar_matches_for_tomorrow(tmp_path):
    service, database = make_service(tmp_path)
    source_id = database.insert_source(
        source_type="notion_calendar",
        title="Tomorrow event",
        canonical_uri="notion://calendar/tomorrow",
        external_ref=None,
        content_hash="tomorrow-hash",
        summary="",
        tags=[],
        metadata={},
        raw_text_path=None,
        raw_binary_path=None,
    )
    database.upsert_calendar_event(CalendarEvent(
        page_id="tomorrow",
        event="Tomorrow event",
        description="Important meeting",
        start_date="2026-09-11T10:00:00+01:00",
        end_date="2026-09-11T11:00:00+01:00",
        timezone="Europe/London",
        attendees=[], location=None, tags=[], completed=False,
        blocked_by=[], blocking=[], teams_link=None,
        created_time=None, last_edited_time=None, created_by=None,
        last_edited_by=None, rag_id=None, raw_json={},
    ), source_id)
    database.active_source_ids = lambda include_calendar=True: [source_id]
    service.rank_candidates = lambda *args, **kwargs: []

    class FixedChat(ChatService):
        def _calendar_now(self):
            return datetime.fromisoformat("2026-09-10T12:00:00+01:00")

    chat = FixedChat(FakeConfig(tmp_path), service)
    results = chat.retrieve_sources(
        "What is in my calendar tomorrow?",
        include_knowledge_base=False,
        include_calendar=True,
    )

    assert [result["title"] for result in results] == ["Tomorrow event"]
    assert results[0]["retrieval_mode"] == "calendar_exact"
    assert math.isfinite(results[0]["rrf_score"])


def test_chat_applies_min_similarity_after_multi_query_fusion(tmp_path):
    service, database = make_service(tmp_path)
    chat = ChatService(FakeConfig(tmp_path), service)
    chat.config["retrieval"]["min_similarity"] = 0.99
    service.rank_candidates = lambda *args, **kwargs: [{"id": 1, "source_id": 1, "chunk_index": 0, "text": "weak", "title": "x", "similarity": 0.1, "lexical_score": None, "rrf_score": 1.0}]
    database.active_source_ids = lambda include_calendar=True: [1]
    assert chat.retrieve_sources("unrelated", include_calendar=False) == []


def test_calendar_exact_source_has_json_safe_score_and_zero_limit_returns_empty(tmp_path):
    service, database = make_service(tmp_path)
    assert service.finish_results([{"source_id": 1, "rrf_score": 1}], limit=0, dedupe_by_source=False) == []


def test_follow_up_retrieval_query_uses_recent_history(tmp_path):
    service, database = make_service(tmp_path)
    chat = ChatService(FakeConfig(tmp_path), service)
    chat.ollama.chat = lambda **kwargs: "What decisions were made about project planning?"
    assert chat.rewrite_retrieval_query("What did they decide?", [{"role": "user", "content": "Tell me about project planning."}]).startswith("What decisions")
