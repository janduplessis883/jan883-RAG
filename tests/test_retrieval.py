from pathlib import Path

from local_rag.chunking import SemanticChunk
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
