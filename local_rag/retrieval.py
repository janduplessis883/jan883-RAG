from __future__ import annotations

from local_rag.embeddings import OllamaClient


class SearchService:
    """Hybrid search: dense (vector) + lexical (FTS5/BM25) fused with Reciprocal Rank Fusion.

    RRF (score = sum over ranked lists of 1 / (k + rank)) needs no per-feature
    score calibration, which makes it the right glue for mixing cosine similarity
    with unbounded BM25 scores.
    """

    def __init__(self, config_manager, database) -> None:
        self.config_manager = config_manager
        self.database = database
        self.config = config_manager.load_merged()
        self.ollama = OllamaClient(self.config["ollama"])

    def _setting(self, key: str, default):
        # Defaults keep pre-existing settings.toml files working.
        return self.config["retrieval"].get(key, default)

    def search(self, query: str, limit: int | None = None, hybrid: bool | None = None) -> list[dict]:
        """Single-query hybrid search for the Search page."""
        desired_limit = limit or int(self._setting("top_k", 8))
        min_similarity = float(self._setting("min_similarity", 0.0))
        dedupe_by_source = bool(self._setting("dedupe_by_source", False))

        records = self.rank_candidates(query, hybrid=hybrid)
        # min_similarity gates weak dense-only matches; exact lexical matches
        # (person names, dates, codes) are never discarded by the semantic floor.
        records = [
            record
            for record in records
            if record.get("lexical_score") is not None
            or (record.get("similarity") is not None and record["similarity"] >= min_similarity)
        ]
        return self.finish_results(records, limit=desired_limit, dedupe_by_source=dedupe_by_source)

    def rank_candidates(self, query: str, hybrid: bool | None = None) -> list[dict]:
        """Hybrid candidate pool for one query, ranked by RRF score.

        Applies no min_similarity gate, source dedupe, or final limit, so
        multi-query fusion can operate on the full ranked list and only dedupe
        at the very end.
        """
        use_hybrid = self._setting("hybrid_enabled", True) if hybrid is None else hybrid
        return self._hybrid_pool(query) if use_hybrid else self._dense_pool(query)

    def finish_results(self, records: list[dict], *, limit: int, dedupe_by_source: bool) -> list[dict]:
        """Order fused records, dedupe by source, truncate, expand context."""
        ordered = sorted(records, key=lambda record: record["rrf_score"], reverse=True)
        selected: list[dict] = []
        seen_sources: set[int] = set()
        for record in ordered:
            if dedupe_by_source and record["source_id"] in seen_sources:
                continue
            record["preview"] = record["text"][:420]
            selected.append(record)
            seen_sources.add(record["source_id"])
            if len(selected) >= limit:
                break
        self.expand_context(selected)
        return selected

    def expand_context(self, results: list[dict]) -> list[dict]:
        """Small-to-big: replace each result's text with its ±radius chunk neighborhood.

        Free (no re-embedding) and helps a lot for meeting transcripts where a
        single chunk is one sentence of a decision.
        """
        radius = int(self._setting("expand_radius", 1))
        if radius <= 0 or not results:
            return results
        for record in results:
            neighborhood = self.database.get_chunk_neighborhood(
                record["source_id"], record["chunk_index"], radius
            )
            if len(neighborhood) > 1:
                record["text"] = " ".join(segment["text"] for segment in neighborhood)
        return results

    def _hybrid_pool(self, query: str) -> list[dict]:
        rrf_k = float(self._setting("rrf_k", 60))
        query_embedding = self.ollama.embed_texts([query])[0]
        dense = self.database.search_candidates(query_embedding, limit=int(self._setting("candidate_k", 24)))
        lexical = self.database.search_lexical(query, limit=int(self._setting("lexical_k", 24)))

        fused: dict[int, dict] = {}

        def _entry(chunk_id: int) -> dict:
            return fused.setdefault(
                chunk_id,
                {
                    "id": chunk_id,
                    "rrf_score": 0.0,
                    "similarity": None,
                    "lexical_score": None,
                    "retrieval_mode": "hybrid",
                },
            )

        for rank, (chunk_id, similarity) in enumerate(dense, start=1):
            record = _entry(chunk_id)
            record["similarity"] = similarity
            record["rrf_score"] += 1.0 / (rrf_k + rank)
        for rank, (chunk_id, lexical_score) in enumerate(lexical, start=1):
            record = _entry(chunk_id)
            record["lexical_score"] = lexical_score
            record["rrf_score"] += 1.0 / (rrf_k + rank)

        chunks = {item["id"]: item for item in self.database.get_chunks(list(fused))}
        results: list[dict] = []
        for record in fused.values():
            chunk = chunks.get(record["id"])
            if chunk is None:
                continue
            record.update(chunk)
            results.append(record)

        results.sort(key=lambda record: record["rrf_score"], reverse=True)
        return results

    def _dense_pool(self, query: str) -> list[dict]:
        """Return the dense ranking without querying FTS5."""
        query_embedding = self.ollama.embed_texts([query])[0]
        dense = self.database.search_candidates(
            query_embedding, limit=int(self._setting("candidate_k", 24))
        )
        chunks = {
            item["id"]: item
            for item in self.database.get_chunks([item[0] for item in dense])
        }
        results = []
        for chunk_id, similarity in dense:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                {
                    **chunk,
                    "rrf_score": similarity,
                    "similarity": similarity,
                    "lexical_score": None,
                    "retrieval_mode": "dense",
                }
            )
        return results
