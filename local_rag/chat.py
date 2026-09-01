from __future__ import annotations

import json
import re
from typing import Iterator

from local_rag.embeddings import OllamaClient
from local_rag.observability import LangfuseObserver


class ChatService:
    def __init__(self, config_manager, retrieval_service) -> None:
        self.config_manager = config_manager
        self.retrieval_service = retrieval_service
        self.config = config_manager.load_merged()
        self.ollama = OllamaClient(self.config["ollama"])
        self.observability = LangfuseObserver(self.config)

    @property
    def langfuse(self) -> LangfuseObserver:
        return self.observability

    def generate_related_questions(self, question: str, model_name: str | None = None) -> list[str]:
        model = model_name or self.config["ollama"]["default_answer_model"]
        system_prompt = (
            "You generate search-query variations for a retrieval-augmented knowledge base. "
            "Return exactly three concise related questions as a JSON array of strings. "
            "Keep the original intent, but vary wording, perspective, and useful terminology. "
            "Return JSON only."
        )
        with self.observability.observation(
            "related-question-expansion",
            as_type="generation",
            model=model,
            input_data={"question": question},
        ) as generation:
            raw = self.ollama.chat(
                model=model,
                system_prompt=system_prompt,
                user_prompt=f"Original question:\n{question}",
            )
            self.observability.update_output(
                generation,
                {
                    "related_questions_count": len(raw),
                    "usage": self.ollama.last_usage or {},
                },
            )
        try:
            match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
            parsed = json.loads(match.group(0) if match else raw)
            if isinstance(parsed, dict):
                parsed = parsed.get("questions", [])
            questions = [
                item.strip()
                for item in parsed
                if isinstance(item, str) and item.strip()
            ]
        except (ValueError, TypeError, json.JSONDecodeError):
            questions = []

        unique = []
        seen = {question.strip().casefold()}
        for related in questions:
            key = related.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(related)
            if len(unique) == 3:
                break
        return unique

    def retrieve_sources(
        self,
        question: str,
        related_questions: list[str] | None = None,
        source_limit: int | None = None,
        hybrid: bool | None = None,
    ) -> list[dict]:
        retrieval = self.config["retrieval"]
        limit = int(source_limit or retrieval.get("max_context_chunks", 8))
        rrf_k = float(retrieval.get("rrf_k", 60))
        dedupe_by_source = bool(retrieval.get("dedupe_by_source", False))

        queries = list(dict.fromkeys([question, *(related_questions or [])]))
        # Fuse at the rank level with RRF across the per-query result lists: a chunk
        # ranking #1 in three queries now out-scores one ranking #1 in a single query,
        # and no per-query top_k/dedupe/min_similarity truncation happens beforehand.
        with self.observability.observation(
            "knowledge-base-retrieval",
            as_type="retriever",
            input_data={"query_count": len(queries), "hybrid": hybrid},
        ) as retriever:
            fused: dict[int, dict] = {}
            for query in queries:
                for rank, record in enumerate(
                    self.retrieval_service.rank_candidates(query, hybrid=hybrid), start=1
                ):
                    combined = fused.get(record["id"])
                    if combined is None:
                        combined = dict(record)
                        combined["rrf_score"] = 0.0
                        fused[record["id"]] = combined
                    combined["rrf_score"] += 1.0 / (rrf_k + rank)
                    if record.get("similarity") is not None:
                        current = combined.get("similarity")
                        if current is None or record["similarity"] > current:
                            combined["similarity"] = record["similarity"]

            results = self.retrieval_service.finish_results(
                list(fused.values()), limit=limit, dedupe_by_source=dedupe_by_source
            )
            self.observability.update_output(
                retriever,
                {"source_count": len(results), "candidate_count": len(fused)},
            )
            return results

    def answer(
        self,
        question: str,
        model_name: str,
        history: list[dict] | None = None,
        source_limit: int | None = None,
        related_questions: list[str] | None = None,
        sources: list[dict] | None = None,
    ) -> dict:
        system_prompt, user_prompt, conversation, sources = self._answer_request(
            question=question,
            related_questions=related_questions,
            source_limit=source_limit,
            sources=sources,
            history=history,
        )
        answer = self.ollama.chat(
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=conversation,
        )
        return {"answer": answer, "sources": sources}

    def answer_stream(
        self,
        question: str,
        model_name: str,
        history: list[dict] | None = None,
        source_limit: int | None = None,
        related_questions: list[str] | None = None,
        sources: list[dict] | None = None,
    ) -> Iterator[str]:
        """Yield answer text as the model produces it."""
        system_prompt, user_prompt, conversation, _ = self._answer_request(
            question=question,
            related_questions=related_questions,
            source_limit=source_limit,
            sources=sources,
            history=history,
        )
        answer = ""
        with self.observability.observation(
            "grounded-answer-generation",
            as_type="generation",
            model=model_name,
            input_data={"question": question, "source_count": len(sources or [])},
        ) as generation:
            for delta in self.ollama.chat_stream(
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=conversation,
            ):
                answer += delta
                yield delta
            self.observability.update_output(
                generation,
                {
                    "answer_length": len(answer),
                    "source_count": len(sources or []),
                    "usage": self.ollama.last_usage or {},
                },
            )

    def _answer_request(
        self,
        question: str,
        related_questions: list[str] | None,
        source_limit: int | None,
        sources: list[dict] | None,
        history: list[dict] | None,
    ) -> tuple[str, str, list[dict], list[dict]]:
        sources = sources if sources is not None else self.retrieve_sources(
            question=question,
            related_questions=related_questions,
            source_limit=source_limit,
        )
        context_blocks = []
        for index, source in enumerate(sources, start=1):
            block = (
                f"[S{index}] Title: {source['title']}\n"
                f"Type: {source['source_type']}\n"
                f"URI: {source['canonical_uri'] or 'local-only'}\n"
                f"Excerpt: {source['text']}\n"
            )
            context_blocks.append(block)

        system_prompt = (
            "You answer questions using only the provided knowledge base context. "
            "Be concise, grounded, and cite claims using source labels like [S1]. "
            "If the answer is uncertain, say so clearly."
        )
        context = "\n\n".join(context_blocks) or "No new relevant knowledge base context was retrieved."
        related_context = "\n".join(f"- {item}" for item in (related_questions or []))
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Related search questions used:\n{related_context or '- None'}\n\n"
            f"Current knowledge base context:\n{context}\n\n"
            "Write a grounded answer with inline citations."
        )
        conversation = [
            {"role": "system", "content": system_prompt},
            *[
                {"role": message["role"], "content": message["content"]}
                for message in (history or [])
                if message.get("role") in {"user", "assistant"} and message.get("content")
            ],
            {"role": "user", "content": user_prompt},
        ]
        return system_prompt, user_prompt, conversation, sources
