from __future__ import annotations

import json
import re
from datetime import datetime, time, timedelta
from typing import Iterator
from zoneinfo import ZoneInfo

from local_rag.embeddings import OllamaClient
from local_rag.observability import LangfuseObserver
from local_rag.system_one import SystemOneAssessment


class ChatService:
    def __init__(self, config_manager, retrieval_service) -> None:
        self.config_manager = config_manager
        self.retrieval_service = retrieval_service
        self.config = config_manager.load_merged()
        self.ollama = OllamaClient(self.config["ollama"])
        self.observability = LangfuseObserver(self.config)
        self.system_one = SystemOneAssessment(self.config.get("system_one", {}))

    def assess_answer(self, question: str, answer: str, sources: list[dict]) -> dict:
        """Assess answer support using the configured local TypeSafe model."""
        return self.system_one.assess(question, answer, sources)

    @property
    def langfuse(self) -> LangfuseObserver:
        return self.observability

    def _calendar_now(self) -> datetime:
        return datetime.now(ZoneInfo("Europe/London"))

    def rewrite_retrieval_query(self, question: str, history: list[dict] | None = None) -> str:
        """Turn a follow-up into a standalone search query using recent context."""
        if not history:
            return question.strip()
        recent = [m["content"] for m in history[-4:] if m.get("role") in {"user", "assistant"} and m.get("content")]
        if not recent:
            return question.strip()
        try:
            rewritten = self.ollama.chat(
                model=self.config["ollama"].get("default_answer_model", "default"),
                system_prompt="Rewrite the latest question as one standalone concise search query. Return only the query.",
                user_prompt="Conversation context:\n" + "\n".join(recent) + f"\n\nLatest question:\n{question}",
            ).strip()
            return rewritten or question.strip()
        except Exception:
            return f"{recent[-1]} {question}".strip()

    def _calendar_date_window(self, question: str) -> tuple[datetime, datetime] | None:
        """Resolve simple calendar date intents without asking the model."""
        normalized = question.casefold()
        now = self._calendar_now().astimezone(ZoneInfo("Europe/London"))
        if re.search(r"\btomorrow\b", normalized):
            day = now.date() + timedelta(days=1)
        elif re.search(r"\btoday\b", normalized):
            day = now.date()
        else:
            match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", normalized)
            if not match:
                return None
            day = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        start = datetime.combine(day, time.min, tzinfo=now.tzinfo)
        return start, start + timedelta(days=1)

    def _exact_calendar_sources(
        self, question: str, source_ids: list[int], limit: int
    ) -> list[dict]:
        window = self._calendar_date_window(question)
        if not window:
            return []
        events = self.retrieval_service.database.search_calendar_events(
            window[0], window[1], source_ids=source_ids
        )
        sources = []
        for event in events[:limit]:
            source = self.retrieval_service.database.get_source(event["source_id"])
            if not source:
                continue
            text = f"Event: {event['event']}\nDescription: {event['description']}".strip()
            sources.append({
                "id": event["source_id"],
                "source_id": event["source_id"],
                "chunk_index": 0,
                "text": text,
                "preview": text[:800],
                "title": source["title"],
                "source_type": source["source_type"],
                "canonical_uri": source["canonical_uri"],
                "tags": source["tags"],
                "metadata": source["metadata"],
                "created_at": source["created_at"],
                "rrf_score": 1_000_000_000.0,
                "similarity": None,
                "lexical_score": None,
                "retrieval_mode": "calendar_exact",
                "calendar_event": event,
            })
        return sources

    def _exact_contact_sources(self, question: str, source_ids: list[int], limit: int | None) -> list[dict]:
        from local_rag.contacts import hydrate_contacts
        database = self.retrieval_service.database
        contacts = database.search_contacts(question, source_ids=source_ids)
        contacts = contacts if limit is None else contacts[:limit]
        sources = []
        for contact in hydrate_contacts(self.config_manager, database, contacts):
            source = database.get_source(contact['source_id'])
            if source is None:
                continue
            text = source.get('full_text') or contact['name']
            sources.append({
                'id': contact['source_id'], 'source_id': contact['source_id'], 'chunk_index': 0,
                'text': text, 'preview': text[:800], 'title': contact['name'],
                'source_type': source['source_type'], 'canonical_uri': source['canonical_uri'],
                'tags': source['tags'], 'metadata': source['metadata'], 'created_at': source['created_at'],
                'rrf_score': 1_000_000_000., 'similarity': None, 'lexical_score': None,
                'retrieval_mode': 'contact_exact', 'contact': contact,
            })
        return sources

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
                    "usage": getattr(self.ollama, "last_usage", None) or {},
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
        include_calendar: bool = True,
        include_knowledge_base: bool = True,
        include_contacts: bool = True,
        history: list[dict] | None = None,
    ) -> list[dict]:
        retrieval = self.config["retrieval"]
        limit = int(source_limit or retrieval.get("max_context_chunks", 8))
        rrf_k = float(retrieval.get("rrf_k", 60))
        dedupe_by_source = bool(retrieval.get("dedupe_by_source", False))

        standalone = self.rewrite_retrieval_query(question, history)
        queries = list(dict.fromkeys([standalone, *(related_questions or [])]))
        database = self.retrieval_service.database
        contact_ids = database.contact_source_ids()
        source_ids = [sid for sid in database.active_source_ids(include_calendar=False) if sid not in contact_ids] if include_knowledge_base else []
        if include_calendar:
            source_ids += database.calendar_source_ids()
        if include_contacts:
            source_ids += contact_ids
        source_ids = list(dict.fromkeys(source_ids))
        if not source_ids:
            return []
        exact_calendar = self._exact_calendar_sources(question, source_ids, limit) if include_calendar else []
        exact_contacts = self._exact_contact_sources(standalone, source_ids, limit) if include_contacts else []
        exact = (exact_contacts + exact_calendar)[:limit]
        # Contact lists come from the complete structured table, independently of top-k retrieval.
        from local_rag.contacts import contact_listing_query
        if include_contacts and contact_listing_query(standalone):
            listing = self._exact_contact_sources(standalone, source_ids, limit=None)
            for source in listing:
                source['contact_listing'] = True
            return listing
        direct_contact_lookup = bool(re.search(r"\b(?:telephone|phone|tel|email|e-mail|address|website|attachment|attachments|contact details)\b", standalone.casefold()))
        if include_contacts and (direct_contact_lookup or (exact_contacts and not include_calendar and not include_knowledge_base)):
            if exact_contacts:
                # Preserve every same-name match so the assistant can ask the user to choose.
                return self._exact_contact_sources(standalone, source_ids, limit=None)
            source_ids = [sid for sid in source_ids if sid not in contact_ids]
            if not source_ids:
                return []
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
                    self.retrieval_service.rank_candidates(
                        query, hybrid=hybrid,
                        source_ids=source_ids,
                    ), start=1
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

            min_similarity = float(retrieval.get("min_similarity", 0.0))
            fused = {
                chunk_id: record
                for chunk_id, record in fused.items()
                if record.get("lexical_score") is not None
                or (record.get("similarity") is not None and record["similarity"] >= min_similarity)
            }

            results = self.retrieval_service.finish_results(
                list(fused.values()), limit=max(limit - len(exact), 0), dedupe_by_source=dedupe_by_source
            )
            exact_source_ids = {result["source_id"] for result in exact}
            results = exact + [
                result for result in results if result["source_id"] not in exact_source_ids
            ]
            self.observability.update_output(
                retriever,
                {"source_count": len(results), "candidate_count": len(fused)},
            )
            return results

    def _contact_listing_answer(self, sources: list[dict] | None) -> str | None:
        if not sources or not all(source.get('contact_listing') for source in sources):
            return None
        lines = [f"{len(sources)} contacts:"]
        for index, source in enumerate(sources, 1):
            contact = source['contact']
            name = contact['name'].replace('[', r'\[').replace(']', r'\]')
            details = [contact[field] for field in ('email', 'tel') if contact[field]]
            lines.append(f"- **{name}**" + (" — " + " · ".join(details) if details else "") + f" [S{index}]")
        return "\n\n".join(lines)

    def answer(
        self,
        question: str,
        model_name: str,
        history: list[dict] | None = None,
        source_limit: int | None = None,
        related_questions: list[str] | None = None,
        sources: list[dict] | None = None,
    ) -> dict:
        listing_answer = self._contact_listing_answer(sources)
        if listing_answer is not None:
            return {'answer': listing_answer, 'sources': sources}
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
        listing_answer = self._contact_listing_answer(sources)
        if listing_answer is not None:
            yield listing_answer
            return
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
                    "usage": getattr(self.ollama, "last_usage", None) or {},
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
            if source.get("calendar_event"):
                event = source["calendar_event"]
                block += (
                    "Structured calendar record (fresh from SQLite):\n"
                    f"Event: {event['event']}\nStart: {event['start_date']}\n"
                    f"End: {event.get('end_date') or 'not specified'}\n"
                    f"Description: {event['description']}\n"
                    f"Location: {event.get('location') or 'not specified'}\n"
                    f"Attendees: {', '.join(event.get('attendees') or []) or 'not specified'}\n"
                )
            if source.get('contact'):
                contact = source['contact']
                fields = {key: contact.get(key) for key in ('name', 'address', 'description', 'email', 'tel', 'url', 'notion_url', 'attachments', 'synced_at')}
                block += "Structured contact record (latest synced SQLite values):\n" + json.dumps(fields, ensure_ascii=False) + "\n"
            context_blocks.append(block)

        today = datetime.now().date().isoformat()
        system_prompt = (
            f"Today's date is {today}. "
            "Use UK English date and time formatting in every answer: format dates as D Mon YYYY "
            "and times using the 24-hour clock, for example ‘22 Sept 2026 13:25’. "
            "Interpret and present calendar times in the Europe/London timezone unless the context "
            "explicitly states another timezone. "
            "You answer questions using only the provided knowledge base context. "
            "Be concise, grounded, and cite claims using source labels like [S1]. "
            "For contacts, use the structured fields for exact phone numbers, email addresses, addresses and URLs. "
            "Never invent missing contact details. If multiple contacts match a name, present the choices and ask which one. "
            "Return attachment URLs as clickable Markdown links; an attachment marked open_in_notion opens its owning "
            "Notion page, so label it accordingly. Attachment contents have not been indexed. "
            "If the answer is uncertain, say so clearly."
        )
        max_chars = int(self.config.get("retrieval", {}).get("max_context_tokens", 6000)) * 4
        context = "\n\n".join(context_blocks)[:max_chars] or "No new relevant knowledge base context was retrieved."
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
