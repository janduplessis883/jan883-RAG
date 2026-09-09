"""Local HTTP API for the personal RAG services."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from local_rag.chat import ChatService
from local_rag.config import ConfigManager
from local_rag.database import Database
from local_rag.retrieval import SearchService


ROOT_DIR = Path(__file__).resolve().parent


def build_runtime(root_dir: Path = ROOT_DIR) -> tuple[ConfigManager, Database, SearchService, ChatService]:
    config = ConfigManager(root_dir)
    config.ensure_defaults()
    database = Database(config)
    database.initialize()
    retrieval = SearchService(config, database)
    chat = ChatService(config, retrieval)
    return config, database, retrieval, chat


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=50)
    hybrid: bool | None = None
    source_ids: list[int] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    model: str | None = None
    source_limit: int = Field(default=8, ge=1, le=50)
    multi_query: bool = True
    hybrid: bool | None = None
    history: list[ChatMessage] = Field(default_factory=list)


def create_app(
    services: tuple[ConfigManager, Database, SearchService, ChatService] | None = None,
) -> FastAPI:
    app = FastAPI(title="Personal Local RAG API", version="0.1.0")
    app.state.services = services or build_runtime()

    @app.get("/health")
    def health(request: Request) -> dict:
        _, database, _, chat = request.app.state.services
        omlx = {"available": False, "models": []}
        try:
            omlx["models"] = chat.ollama.list_models()
            omlx["available"] = True
        except Exception as exc:  # noqa: BLE001 - health should report dependency failure
            omlx["error"] = str(exc)

        return {
            "status": "ok",
            "database": database.get_stats(),
            "omlx": omlx,
        }

    @app.post("/search")
    def search(payload: SearchRequest, request: Request) -> dict:
        _, _, retrieval, _ = request.app.state.services
        try:
            results = retrieval.search(
                query=payload.query,
                limit=payload.limit,
                hybrid=payload.hybrid,
                source_ids=payload.source_ids,
            )
        except Exception as exc:  # noqa: BLE001 - expose a stable API error
            raise HTTPException(status_code=503, detail=f"Search unavailable: {exc}") from exc
        return {"query": payload.query, "results": results}

    @app.post("/chat")
    def chat(payload: ChatRequest, request: Request) -> dict:
        _, _, _, chat_service = request.app.state.services
        model = payload.model or chat_service.config["ollama"]["default_answer_model"]
        history = [message.model_dump() for message in payload.history]

        try:
            related_questions = (
                chat_service.generate_related_questions(payload.question)
                if payload.multi_query
                else []
            )
            sources = chat_service.retrieve_sources(
                question=payload.question,
                related_questions=related_questions,
                source_limit=payload.source_limit,
                hybrid=payload.hybrid,
            )
            if sources:
                answer = chat_service.answer(
                    question=payload.question,
                    model_name=model,
                    history=history,
                    source_limit=payload.source_limit,
                    related_questions=related_questions,
                    sources=sources,
                )["answer"]
            else:
                answer = (
                    "I couldn’t find supporting passages in your library. "
                    "Try a more specific question or add relevant documents."
                )
        except Exception as exc:  # noqa: BLE001 - expose a stable API error
            raise HTTPException(status_code=503, detail=f"Chat unavailable: {exc}") from exc

        return {
            "question": payload.question,
            "answer": answer,
            "model": model,
            "related_questions": related_questions,
            "sources": sources,
        }

    return app


app = create_app()
