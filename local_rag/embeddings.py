from __future__ import annotations

import json
import re
from typing import Any, Iterator

import requests


class OllamaClient:
    def __init__(self, config: dict) -> None:
        self.base_url = config["base_url"].rstrip("/")
        self.api_key = config.get("api_key", "")
        self.embedding_model = config["embedding_model"]
        self.timeout = int(config["request_timeout_seconds"])
        self.embed_batch_size = int(config.get("embed_batch_size", 64))
        self.last_usage: dict[str, Any] | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts if text.strip()]
        if not cleaned:
            return []

        results: list[list[float]] = []
        for start in range(0, len(cleaned), self.embed_batch_size):
            results.extend(self._embed_batch(cleaned[start : start + self.embed_batch_size]))
        return results

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json={"model": self.embedding_model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        self.last_usage = payload.get("usage")
        return [item["embedding"] for item in payload.get("data", [])]

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        request_messages = messages or [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": model,
                "messages": request_messages,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        self.last_usage = payload.get("usage")
        return payload["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        messages: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        """Yield assistant text deltas from an OpenAI-compatible SSE response."""
        request_messages = messages or [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.last_usage = None
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": model,
                "messages": request_messages,
                "stream": True,
            },
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                line = line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    payload: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = payload.get("choices", [])
                if not choices:
                    if payload.get("usage") is not None:
                        self.last_usage = payload["usage"]
                    continue
                if payload.get("usage") is not None:
                    self.last_usage = payload["usage"]
                delta = choices[0].get("delta", {})
                content = delta.get("content", "") if isinstance(delta, dict) else ""
                if content:
                    yield content
        finally:
            response.close()

    def is_meaningful_chunk(self, text: str, model: str) -> bool:
        """Ask the chat model whether a chunk is useful for knowledge retrieval."""
        system_prompt = (
            "Classify text for a semantic search index. Keep text that contains meaningful "
            "facts, instructions, decisions, questions, answers, transcript content, code, "
            "or structured information. Reject empty text, navigation, repeated boilerplate, "
            "isolated markup, and obvious extraction noise. Return JSON only: "
            '{"keep": true} or {"keep": false}. Do not explain.'
        )
        user_prompt = f"Text to classify:\n\n{text}"
        try:
            raw = self.chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
            match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
            payload = json.loads(match.group(0) if match else raw)
            return bool(payload.get("keep", True))
        except Exception:  # noqa: BLE001 - quality filtering should fail open
            return True

    def list_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        models = payload.get("data", [])
        names = [str(model.get("id")) for model in models if isinstance(model, dict) and model.get("id")]
        return sorted(names, key=str.lower)
