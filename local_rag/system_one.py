from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any


EVIDENCE_QUALITY_CRITERIA = ["Poor", "Limited", "Good", "Strong"]


def _json_ready(value: Any) -> Any:
    """Convert SDK response objects into regular JSON-compatible values."""
    if hasattr(value, "model_dump"):
        return _json_ready(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    if hasattr(value, "dict") and callable(value.dict):
        return _json_ready(value.dict())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_ready(vars(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class SystemOneAssessment:
    """Run a small, structured evidence assessment with the local TypeSafe model."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.timeout_seconds = float(settings.get("timeout_seconds", 120))
        self.settings = settings

    def assess(self, question: str, answer: str, sources: list[dict]) -> dict[str, Any]:
        env_settings = {
            "TYPESAFE_BASE_URL": self.settings.get("base_url"),
            "TYPESAFE_API_KEY": self.settings.get("api_key"),
            "TYPESAFE_DEFAULT_MODEL": self.settings.get("default_model"),
        }
        for name, value in env_settings.items():
            if value:
                os.environ[name] = str(value)

        # Import lazily so deployments that do not use the optional assessment
        # feature can still import and run the core RAG application.
        from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

        evidence = "\n\n".join(
            f"[S{index}] {source.get('title', 'Untitled source')}\n{source.get('text', '')}"
            for index, source in enumerate(sources, start=1)
        )
        state = {
            "question": question,
            "draft_answer": answer,
            "retrieved_evidence": evidence,
        }
        questions = {
            "support_level": Choice(
                instructions=(
                    "How well does the retrieved evidence support the draft answer? "
                    "Judge only the evidence provided, not outside knowledge."
                ),
                criteria={
                    "supported": "The main claims are directly supported by the passages.",
                    "partially_supported": "Some claims are supported, but important details are missing or uncertain.",
                    "unsupported": "The passages do not support the main claims, or contradict them.",
                },
            ),
            "direct_support": Noul(
                instructions=(
                    "Do the retrieved passages directly support the draft answer to the question? "
                    "Use a high probability only when the evidence explicitly supports its main claims."
                ),
            ),
            "evidence_quality": Score(
                instructions="How useful and relevant are these retrieved passages for answering the question?",
                criteria=EVIDENCE_QUALITY_CRITERIA,
            ),
        }

        with TypeSafeClient(timeout=self.timeout_seconds) as client:
            result = client.system_one(state=state, questions=questions)

        raw_output = {
            "choices": _json_ready(result.choices),
            "nouls": _json_ready(result.nouls),
            "scores": _json_ready(result.scores),
        }
        for field in ("model", "usage"):
            if hasattr(result, field):
                raw_output[field] = _json_ready(getattr(result, field))

        return {
            "support_level": result.choices["support_level"].choice,
            "direct_support_probability": float(result.nouls["direct_support"].noul),
            "evidence_quality": float(result.scores["evidence_quality"].score),
            "evidence_quality_max": len(EVIDENCE_QUALITY_CRITERIA) - 1,
            "raw_output": raw_output,
        }
