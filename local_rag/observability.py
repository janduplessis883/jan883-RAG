from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Iterator


class LangfuseObserver:
    """Small, failure-tolerant Langfuse adapter for the local RAG workflow."""

    def __init__(self, config: dict) -> None:
        settings = config.get("langfuse", {})
        self.capture_content = bool(settings.get("capture_content", False))
        self.last_trace_url: str | None = None
        self.client = None

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", settings.get("public_key", ""))
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", settings.get("secret_key", ""))
        base_url = os.getenv(
            "LANGFUSE_BASE_URL",
            settings.get("base_url", "http://localhost:3000"),
        ).rstrip("/")
        if not public_key or not secret_key:
            return

        # Langfuse reads configuration when its client is created. Keep this import
        # here so a missing optional dependency never prevents the app from starting.
        try:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
            # Python SDK v4 documents LANGFUSE_BASE_URL for self-hosted instances.
            # Keep LANGFUSE_HOST as a compatibility alias for older deployments.
            os.environ.setdefault("LANGFUSE_BASE_URL", base_url)
            os.environ.setdefault("LANGFUSE_HOST", base_url)
            os.environ.setdefault(
                "LANGFUSE_TRACING_ENVIRONMENT",
                str(settings.get("environment", "local")),
            )
            from langfuse import get_client

            self.client = get_client()
        except Exception:  # noqa: BLE001 - observability must never break the app
            self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _input(self, value: Any) -> Any:
        return value if self.capture_content else {"content_captured": False}

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        session_id: str,
        input_data: dict[str, Any],
        tags: list[str] | None = None,
    ) -> Iterator[Any]:
        if not self.client:
            yield None
            return

        try:
            from langfuse import propagate_attributes
            attributes = propagate_attributes(
                session_id=session_id,
                tags=tags or [],
                metadata={"content_captured": self.capture_content},
            )
            observation_context = self.client.start_as_current_observation(
                as_type="span",
                name=name,
                input=self._input(input_data),
            )
        except Exception:
            # If Langfuse is unavailable or changes incompatibly, preserve RAG use.
            yield None
            return

        with attributes:
            with observation_context as observation:
                try:
                    yield observation
                except Exception as exc:
                    observation.update(level="ERROR", status_message=str(exc)[:500])
                    raise
                finally:
                    trace_id = self.client.get_current_trace_id()
                    if trace_id:
                        self.last_trace_url = self.client.get_trace_url(trace_id=trace_id)

    @contextmanager
    def observation(
        self,
        name: str,
        *,
        as_type: str,
        input_data: dict[str, Any],
        model: str | None = None,
    ) -> Iterator[Any]:
        if not self.client:
            yield None
            return

        try:
            kwargs = {
                "as_type": as_type,
                "name": name,
                "input": self._input(input_data),
            }
            if model:
                kwargs["model"] = model
        except Exception:
            yield None
            return

        with self.client.start_as_current_observation(**kwargs) as observation:
            yield observation

    def update_output(self, observation: Any, output: dict[str, Any]) -> None:
        if observation is not None:
            try:
                observation.update(output=output)
            except Exception:
                pass

    def flush(self) -> None:
        if self.client:
            try:
                self.client.flush()
            except Exception:
                pass
