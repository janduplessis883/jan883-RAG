"""Render entrypoint for the email-ingestion service.

The endpoint is kept intentionally small until the Resend and Notion API
credentials are configured in Render. The full parsing and page creation flow
will be added here without coupling it to the local RAG runtime.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException, Request


app = FastAPI(title="Personal RAG email receiver", version="0.1.0")


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhooks/resend")
async def receive_resend_webhook(
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
) -> dict[str, str]:
    """Receive a Resend event while the parser implementation is completed."""

    expected_secret = os.getenv("WEBHOOK_SECRET")
    if expected_secret and x_webhook_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    event = await request.json()
    if event.get("type") != "email.received":
        return {"status": "ignored"}

    # Do not silently discard work emails: fail loudly until the downstream
    # Resend retrieval and Notion page creation flow is enabled.
    raise HTTPException(
        status_code=503,
        detail="Email parser is not enabled yet; webhook configuration is not ready.",
    )
