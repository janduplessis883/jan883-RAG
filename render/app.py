"""Render webhook service for archiving received email in Notion."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import hmac
import os
import re
import tempfile
from typing import Any

from bs4 import BeautifulSoup
import httpx
from fastapi import FastAPI, HTTPException, Request
from notionhelper import NotionHelper


RESEND_API_BASE = "https://api.resend.com"
NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = os.getenv("NOTION_VERSION", "2026-03-11")
MAX_NOTION_SINGLE_PART_BYTES = 20 * 1024 * 1024
WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS = 300

app = FastAPI(title="Personal RAG email receiver", version="0.2.0")


def required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def notion_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {required_setting('NOTION_TOKEN')}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def verify_resend_signature(payload: bytes, headers: dict[str, str]) -> None:
    """Verify Resend's Svix signature using the raw request body."""

    secret = required_setting("RESEND_WEBHOOK_SECRET")
    svix_id = headers.get("svix-id")
    timestamp = headers.get("svix-timestamp")
    signatures = headers.get("svix-signature")
    if not svix_id or not timestamp or not signatures:
        raise HTTPException(status_code=400, detail="Missing webhook signature headers")

    try:
        timestamp_seconds = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid webhook timestamp") from exc

    now = int(datetime.now(timezone.utc).timestamp())
    if abs(now - timestamp_seconds) > WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS:
        raise HTTPException(status_code=400, detail="Expired webhook timestamp")

    encoded_secret = secret.removeprefix("whsec_")
    try:
        secret_bytes = base64.b64decode(encoded_secret + "===")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Invalid webhook secret configuration") from exc

    signed_content = f"{svix_id}.{timestamp}.".encode() + payload
    expected = base64.b64encode(
        hmac.new(secret_bytes, signed_content, hashlib.sha256).digest()
    ).decode()
    valid = any(
        hmac.compare_digest(signature.removeprefix("v1,"), expected)
        for signature in signatures.split(" ")
        if signature.startswith("v1,")
    )
    if not valid:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")


def unwrap_data(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload


def clean_email_body(email: dict[str, Any]) -> str:
    text = (email.get("text") or "").strip()
    if not text:
        raw_html = email.get("html") or ""
        soup = BeautifulSoup(raw_html, "html.parser")
        for element in soup(["script", "style", "head"]):
            element.decompose()
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())

    # NHS Mail appends Jan's fixed signature before the forwarded message.
    # When the forwarded From header is present, retaining the message from
    # that header onward is safer than hard-coding a signature name.
    forwarded_from = re.search(r"(?im)^\s*From:\s*.+$", text)
    if forwarded_from:
        text = text[forwarded_from.start() :]

    # Remove the recurring NHS Mail confidentiality notice from all archived
    # messages so it does not become repetitive RAG context.
    footer = re.search(
        r"(?ims)^\s*(?:\*\*)?This message may contain confidential information\.",
        text,
    )
    if footer:
        text = text[: footer.start()]

    # NHS Mail may add one or more long rows of asterisks as a trailing
    # separator, sometimes with whitespace between the asterisks.
    text = re.sub(r"(?m)^\s*(?:\*\s*){20,}$", "", text)
    text = re.sub(r"(?m)^\s*---\s*$", "", text)
    return "\n\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def extract_original_sender(body: str, fallback: str) -> str:
    """Prefer the original sender in a forwarded NHS Mail From header."""

    match = re.search(r"(?im)^\s*From:\s*(.+?)\s*$", body)
    return match.group(1).strip() if match else fallback


def display_addresses(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def rich_text(value: str) -> dict[str, list[dict[str, dict[str, str]]]]:
    if not value:
        return {"rich_text": []}
    return {"rich_text": [{"type": "text", "text": {"content": value[:2000]}}]}


def title_property(value: str) -> dict[str, list[dict[str, dict[str, str]]]]:
    return {"title": [{"type": "text", "text": {"content": value[:2000]}}]}


def paragraph_blocks(text: str) -> list[dict[str, Any]]:
    blocks = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for start in range(0, len(paragraph), 2000):
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": paragraph[start : start + 2000]},
                            }
                        ]
                    },
                }
            )
    return blocks


def is_calendar_email(body: str) -> bool:
    """Return whether the email body starts with the Calendar routing word."""

    return bool(re.match(r"^\s*calendar\b", body, flags=re.IGNORECASE))


async def resend_get(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    response = await client.get(
        f"{RESEND_API_BASE}{path}",
        headers={"Authorization": f"Bearer {required_setting('RESEND_API_KEY')}"},
    )
    return unwrap_data(response)


async def download_attachment(
    client: httpx.AsyncClient,
    attachment: dict[str, Any],
    email_id: str,
) -> tuple[str, str]:
    filename = str(attachment.get("filename") or "attachment")
    content_type = str(attachment.get("content_type") or "application/octet-stream")
    size = int(attachment.get("size") or 0)
    if size > MAX_NOTION_SINGLE_PART_BYTES:
        raise ValueError(f"Attachment is larger than 20 MB: {filename}")

    detail = await resend_get(
        client,
        f"/emails/receiving/{email_id}/attachments/{attachment['id']}",
    )
    download_url = detail.get("download_url")
    if not download_url:
        raise ValueError(f"Resend did not provide a download URL for {filename}")
    download = await client.get(download_url)
    download.raise_for_status()
    content = download.content
    if len(content) > MAX_NOTION_SINGLE_PART_BYTES:
        raise ValueError(f"Attachment is larger than 20 MB: {filename}")

    temporary_file = tempfile.NamedTemporaryFile(
        prefix="resend-attachment-", suffix=f"-{filename}", delete=False
    )
    try:
        temporary_file.write(content)
        return temporary_file.name, filename
    finally:
        temporary_file.close()


async def already_archived(client: httpx.AsyncClient, message_id: str) -> bool:
    response = await client.post(
        f"{NOTION_API_BASE}/data_sources/{required_setting('NOTION_DATA_SOURCE_ID')}/query",
        headers=notion_headers(),
        json={"filter": {"property": "Message ID", "rich_text": {"equals": message_id}}},
    )
    response.raise_for_status()
    return bool(response.json().get("results"))


async def create_notion_page(
    client: httpx.AsyncClient,
    email: dict[str, Any],
) -> dict[str, Any]:
    message_id = str(email.get("message_id") or email.get("email_id") or "")
    subject = str(email.get("subject") or "(no subject)")
    recipients = display_addresses(email.get("to"))
    received_at = str(email.get("created_at") or datetime.now(timezone.utc).isoformat())
    body = clean_email_body(email)
    sender = extract_original_sender(body, display_addresses(email.get("from")))
    content = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Email"}}]},
        },
        *paragraph_blocks(body or "(empty body)"),
    ]
    page = await client.post(
        f"{NOTION_API_BASE}/pages",
        headers=notion_headers(),
        json={
            "parent": {"type": "data_source_id", "data_source_id": required_setting("NOTION_DATA_SOURCE_ID")},
            "properties": {
                "Subject": title_property(subject),
                "Sender": rich_text(sender),
                "Recipients": rich_text(recipients),
                "Date Received": {"date": {"start": received_at}},
                "Tags": {"multi_select": [{"name": "email"}, {"name": "work"}]},
                "Attachments": {"files": []},
                "Message ID": rich_text(message_id),
                "Source": {"url": f"https://resend.com/emails/receiving/{email.get('email_id', '')}"},
            },
            "children": content[:100],
        },
    )
    page.raise_for_status()
    result = page.json()
    remaining = content[100:]
    while remaining:
        batch, remaining = remaining[:100], remaining[100:]
        append = await client.patch(
            f"{NOTION_API_BASE}/blocks/{result['id']}/children",
            headers=notion_headers(),
            json={"children": batch},
        )
        append.raise_for_status()
    return result


async def create_calendar_notion_page(
    client: httpx.AsyncClient,
    email: dict[str, Any],
    body: str,
) -> dict[str, Any]:
    """Create a minimal Calendar page without processing email attachments."""

    subject = str(email.get("subject") or "(no subject)")
    content = paragraph_blocks(body or "(empty body)")
    page = await client.post(
        f"{NOTION_API_BASE}/pages",
        headers=notion_headers(),
        json={
            "parent": {
                "type": "data_source_id",
                "data_source_id": required_setting("NOTION_CALENDAR_DATA_SOURCE_ID"),
            },
            "properties": {
                "Event": title_property(subject),
                "Date": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
            },
            "children": content[:100],
        },
    )
    page.raise_for_status()
    result = page.json()
    remaining = content[100:]
    while remaining:
        batch, remaining = remaining[:100], remaining[100:]
        append = await client.patch(
            f"{NOTION_API_BASE}/blocks/{result['id']}/children",
            headers=notion_headers(),
            json={"children": batch},
        )
        append.raise_for_status()
    return result


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhooks/resend")
async def receive_resend_webhook(request: Request) -> dict[str, Any]:
    payload = await request.body()
    verify_resend_signature(payload, {key.lower(): value for key, value in request.headers.items()})
    event = await request.json()
    if event.get("type") != "email.received":
        return {"status": "ignored"}

    event_data = event.get("data") or {}
    email_id = event_data.get("email_id")
    if not email_id:
        raise HTTPException(status_code=400, detail="email.received event has no email_id")

    async with httpx.AsyncClient(timeout=60) as client:
        email = await resend_get(client, f"/emails/receiving/{email_id}")
        message_id = str(email.get("message_id") or email_id)
        body = clean_email_body(email)
        if is_calendar_email(body):
            page = await create_calendar_notion_page(client, email, body)
            return {
                "status": "calendar_created",
                "message_id": message_id,
                "notion_page_id": page["id"],
            }

        if await already_archived(client, message_id):
            return {"status": "duplicate", "message_id": message_id}

        attachments = await resend_get(client, f"/emails/receiving/{email_id}/attachments")
        page = await create_notion_page(client, email)
        notion = NotionHelper(required_setting("NOTION_TOKEN"), request_timeout=60)
        notion_files = []
        for attachment in attachments.get("data", attachments.get("attachments", [])):
            temporary_path, filename = await download_attachment(client, attachment, email_id)
            try:
                upload = await asyncio.to_thread(notion.upload_file, temporary_path)
                notion_files.append(
                    {
                        "type": "file_upload",
                        "file_upload": {"id": str(upload["id"])},
                        "name": filename,
                    }
                )
            finally:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

        if notion_files:
            attachment_update = await client.patch(
                f"{NOTION_API_BASE}/pages/{page['id']}",
                headers=notion_headers(),
                json={"properties": {"Attachments": {"files": notion_files}}},
            )
            attachment_update.raise_for_status()

    return {"status": "archived", "message_id": message_id, "notion_page_id": page["id"]}
