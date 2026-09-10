from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from typing import Any, Callable

import requests


CALENDAR_DATA_SOURCE_ID = "303fdfd68a9780f6bbcc000b5fe219ab"
NOTION_API_VERSION = "2025-09-03"
CALENDAR_TZ = "Europe/London"


def _plain(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    return "".join(str(item.get("plain_text", item.get("text", {}).get("content", ""))) for item in items if isinstance(item, dict)).strip()


def _prop(properties: dict, name: str, default=None):
    value = properties.get(name) or properties.get(name.lower()) or {}
    kind = value.get("type")
    if kind in value:
        return value.get(kind)
    return default


@dataclass
class CalendarEvent:
    page_id: str
    event: str
    description: str
    start_date: str
    end_date: str | None
    timezone: str | None
    attendees: list[str]
    location: str | None
    tags: list[str]
    completed: bool
    blocked_by: list[dict]
    blocking: list[dict]
    teams_link: str | None
    created_time: str | None
    last_edited_time: str | None
    created_by: dict | None
    last_edited_by: dict | None
    rag_id: str | None
    raw_json: dict

    @classmethod
    def from_notion_page(cls, page: dict) -> "CalendarEvent | None":
        properties = page.get("properties", {})
        date_value = _prop(properties, "Date")
        if not isinstance(date_value, dict) or not date_value.get("start"):
            return None
        attendees = _prop(properties, "Attendees")
        if isinstance(attendees, str):
            attendees = [attendees]
        elif not isinstance(attendees, list):
            attendees = []
        tags = _prop(properties, "Tag", []) or []
        tags = [item.get("name", "") if isinstance(item, dict) else str(item) for item in tags]
        place = _prop(properties, "Location")
        location = place.get("name") if isinstance(place, dict) else place
        return cls(
            page_id=page["id"].replace("-", ""), event=_plain(_prop(properties, "Event", [])),
            description=_plain(_prop(properties, "Description", [])), start_date=date_value["start"],
            end_date=date_value.get("end"), timezone=date_value.get("time_zone") or CALENDAR_TZ,
            attendees=[item for item in attendees if item], location=location,
            tags=[item for item in tags if item], completed=bool(_prop(properties, "Completed", False)),
            blocked_by=_prop(properties, "Blocked by", []) or [], blocking=_prop(properties, "Blocking", []) or [],
            teams_link=_prop(properties, "Teams Link"), created_time=_prop(properties, "Created time") or page.get("created_time"),
            last_edited_time=_prop(properties, "Last edited time") or page.get("last_edited_time"),
            created_by=_prop(properties, "Created by") or page.get("created_by"),
            last_edited_by=_prop(properties, "Last edited by") or page.get("last_edited_by"),
            rag_id=_plain(_prop(properties, "RAG ID", [])) or None, raw_json=page,
        )

    def embedding_text(self) -> str:
        return f"Event: {self.event}\nDescription: {self.description}".strip()


class NotionCalendarService:
    def __init__(self, config_manager, database, ingestion=None) -> None:
        self.config_manager = config_manager
        self.config = config_manager.load_merged()
        self.database = database
        self.ingestion = ingestion
        notion = self.config.get("notion", {})
        self.token = notion.get("api_token")
        self.timeout = float(notion.get("request_timeout_seconds", 30))
        self.data_source_id = notion.get("calendar_data_source_id", CALENDAR_DATA_SOURCE_ID)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        response = requests.request(method, f"https://api.notion.com/v1{path}", headers={
            "Authorization": f"Bearer {self.token}", "Notion-Version": NOTION_API_VERSION,
            "Content-Type": "application/json",
        }, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def fetch_all_pages(self) -> list[dict]:
        pages, cursor = [], None
        while True:
            payload = {"page_size": 100}
            if cursor:
                payload["start_cursor"] = cursor
            response = self._request("POST", f"/data_sources/{self.data_source_id}/query", payload)
            pages.extend(response.get("results", []))
            if not response.get("has_more"):
                return pages
            cursor = response.get("next_cursor")

    def _write_rag_id(self, event: CalendarEvent, rag_id: int) -> None:
        self._request("PATCH", f"/pages/{event.page_id}", {"properties": {
            "RAG ID": {"rich_text": [{"text": {"content": str(rag_id)}}]}
        }})

    def sync(self, *, full_sync: bool = False, now: datetime | None = None,
             progress_callback: Callable[[int, int, str], None] | None = None) -> dict:
        now = now or datetime.now().astimezone()
        cutoff = now - timedelta(days=28)
        pages = self.fetch_all_pages()
        seen_ids = set()
        upserted = skipped = deleted = errors = 0
        error_details = []
        candidates = [page for page in pages if CalendarEvent.from_notion_page(page) is not None]
        total = len(candidates)
        for index, page in enumerate(candidates, 1):
            event = CalendarEvent.from_notion_page(page)
            seen_ids.add(event.page_id)
            start = datetime.fromisoformat(event.start_date.replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=now.tzinfo)
            if not full_sync and start < cutoff:
                skipped += 1
                continue
            try:
                self._upsert_event(event)
                upserted += 1
            except Exception as exc:
                errors += 1
                error_details.append({"page_id": event.page_id, "event": event.event, "error": str(exc)})
            if progress_callback:
                progress_callback(index, total, event.event or event.page_id)
        for row in self.database.calendar_page_ids():
            if row not in seen_ids:
                try:
                    self.database.delete_calendar_event(row)
                    deleted += 1
                except Exception as exc:
                    errors += 1
                    error_details.append({"page_id": row, "operation": "delete", "error": str(exc)})
        return {"status": "ok", "page_count": len(pages), "upserted": upserted, "skipped": skipped,
                "deleted": deleted, "errors": errors, "error_details": error_details[:50]}

    def _upsert_event(self, event: CalendarEvent) -> None:
        text = event.embedding_text()
        if not text:
            text = event.event or event.description or "Calendar event"
        canonical_uri = f"notion://calendar/{event.page_id}"
        existing = self.database.find_source_by_uri(canonical_uri)
        metadata = {"notion_page_id": event.page_id, "notion_data_source_id": self.data_source_id, "calendar": True}
        if existing:
            source_id = int(existing["id"])
            if existing["content_hash"] != self.database.hash_text(text):
                self.ingestion.update_source(source_id, title=event.event, text=text, tags=event.tags)
            current = self.database.get_source(source_id) or {}
            metadata = {**current.get("metadata", {}), **metadata}
            self.database.update_source_metadata(source_id, metadata)
        else:
            result = self.ingestion.ingest_text(title=event.event or "Calendar event", text=text, tags=event.tags,
                source_type="notion_calendar", canonical_uri=canonical_uri,
                external_ref=f"notion://data-source/{self.data_source_id}", metadata=metadata,
                allow_duplicate_content=True)
            source_id = int(result["source_id"])
        self.database.upsert_calendar_event(event, source_id)
        self._write_rag_id(event, source_id)
