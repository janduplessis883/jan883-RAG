from datetime import datetime, timezone
import json

from local_rag.calendar import CalendarEvent, NotionCalendarService


def test_calendar_event_extracts_normalized_properties_and_skips_missing_start():
    page = {
        "id": "25a73f97-6d2f-4d6f-99ac-ec5ae2454f5c",
        "last_edited_time": "2026-09-10T08:56:00.000Z",
        "created_time": "2026-09-10T08:55:00.000Z",
        "properties": {
            "Event": {"type": "title", "title": [{"plain_text": "Project kickoff"}]},
            "Description": {"type": "rich_text", "rich_text": [{"plain_text": "Agenda"}]},
            "Date": {"type": "date", "date": {"start": "2026-09-14T10:00:00+01:00", "end": "2026-09-14T11:00:00+01:00", "time_zone": "Europe/London"}},
            "Attendees": {"type": "email", "email": "alice@example.com"},
            "Location": {"type": "place", "place": {"name": "Room 4"}},
            "Tag": {"type": "multi_select", "multi_select": [{"name": "Work"}]},
            "Completed": {"type": "checkbox", "checkbox": False},
            "Teams Link": {"type": "url", "url": "https://teams.example/1"},
            "RAG ID": {"type": "rich_text", "rich_text": []},
        },
    }
    event = CalendarEvent.from_notion_page(page)
    assert event is not None
    assert event.page_id == "25a73f976d2f4d6f99acec5ae2454f5c"
    assert event.event == "Project kickoff"
    assert event.description == "Agenda"
    assert event.start_date == "2026-09-14T10:00:00+01:00"
    assert event.end_date == "2026-09-14T11:00:00+01:00"
    assert event.timezone == "Europe/London"
    assert event.attendees == ["alice@example.com"]
    assert event.location == "Room 4"
    assert event.tags == ["Work"]

    page["properties"]["Date"]["date"] = None
    assert CalendarEvent.from_notion_page(page) is None


def test_calendar_service_upserts_recent_events_and_deletes_missing_sources(tmp_path, monkeypatch):
    class Config:
        def load_merged(self):
            return {"app": {"database_path": str(tmp_path / "rag.sqlite3")}, "ollama": {"embedding_dimensions": 3}, "chunking": {}, "notion": {}}
        def resolve_path(self, value):
            from pathlib import Path
            return Path(value)

    from local_rag.database import Database

    db = Database(Config())
    db.initialize()
    class FakeIngestion:
        def ingest_text(self, **kwargs):
            source_id = db.insert_source(source_type=kwargs["source_type"], title=kwargs["title"], canonical_uri=kwargs["canonical_uri"],
                external_ref=kwargs["external_ref"], content_hash=db.hash_text(kwargs["text"]), summary=kwargs["text"], full_text=kwargs["text"],
                tags=kwargs["tags"], metadata=kwargs["metadata"], raw_text_path=None, raw_binary_path=None)
            return {"source_id": source_id}
    service = NotionCalendarService(Config(), db, FakeIngestion())
    monkeypatch.setattr(service, "_write_rag_id", lambda event, rag_id: None)
    monkeypatch.setattr(service, "fetch_all_pages", lambda: [
        {"id": "25a73f97-6d2f-4d6f-99ac-ec5ae2454f5c", "last_edited_time": "2026-09-10T08:00:00Z", "properties": {
            "Event": {"type": "title", "title": [{"plain_text": "Keep"}]},
            "Description": {"type": "rich_text", "rich_text": [{"plain_text": "Details"}]},
            "Date": {"type": "date", "date": {"start": "2026-09-10", "end": None, "time_zone": None}},
        }},
    ])
    result = service.sync(full_sync=False, now=datetime(2026, 9, 10, tzinfo=timezone.utc))
    assert result["upserted"] == 1
    assert db.list_calendar_events()[0]["event"] == "Keep"

    monkeypatch.setattr(service, "fetch_all_pages", lambda: [])
    result = service.sync(full_sync=False, now=datetime(2026, 9, 10, tzinfo=timezone.utc))
    assert result["deleted"] == 1
    assert db.list_calendar_events() == []


def test_calendar_events_with_identical_text_keep_distinct_sources(tmp_path):
    class Config:
        def load_merged(self):
            return {"app": {"database_path": str(tmp_path / "rag.sqlite3")}, "ollama": {"embedding_dimensions": 3}, "chunking": {}, "notion": {}}
        def resolve_path(self, value):
            from pathlib import Path
            return Path(value)

    from local_rag.database import Database

    db = Database(Config())
    db.initialize()
    class FakeIngestion:
        def ingest_text(self, **kwargs):
            content_hash = db.hash_text(kwargs["text"] + (kwargs["canonical_uri"] if kwargs.get("allow_duplicate_content") else ""))
            source_id = db.insert_source(source_type="notion_calendar", title=kwargs["title"], canonical_uri=kwargs["canonical_uri"],
                external_ref=kwargs["external_ref"], content_hash=content_hash, summary=kwargs["text"], full_text=kwargs["text"],
                tags=kwargs["tags"], metadata=kwargs["metadata"], raw_text_path=None, raw_binary_path=None)
            return {"source_id": source_id}

    service = NotionCalendarService(Config(), db, FakeIngestion())
    service._write_rag_id = lambda event, rag_id: None
    pages = []
    for page_id in ("25a73f97-6d2f-4d6f-99ac-ec5ae2454f5c", "35a73f97-6d2f-4d6f-99ac-ec5ae2454f5c"):
        pages.append({"id": page_id, "properties": {
            "Event": {"type": "title", "title": [{"plain_text": "Same event"}]},
            "Description": {"type": "rich_text", "rich_text": [{"plain_text": "Same details"}]},
            "Date": {"type": "date", "date": {"start": "2026-09-10", "end": None, "time_zone": None}},
        }})
    service.fetch_all_pages = lambda: pages
    result = service.sync(now=datetime(2026, 9, 10, tzinfo=timezone.utc))
    assert result["errors"] == 0
    assert len({row["source_id"] for row in db.list_calendar_events()}) == 2
