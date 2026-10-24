from watcher.notion_sync import _already_marked
from unittest.mock import Mock

from local_rag.ingestion import IngestionService


def test_empty_page_is_skipped_and_rechecked_after_content_is_added():
    service = IngestionService.__new__(IngestionService)
    page_id = "3d7fdfd6-8a97-809a-ab1a-f95be55156c9"
    service.database = Mock()
    service.database.find_source_by_uri.return_value = None
    service.preview_notion_data_source = Mock(return_value={
        "page_count": 1, "data_source_id": "mail", "title": "Mail",
        "records": [{"notion_page_id": page_id}],
    })
    preview = {"title": "Blank page", "text": " \n", "metadata": {},
               "canonical_uri": f"notion://page/{page_id}", "raw_text": ""}
    service.preview_notion_page = Mock(return_value=preview)
    service._store_document = Mock(return_value={"status": "ingested", "source_id": 1})
    result = service.ingest_notion_data_source("mail")
    assert result["errors"] == 0
    assert result["skipped"] == 1
    assert result["items"][0]["reason"] == "empty_content"
    service._store_document.assert_not_called()

    preview["text"] = preview["raw_text"] = "Content added later"
    result = service.ingest_notion_data_source("mail")
    assert result["ingested"] == 1
    assert result["skipped"] == result["errors"] == 0
    assert service.preview_notion_page.call_count == 2

    service.preview_notion_page.side_effect = RuntimeError("Notion request failed")
    result = service.ingest_notion_data_source("mail")
    assert result["errors"] == 1
    assert result["skipped"] == 0


def test_already_marked_page_needs_no_notion_write():
    item = {"notion_row": {"Ingested": True, "Tags": ["email", "work"]}}
    assert _already_marked(item, ["email", "work"])


def test_changed_tags_or_unmarked_page_still_needs_write():
    item = {"notion_row": {"Ingested": True, "Tags": ["email", "work"]}}
    assert not _already_marked(item, ["email", "work", "project"])
    assert not _already_marked({"notion_row": {"Ingested": False, "Tags": ["email", "work"]}}, ["email", "work"])
