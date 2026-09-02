from pathlib import Path

from watcher import folder_watcher


class FakeDatabase:
    def __init__(self):
        self.logs = []

    def get_ingestion_log_entry(self, file_path):
        return None

    def record_ingestion_log(self, **kwargs):
        self.logs.append(kwargs)


class FakeIngestion:
    def __init__(self):
        self.calls = []

    def ingest_file(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "ingested", "source_id": 42, "chunk_count": 1, "tags": kwargs["tags"]}


def test_markdown_file_is_ingested_by_onedrive_watcher(tmp_path, monkeypatch):
    path = tmp_path / "meeting-notes.md"
    content = b"# Meeting notes\n\nMarkdown content for the local RAG.\n"
    path.write_bytes(content)
    ingestion = FakeIngestion()
    database = FakeDatabase()

    monkeypatch.setattr(folder_watcher, "wait_for_stable_file", lambda _: content)

    result = folder_watcher.ingest_watched_file(path, ingestion, database)

    assert result["status"] == "ingested"
    assert ingestion.calls[0]["filename"] == "meeting-notes.md"
    assert ingestion.calls[0]["source_type"] == "onedrive_file"
    assert ingestion.calls[0]["tags"] == ["onedrive", "md"]
    assert database.logs[0]["status"] == "ingested"


def test_unsupported_file_types_are_still_ignored(tmp_path):
    path = Path(tmp_path) / "image.png"

    result = folder_watcher.ingest_watched_file(path, FakeIngestion(), FakeDatabase())

    assert result["status"] == "ignored"
    assert result["reason"] == "unsupported_extension"
