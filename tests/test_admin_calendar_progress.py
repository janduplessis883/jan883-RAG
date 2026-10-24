"""Exercise the real Admin page across reruns without external services."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch
import fcntl

from streamlit.testing.v1 import AppTest

from local_rag.config import ConfigManager
from local_rag.calendar import NotionCalendarService
from local_rag.database import Database
from local_rag.ingestion import IngestionService
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


class AdminCalendarProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(TemporaryDirectory())
        config = ConfigManager(Path(temporary))
        config.ensure_defaults()
        self.runtime = (config, object(), object(), None, None, object())

    def start(self, sync):
        self.enterContext(patch("watcher.notion_sync.sync_calendar_once", side_effect=sync))
        app = AppTest.from_file(ROOT / "app_pages/admin.py", default_timeout=10)
        app.session_state["runtime"] = self.runtime
        app.run()
        next(button for button in app.button if button.label == "Calendar master sync").click().run()
        return app

    def test_progress_and_result_survive_full_page_reruns(self):
        advanced, release = Event(), Event()
        calls = []

        def sync(config, database, ingestion, *, full_sync, progress_callback):
            calls.append((config, database, ingestion, full_sync))
            progress_callback(1, 3, "Processing second event")
            advanced.set()
            if not release.wait(10):
                raise TimeoutError("Test did not release sync")
            progress_callback(3, 3, "Finished")
            return {"status": "ok", "upserted": 3, "deleted": 0, "errors": 0}

        app = self.start(sync)
        try:
            self.assertTrue(advanced.wait(5))
            app.run()
            self.assertFalse(app.exception)
            progress = app.get("progress")[0].proto
            self.assertEqual(progress.value, 33)
            self.assertIn("2 events remaining", progress.text)
            self.assertIn("Processing second event", progress.text)
            self.assertEqual(calls, [(*self.runtime[:3], True)])
            self.assertTrue(next(b for b in app.button if b.label == "Calendar master sync").disabled)
        finally:
            release.set()
            app.session_state["calendar_sync_future"].result(timeout=5)
        for _ in range(2):
            app.run()
            self.assertFalse(app.exception)
            self.assertIn("3 stored", app.success[0].value)

    def test_lock_contention_displays_skipped_message(self):
        app = self.start(lambda *args, **kwargs: {
            "status": "skipped", "message": "Another calendar sync is already running."
        })
        if "calendar_sync_future" in app.session_state:
            app.session_state["calendar_sync_future"].result(timeout=5)
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("already running", app.info[0].value)

    def test_background_error_remains_visible(self):
        app = self.start(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Notion unavailable")))
        if "calendar_sync_future" in app.session_state:
            app.session_state["calendar_sync_future"].result(timeout=5)
        for _ in range(2):
            app.run()
            self.assertFalse(app.exception)
            self.assertIn("Notion unavailable", app.error[0].value)


class CalendarMasterPathTests(unittest.TestCase):
    def test_wrapper_passes_full_sync_and_progress_and_respects_lock(self):
        from watcher.notion_sync import sync_calendar_once

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            with patch("watcher.notion_sync.PROJECT_ROOT", root), patch("watcher.notion_sync.NotionCalendarService") as service:
                database, config, ingestion, progress = Mock(), Mock(), Mock(), Mock()
                with (root / "data/notion_calendar_sync.lock").open("w") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    result = sync_calendar_once(config, database, ingestion, full_sync=True, progress_callback=progress)
                    self.assertEqual(result["status"], "skipped")
                    service.assert_not_called()
                service.return_value.sync.return_value = {"status": "ok", "upserted": 0, "deleted": 0, "errors": 0}
                result = sync_calendar_once(config, database, ingestion, full_sync=True, progress_callback=progress)
                service.assert_called_once_with(config, database, ingestion)
                service.return_value.sync.assert_called_once_with(full_sync=True, progress_callback=progress)
                database.record_operation.assert_called_once_with("sync:notion_calendar", result)

    def test_identical_events_can_be_synced_and_updated_repeatedly(self):
        with TemporaryDirectory() as directory:
            config = ConfigManager(Path(directory))
            config.ensure_defaults()
            settings = config.read_text(config.settings_path).replace("embedding_dimensions = 768", "embedding_dimensions = 3")
            config.write_text(config.settings_path, settings)
            database = Database(config)
            database.initialize()
            self.addCleanup(database.connection.close)
            ingestion = IngestionService(config, database)
            self.enterContext(patch.object(ingestion.ollama, "embed_texts", side_effect=lambda texts: [[1., 0., 0.] for _ in texts]))
            service = NotionCalendarService(config, database, ingestion)
            pages = [{"id": page_id, "properties": {
                "Event": {"type": "title", "title": [{"plain_text": "Recurring event"}]},
                "Date": {"type": "date", "date": {"start": "2026-10-01"}},
            }} for page_id in ("event-a", "event-b")]
            self.enterContext(patch.object(service, "fetch_all_pages", return_value=pages))
            self.enterContext(patch.object(service, "_write_rag_id"))
            now = datetime(2026, 10, 1, tzinfo=timezone.utc)
            for _ in range(2):
                progress = []
                result = service.sync(full_sync=True, now=now, progress_callback=lambda *args: progress.append(args))
                self.assertEqual(result["errors"], 0, result["error_details"])
                self.assertEqual(result["upserted"], 2)
                self.assertEqual([p[0] for p in progress], sorted(p[0] for p in progress))
                self.assertEqual(progress[-1][:2], (2, 2))
            for page in pages:
                page["properties"]["Event"]["title"][0]["plain_text"] = "Updated recurring event"
            result = service.sync(full_sync=True, now=now)
            self.assertEqual(result["errors"], 0, result["error_details"])
            self.assertEqual(len(database.list_calendar_events()), 2)


if __name__ == "__main__":
    unittest.main()
