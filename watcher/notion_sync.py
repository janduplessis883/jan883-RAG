"""Periodically ingest curated Notion email pages into the local RAG database."""

from __future__ import annotations

from pathlib import Path
import sys
import time

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from notionhelper import NotionHelper

# Support direct execution with `python watcher/notion_sync.py` from any
# working directory, as well as module execution with `python -m watcher.notion_sync`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_rag.config import ConfigManager
from local_rag.database import Database
from local_rag.ingestion import IngestionService


NOTION_DATA_SOURCE_ID = "f07c5456-62e7-4589-848d-d87fca9a483c"
SYNC_INTERVAL_SECONDS = 10 * 60
LOG_FILE = PROJECT_ROOT / "logs/notion-sync.log"
SYNC_TAGS = ["email", "work"]
NOTION_API_BASE = "https://api.notion.com/v1"


def configure_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        LOG_FILE,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    )


def sync_once(ingestion: IngestionService) -> dict:
    """Ingest pages and mark successful/known pages as ingested in Notion."""

    result = ingestion.ingest_notion_data_source(
        data_source_id=NOTION_DATA_SOURCE_ID,
        tags=SYNC_TAGS,
    )

    notion = NotionHelper(
        notion_token=ingestion.config["notion"]["api_token"],
        request_timeout=float(ingestion.config["notion"]["request_timeout_seconds"]),
    )
    notion_updates = 0
    notion_update_errors = 0

    for item in result.get("items", []):
        if item.get("status") not in {"ingested", "duplicate"}:
            continue
        page_id = item.get("notion_page_id")
        if not page_id:
            continue
        try:
            notion._make_request(  # noqa: SLF001 - page-property update is not exposed publicly
                "PATCH",
                f"{NOTION_API_BASE}/pages/{page_id}",
                {"properties": {"Ingested": {"checkbox": True}}},
                request_timeout=float(ingestion.config["notion"]["request_timeout_seconds"]),
            )
            notion_updates += 1
        except Exception as exc:  # noqa: BLE001 - report one failed checkbox update and continue
            notion_update_errors += 1
            logger.exception(
                "Failed to mark Notion page as ingested: page_id={page_id} error={error}",
                page_id=page_id,
                error=exc,
            )

    result["notion_updates"] = notion_updates
    result["notion_update_errors"] = notion_update_errors
    result["errors"] = result.get("errors", 0) + notion_update_errors
    return result


def run_sync() -> None:
    console = Console()
    configure_logging()

    config = ConfigManager(PROJECT_ROOT)
    config.ensure_defaults()
    database = Database(config)
    database.initialize()
    ingestion = IngestionService(config, database)

    console.print(
        Panel.fit(
            f"[bold cyan]Notion data source:[/bold cyan] {NOTION_DATA_SOURCE_ID}\n"
            f"[bold cyan]Interval:[/bold cyan] every {SYNC_INTERVAL_SECONDS // 60} minutes\n"
            f"[bold cyan]Log file:[/bold cyan] {LOG_FILE}\n"
            "[dim]Press Ctrl+C to stop.[/dim]",
            title="Local Notion → RAG sync",
            border_style="cyan",
        )
    )
    logger.info(
        "Notion sync started: data_source={data_source} interval_seconds={interval}",
        data_source=NOTION_DATA_SOURCE_ID,
        interval=SYNC_INTERVAL_SECONDS,
    )

    try:
        while True:
            started = time.monotonic()
            try:
                result = sync_once(ingestion)
                logger.info(
                    "Sync completed: ingested={ingested} duplicates={duplicates} "
                    "errors={errors} notion_updates={notion_updates} pages={pages}",
                    ingested=result.get("ingested", 0),
                    duplicates=result.get("duplicates", 0),
                    errors=result.get("errors", 0),
                    notion_updates=result.get("notion_updates", 0),
                    pages=result.get("page_count", 0),
                )
                console.print(
                    f"[green]SYNC COMPLETE[/green] "
                    f"{result.get('ingested', 0)} ingested, "
                    f"{result.get('duplicates', 0)} already present, "
                    f"{result.get('notion_updates', 0)} marked in Notion, "
                    f"{result.get('errors', 0)} errors"
                )
            except Exception as exc:  # noqa: BLE001 - keep the periodic worker alive
                logger.exception("Notion sync failed: {error}", error=exc)
                console.print(f"[bold red]SYNC FAILED[/bold red] {exc}")

            elapsed = time.monotonic() - started
            time.sleep(max(0, SYNC_INTERVAL_SECONDS - elapsed))
    except KeyboardInterrupt:
        console.print("\n[yellow]Notion sync stopped.[/yellow]")
        logger.info("Notion sync stopped by user")


if __name__ == "__main__":
    run_sync()
