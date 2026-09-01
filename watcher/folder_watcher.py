"""Watch the OneDrive ingestion folder and report filesystem changes."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from watchfiles import Change, watch

# Support both `python -m watcher.folder_watcher` and direct execution with
# `python watcher/folder_watcher.py` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_rag.config import ConfigManager
from local_rag.database import Database
from local_rag.ingestion import IngestionService


SCANNING_DIRECTORY = Path(
    "/Users/janduplessis/Library/CloudStorage/OneDrive-NHS/INGESTION-FOLDER-jan883RAG"
)
LOG_FILE = PROJECT_ROOT / "logs/folder-watcher.log"
DEBOUNCE_MS = 1_600
FILE_STABILITY_CHECKS = 3
FILE_STABILITY_DELAY_SECONDS = 1.0
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

CHANGE_LABELS = {
    Change.added: ("ADDED", "green"),
    Change.modified: ("MODIFIED", "yellow"),
    Change.deleted: ("DELETED", "red"),
}


def configure_logging(log_file: Path) -> None:
    """Write rotating watcher logs to ``log_file``."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        log_file,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    )


def describe_change(change: Change) -> tuple[str, str]:
    """Return the human-readable label and Rich colour for a change."""

    return CHANGE_LABELS.get(change, (str(change).upper(), "white"))


def tags_for_file(path: Path) -> list[str]:
    """Return consistent tags for an automatically ingested file."""

    return ["onedrive", path.suffix.lower().lstrip(".")]


def wait_for_stable_file(path: Path) -> bytes:
    """Wait until a synced file stops changing, then return its bytes."""

    previous_signature: tuple[int, int] | None = None
    for _ in range(FILE_STABILITY_CHECKS):
        if not path.is_file():
            raise FileNotFoundError(f"File disappeared before ingestion: {path}")
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature == previous_signature:
            return path.read_bytes()
        previous_signature = signature
        time.sleep(FILE_STABILITY_DELAY_SECONDS)

    if not path.is_file():
        raise FileNotFoundError(f"File disappeared before ingestion: {path}")
    return path.read_bytes()


def ingest_watched_file(path: Path, ingestion: IngestionService, database: Database) -> dict:
    """Ingest one supported OneDrive file and record its outcome."""

    path = path.expanduser().resolve()
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"status": "ignored", "reason": "unsupported_extension", "file_path": str(path)}

    file_path = str(path)
    logged = database.get_ingestion_log_entry(file_path)
    if logged and logged["status"] in {"ingested", "duplicate"}:
        return {
            "status": "skipped_logged",
            "file_path": file_path,
            "source_id": logged["source_id"],
            "message": logged["message"],
        }

    content = wait_for_stable_file(path)
    content_hash = hashlib.sha256(content).hexdigest()
    tags = tags_for_file(path)
    try:
        result = ingestion.ingest_file(
            filename=path.name,
            content=content,
            tags=tags,
            source_type="onedrive_file",
            canonical_uri=path.as_uri(),
            external_ref=str(SCANNING_DIRECTORY),
            metadata={
                "folder": str(SCANNING_DIRECTORY),
                "file_path": file_path,
                "file_extension": path.suffix.lower(),
                "automated": True,
            },
        )
        log_status = result["status"]
        message = (
            "Already present in the database."
            if log_status == "duplicate"
            else f"Stored {result.get('chunk_count', 0)} chunks."
        )
        database.record_ingestion_log(
            file_path=file_path,
            filename=path.name,
            content_hash=content_hash,
            source_id=result.get("source_id"),
            status=log_status,
            message=message,
        )
        return {**result, "file_path": file_path}
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the watcher
        database.record_ingestion_log(
            file_path=file_path,
            filename=path.name,
            content_hash=content_hash,
            source_id=None,
            status="error",
            message=str(exc),
        )
        raise


def run_watcher() -> None:
    """Watch the configured OneDrive folder until interrupted."""

    folder = SCANNING_DIRECTORY.expanduser().resolve()
    console = Console()
    configure_logging(LOG_FILE)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder}")

    config = ConfigManager(PROJECT_ROOT)
    config.ensure_defaults()
    database = Database(config)
    database.initialize()
    ingestion = IngestionService(config, database)

    console.print(
        Panel.fit(
            f"[bold cyan]Watching:[/bold cyan] {folder}\n"
            f"[bold cyan]Log file:[/bold cyan] {LOG_FILE}\n"
            "[dim]Press Ctrl+C to stop.[/dim]",
            title="OneDrive folder watcher",
            border_style="cyan",
        )
    )
    logger.info("Watcher started: folder={folder} debounce_ms={debounce_ms}", folder=folder, debounce_ms=DEBOUNCE_MS)

    try:
        for changes in watch(folder, recursive=True, debounce=DEBOUNCE_MS):
            logger.info("Detected batch of {count} change(s)", count=len(changes))
            for change, changed_path in sorted(changes, key=lambda item: str(item[1])):
                path = Path(changed_path)
                label, colour = describe_change(change)
                console.print(f"[{colour}]{label:<8}[/{colour}] {path}")
                logger.info("{label}: {path}", label=label, path=path)

                if change not in {Change.added, Change.modified} or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        console.print(f"[dim]IGNORED  Unsupported file type: {path.name}[/dim]")
                    continue

                try:
                    result = ingest_watched_file(path, ingestion, database)
                    status = result["status"]
                    if status == "ingested":
                        console.print(
                            f"[bold green]INGESTED[/bold green] {path.name} "
                            f"({result.get('chunk_count', 0)} chunks; tags: {', '.join(result.get('tags', []))})"
                        )
                    elif status == "duplicate":
                        console.print(f"[yellow]DUPLICATE[/yellow] {path.name} already exists in the database")
                    elif status == "skipped_logged":
                        console.print(f"[dim]SKIPPED  {path.name} was already handled[/dim]")
                except Exception as exc:  # noqa: BLE001 - continue watching other files
                    console.print(f"[bold red]INGESTION FAILED[/bold red] {path.name}: {exc}")
                    logger.exception("Ingestion failed for {path}", path=path)
    except KeyboardInterrupt:
        console.print("\n[yellow]Watcher stopped.[/yellow]")
        logger.info("Watcher stopped by user")


if __name__ == "__main__":
    run_watcher()
