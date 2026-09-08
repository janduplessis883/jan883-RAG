from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sqlite3


class BackupService:
    def __init__(self, config_manager) -> None:
        self.config_manager = config_manager
        self.config = config_manager.load_merged()
        self.workspace_dir = config_manager.root_dir.resolve()
        self.backup_root = config_manager.resolve_path(self.config["app"]["backup_root"])

    def run_backup(self) -> dict:
        from local_rag.database import Database

        if self.backup_root.resolve().is_relative_to(self.workspace_dir):
            raise ValueError("Choose a backup location outside the workspace.")
        self.backup_root.mkdir(parents=True, exist_ok=True)
        destination = self.backup_root / datetime.now().strftime("snapshot_%Y%m%d_%H%M%S_%f")
        db_path = self.config_manager.resolve_path(self.config["app"]["database_path"])
        excluded = {db_path.resolve(), Path(str(db_path) + "-wal").resolve(), Path(str(db_path) + "-shm").resolve()}

        def ignore(directory, names):
            return [name for name in names if name in {"__pycache__", ".pytest_cache", ".DS_Store", ".git", ".venv"}
                    or (Path(directory) / name).resolve() in excluded]

        shutil.copytree(self.workspace_dir, destination, ignore=ignore)
        relative_db = db_path.relative_to(self.workspace_dir) if db_path.is_relative_to(self.workspace_dir) else Path("database/rag.sqlite3")
        snapshot = destination / relative_db
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        target = sqlite3.connect(snapshot)
        try:
            source.backup(target)
            checks = [row[0] for row in target.execute("PRAGMA integrity_check")]
            if checks != ["ok"]:
                raise ValueError("Backup database integrity check failed.")
        finally:
            source.close()
            target.close()
        raw_dir = self.config_manager.resolve_path(self.config["app"]["raw_files_dir"])
        if not raw_dir.is_relative_to(self.workspace_dir) and raw_dir.exists():
            shutil.copytree(raw_dir, destination / "external_raw")
        result = {"status": "ok", "destination": str(destination), "database": str(relative_db),
                  "verified_at": datetime.now(timezone.utc).isoformat(),
                  "verification": "SQLite integrity_check passed; full restore not tested"}
        (destination / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        database = Database(self.config_manager)
        try:
            database.initialize()
            database.record_operation("backup", result)
        finally:
            database.connection.close()
        return result
