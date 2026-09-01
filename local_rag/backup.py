from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil


class BackupService:
    def __init__(self, config_manager) -> None:
        self.config_manager = config_manager
        self.config = config_manager.load_merged()
        self.workspace_dir = self.config_manager.root_dir
        self.backup_root = Path(self.config["app"]["backup_root"])

    def run_backup(self) -> dict:
        self.backup_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = self.backup_root / f"snapshot_{timestamp}"

        shutil.copytree(
            self.workspace_dir,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".DS_Store"),
        )
        return {
            "status": "ok",
            "destination": str(destination),
        }

