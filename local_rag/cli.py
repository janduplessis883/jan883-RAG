from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_rag.backup import BackupService
from local_rag.config import ConfigManager
from local_rag.database import Database


def build_runtime(root_dir: Path):
    config = ConfigManager(root_dir)
    config.ensure_defaults()
    database = Database(config)
    database.initialize()
    backup = BackupService(config)
    return config, database, backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal local RAG maintenance CLI")
    parser.add_argument("command", choices=["init", "stats", "sync-telegram", "backup", "reindex"])
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    config, database, backup = build_runtime(root_dir)

    if args.command == "init":
        print(json.dumps({"status": "ok", "database": str(database.db_path)}, indent=2))
    elif args.command == "stats":
        print(json.dumps(database.get_stats(), indent=2))
    elif args.command == "sync-telegram":
        from local_rag.ingestion import IngestionService

        ingestion = IngestionService(config, database)
        print(json.dumps(ingestion.sync_telegram(), indent=2))
    elif args.command == "backup":
        print(json.dumps(backup.run_backup(), indent=2))
    elif args.command == "reindex":
        database.rebuild_vector_index()
        database.rebuild_fts_index()
        print(json.dumps({"status": "ok", "message": "Vector and FTS indexes rebuilt."}, indent=2))


if __name__ == "__main__":
    main()
