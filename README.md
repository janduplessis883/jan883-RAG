# Personal Local RAG

Local-first Python RAG system for articles, PDFs, pasted text, and Telegram bot ingestion.

## Features

- Streamlit app for ingest, search, chat, config editing, and backups
- Local Ollama embeddings with `nomic-embed-text:v1.5`
- Selectable local answer models
- SQLite storage with `sqlite-vec` support and a Python fallback path
- Hybrid dense + SQLite FTS5/BM25 retrieval fused with Reciprocal Rank Fusion (`rrf_k = 60`)
- Semantic chunking with configurable thresholds
- Deduplication by normalized URL and content hash
- Notion page and data-source ingestion via `notionhelper`
- Telegram bot polling for links, forwarded messages, and PDF/document ingest
- One-click timestamped backups to a separate partition
- Read-only OneDrive folder watcher with Rich terminal output and Loguru logs

## Quick Start

1. Install dependencies into your existing `pyenv` environment:

```bash
pip install -e .
```

2. Review `config/settings.toml` and `config/secrets.toml`.
   Add your Notion integration token under `[notion]` in `config/secrets.toml` if you want Notion ingestion.

3. Start Ollama and ensure your local models are available:

```bash
ollama list
```

4. Launch the app:

```bash
make app
```

## Notes

- Hybrid retrieval is enabled by default and can be disabled in the Chat sidebar with **Hybrid retrieval (FTS5 + dense)**, or via `retrieval.hybrid_enabled` in `config/settings.toml`.

- `config/secrets.toml` is ignored by git and intended for local-only secrets.
- Optional Langfuse tracing is enabled when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
  are set in `.streamlit/secrets.toml` (or the shell environment). For the local Docker
  instance, use `LANGFUSE_BASE_URL=http://localhost:3000`.
  If Streamlit also runs inside Docker, use `http://host.docker.internal:3000` instead.
  Content capture is disabled by default; set `langfuse.capture_content = true` only when
  it is appropriate to send prompts and answers to your Langfuse instance.
- Telegram ingestion is poll-based. You can trigger it from the Streamlit admin page or with `make sync-telegram`.
- Backups are written into timestamped folders below the configured backup root.

## OneDrive folder watcher

Install the updated dependencies into the existing environment:

```bash
pip install -e .
```

Start watching the configured OneDrive folder:

```bash
make watch-folder
```

Or run it directly:

```bash
python -m watcher.folder_watcher
```

The scanning directory is defined by `SCANNING_DIRECTORY` in `watcher/folder_watcher.py`. The watcher recursively reports added, modified, and deleted files in the terminal. New or changed PDF and TXT files are automatically passed to the existing RAG ingestion service, embedded with the configured Ollama model, and stored in the SQLite database. Files are tagged with `onedrive` and their file type. It writes a rotating log to `logs/folder-watcher.log`; unsupported file types are reported and ignored. Stop it with `Ctrl+C`.

### Run automatically in the background on macOS

The repository includes a LaunchAgent at `launchd/com.janduplessis.jan883-rag.folder-watcher.plist`. It starts the watcher when you log in and restarts it if it exits.

Install and load it for your user account:

```bash
mkdir -p logs ~/Library/LaunchAgents
cp launchd/com.janduplessis.jan883-rag.folder-watcher.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.janduplessis.jan883-rag.folder-watcher.plist
```

Check its status:

```bash
launchctl print gui/$(id -u)/com.janduplessis.jan883-rag.folder-watcher
```

Stop and remove it:

```bash
launchctl bootout gui/$(id -u)/com.janduplessis.jan883-rag.folder-watcher
rm ~/Library/LaunchAgents/com.janduplessis.jan883-rag.folder-watcher.plist
```

LaunchAgent stdout and stderr are written to `logs/folder-watcher.stdout.log` and `logs/folder-watcher.stderr.log`.

## Local Notion-to-RAG sync

The Notion email archive can be synced into the local RAG database every 10 minutes:

```bash
make sync-notion
```

The worker runs one sync immediately, then repeats every 10 minutes. It uses the
Notion data source `f07c5456-62e7-4589-848d-d87fca9a483c`, ingests page text only,
and skips pages already present in the local database. PDF and other attachments
are deliberately not downloaded or ingested yet.
After a page is successfully ingested, or is found to be already present locally,
the worker sets its Notion `Ingested` checkbox to true.

To run it automatically at login, install its LaunchAgent:

```bash
mkdir -p logs ~/Library/LaunchAgents
cp launchd/com.janduplessis.jan883-rag.notion-sync.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.janduplessis.jan883-rag.notion-sync.plist
```

Its Loguru log is `logs/notion-sync.log`; LaunchAgent stdout and stderr are
`logs/notion-sync.stdout.log` and `logs/notion-sync.stderr.log`.
