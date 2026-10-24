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

## Local API

Start the local FastAPI service alongside the Streamlit app:

```bash
make api
```

The API binds to `127.0.0.1:8000` and exposes:

- `GET /health` — database statistics and Ollama availability
- `POST /search` — hybrid or dense/lexical search
- `POST /chat` — grounded answers with related questions and source passages

Example search request:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What decisions were made?","limit":5}'
```

## Local Nimble evidence assessment

The Chat page can optionally ask a local TypeSafe System One model to assess
whether retrieved passages support the generated answer. The assessment uses
the question, draft answer, and retrieved passage text, and is shown with the
answer and citations. It does not replace retrieval or the answer model.

Install the project dependencies and add the TypeSafe settings to
`.streamlit/secrets.toml`:

```toml
TYPESAFE_BASE_URL = "http://localhost:11434"
TYPESAFE_API_KEY = "ollama"
TYPESAFE_DEFAULT_MODEL = "nimble"
```

Run `pip install -e .` if you have not installed the project dependencies yet.
The Chat page reads these settings when it calls the SDK; shell environment
variables remain a fallback. In the Chat page's **Advanced settings**, enable
**Nimble evidence assessment**. It is enabled by default in
`config/settings.toml`; turn it off there or in the chat sidebar to skip the
additional local model request. If Nimble is unavailable, the RAG answer is
still shown and the assessment is marked unavailable.

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

The scanning directory is defined by `SCANNING_DIRECTORY` in `watcher/folder_watcher.py`. The watcher recursively reports added, modified, and deleted files in the terminal. New or changed PDF, TXT, and Markdown (`.md`) files are automatically passed to the existing RAG ingestion service, embedded with the configured Ollama model, and stored in the SQLite database. Files are tagged with `onedrive` and their file type. It writes a rotating log to `logs/folder-watcher.log`; unsupported file types are reported and ignored. Stop it with `Ctrl+C`.

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

The Notion email archive and calendar can be synced into the local RAG database every hour:

```bash
make sync-notion
```

The worker runs one sync immediately, then repeats every hour. It uses the
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

## Notion contacts

Contacts use the same local SQLite and vector index as the calendar. Configure
`notion.contacts_data_source_id` in `config/settings.toml` and use the existing
Notion token. This workspace is configured for data source
`202fdfd6-8a97-81dd-9a7b-000b08218e12`.

The integration reads **Name / Company**, **Address** (Place or text),
**Description**, **Email**, **Files & media**, **Tel**, and **URL**. Each contact
has a source-linked structured record. Name, address and description are embedded;
exact email, telephone and attachment details are read from the structured record.
Telephone search normalizes punctuation, international prefixes and UK local forms,
while answers preserve the original display number. Empty fields remain empty.

Use **Contacts** in the Chat page's **Chat with** selector. It can be selected
alone or with Knowledge Base and Calendar. Example questions:

- “What is Sarah's telephone number?”
- “Who owns 020 7946 0123?”
- “Who do I know who specialises in kitchen renovations?”
- “List all contacts without an email address.”

The **Contacts** page provides a directory and missing-field filters. **Sync
contacts** is available there and in Admin. A one-off sync is also available with
`python -m local_rag.cli sync-contacts`. The existing Notion worker includes
contacts in its configured sync cycle; restart a previously running worker once
to load the new code.

All contacts are synced, with no date cutoff. Page IDs preserve identity across
renames; identical descriptions do not merge separate contacts. Changes to contact
details update the structured record, and only changes to embedding text require
new embeddings. Removed contacts are excluded from search after a complete,
successful listing; failed or malformed scans retain the previous records.

Attachments are **links only**: no downloads, OCR, or attachment content embeddings.
Notion-hosted URLs are refreshed when they expire. If refreshing is unavailable,
the link opens the owning Notion page and is labelled accordingly. Website URLs
are stored without crawling. Notion contact properties are never written back.
