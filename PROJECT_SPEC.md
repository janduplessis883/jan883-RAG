# jan883-RAG Project Specification and Handoff

**Last updated:** 1 September 2026  
**Owner:** Jan du Plessis  
**Repository:** `jan883-RAG`  
**Primary machine:** macOS laptop  
**Project root:** `/Users/janduplessis/code/janduplessis883/streamlit-projects/jan883-RAG`

## 1. Purpose

This project is a local retrieval-augmented generation (RAG) system. It stores searchable content in a local SQLite database, creates embeddings with a local Ollama service, and provides a Streamlit interface for searching and asking questions against the indexed material.

The current automation scope is:

1. Watch a OneDrive-synchronised folder for PDF, TXT, and Markdown files and ingest them automatically.
2. Receive forwarded work emails through Resend on a Render-hosted webhook service.
3. Clean and archive those emails as pages in a Notion database.
4. Periodically sync the text of new Notion email pages into the local RAG database.

Attachments are stored in Notion by the email parser, but are deliberately **not yet downloaded or ingested into the local RAG system**.

## 2. Current status as of 1 September 2026

Implemented and working:

- Local Streamlit RAG application.
- Local SQLite database and vector/lexical search indexes.
- Local Ollama embeddings.
- PDF, TXT, and Markdown ingestion from the OneDrive folder watcher.
- Rich terminal output and rotating Loguru logs for the folder watcher.
- macOS LaunchAgent for the folder watcher.
- Render-hosted FastAPI email receiver.
- Resend inbound email webhook handling.
- Resend email retrieval and attachment download.
- Email cleanup for NHS Mail forwarding artefacts.
- Extraction of the original sender from the first forwarded `From:` header.
- Removal of the repeated NHS Mail signature/footer and repeated-asterisk footer.
- Notion email archive database and data source.
- Upload of all email attachments to the Notion `Attachments` Files & Media property.
- Local Notion-to-RAG worker, running as a macOS LaunchAgent every 10 minutes.
- Notion `Ingested` checkbox updates after successful or already-known page ingestion.
- First local Notion sync completed successfully: 11 pages ingested, 0 duplicates, 0 errors.

Not yet implemented:

- Ingesting Notion attachments into the local RAG database.
- Updating already-indexed Notion pages when their content changes.
- Deleting or tombstoning local RAG records when a Notion page is deleted.
- A durable cross-machine queue between Render and the laptop.
- Automated tests for the full Resend → Render → Notion → local RAG flow.

## 3. Repository layout

Important directories and files:

```text
app.py                         Streamlit application entrypoint
app_pages/                     Streamlit pages
local_rag/                     Core RAG, database, extraction, and sync code
watcher/folder_watcher.py      OneDrive PDF/TXT watcher
watcher/notion_sync.py         Periodic Notion-to-RAG worker
render/                        Render email parser service only
launchd/                       macOS LaunchAgent templates
config/settings.toml           Non-secret application settings
config/secrets.toml            Local secrets; ignored by git
data/                          Local SQLite database and data; ignored by git
logs/                          Runtime logs; ignored by git
tests/                         Existing tests
Makefile                      Common local commands
pyproject.toml                Python dependencies and package configuration
uv.lock                      Locked dependency versions
README.md                     User-facing setup and operating notes
PROJECT_SPEC.md               This handoff specification
```

The Render files intentionally live in the separate `render/` directory so the cloud email service remains isolated from the local worker and Streamlit application.

## 4. Local RAG architecture

The main runtime is composed of:

- `local_rag/config.py`: loads TOML settings and secrets.
- `local_rag/database.py`: creates and manages the SQLite database, source records, chunks, embeddings, FTS, and vector search structures.
- `local_rag/extractors.py`: extracts text from PDFs and text documents.
- `local_rag/chunking.py`: chunks documents and creates chunk-level embeddings.
- `local_rag/embeddings.py`: calls the local Ollama HTTP API.
- `local_rag/ingestion.py`: orchestrates file, Notion page, Notion data-source, and other ingestion workflows.
- `local_rag/retrieval.py`: combines dense/vector and lexical search.
- `local_rag/chat.py`: sends retrieved context to the configured local answer model.
- `app.py` and `app_pages/`: Streamlit UI.

The local database path is configured as `data/rag.sqlite3`. The database is machine-local and must not be committed or uploaded to GitHub.

### Local model configuration

The current settings file configures Ollama, including:

- Ollama base URL: configured in `config/settings.toml`.
- Embedding model: `embeddinggemma-300m-4bit`.
- Embedding dimensions: `768`.
- Default answer model: configured in `config/settings.toml`.

The Ollama service must be running on the laptop for new content to be embedded. Check available models with:

```bash
ollama list
```

Do not put API tokens, passwords, or private URLs in source files.

## 5. OneDrive folder watcher

### Scope

`watcher/folder_watcher.py` monitors this fixed directory:

```text
/Users/janduplessis/Library/CloudStorage/OneDrive-NHS/INGESTION-FOLDER-jan883RAG
```

It recursively observes filesystem changes. New or changed PDF, TXT, and Markdown files are passed to `IngestionService`. Unsupported files are reported and ignored.

Files ingested from this watcher receive the tags:

- `onedrive`
- the lower-case file extension, such as `pdf`, `txt`, or `md`

### Run manually

From the project root:

```bash
make watch-folder
```

or:

```bash
.venv/bin/python -m watcher.folder_watcher
```

### Background operation

The LaunchAgent template is:

```text
launchd/com.janduplessis.jan883-rag.folder-watcher.plist
```

The expected service label is:

```text
com.janduplessis.jan883-rag.folder-watcher
```

Its logs are:

```text
logs/folder-watcher.log
logs/folder-watcher.stdout.log
logs/folder-watcher.stderr.log
```

## 6. Render email parser

### Purpose

The `render/` service receives forwarded work emails through a Resend receiving address, retrieves the full email and attachments from Resend, cleans the message, and creates one Notion page per email.

### Files

- `render/app.py`: FastAPI application and webhook logic.
- `render/requirements.txt`: cloud service dependencies, including `notionhelper`.
- `render/Dockerfile`: Python 3.12 container definition.
- `render/README.md`: Render-specific deployment instructions.
- `render/.env.example`: names of required environment variables only.

### Required Render environment variables

The Render service requires:

```text
RESEND_API_KEY
RESEND_WEBHOOK_SECRET
NOTION_TOKEN
NOTION_DATA_SOURCE_ID
NOTION_CALENDAR_DATA_SOURCE_ID
```

The values must be configured in Render's environment settings and must not be committed to GitHub.

The current Notion data source ID is:

```text
f07c5456-62e7-4589-848d-d87fca9a483c
```

The Render service should use the `render/` directory as its root directory. Its container starts Uvicorn on Render's `$PORT`.

Emails whose cleaned body starts with the word `calendar` (case-insensitive) are
routed to `NOTION_CALENDAR_DATA_SOURCE_ID`. The subject is stored in the
Calendar data source's `Event` title property, the current UTC time is stored
in its `Date` property as an ISO 8601 timestamp, and the cleaned body becomes
the page content. This route does not check duplicates and does not retrieve
or upload attachments. All other emails continue through the Work Email
Archive duplicate and attachment flow.

### Webhook behavior

The service:

1. Accepts the raw request body.
2. Verifies the Resend/Svix webhook signature.
3. Reads the `email.received` event.
4. Retrieves full email content from Resend's Receiving API.
5. Extracts sender, recipients, subject, received date, cleaned body, and attachments.
6. Checks Notion for an existing page using `Message ID` to avoid duplicates.
7. Creates a Notion page.
8. Uploads every attachment and sets the complete list on the Notion Files & Media property.

### Email cleanup rules

NHS Mail forwarding commonly adds the user's own signature above the forwarded message. The parser starts the useful message at the first forwarded `From:` header, so the repeated signature is excluded.

The parser also removes:

- The NHS Mail confidentiality footer beginning with the standard confidential-information warning.
- The NHS.net Connect footer and its links.
- Horizontal separator lines represented by `---`.
- Lines made from repeated asterisks, including wrapped or split rows such as the approximately 116-asterisk footer shown in forwarded messages.

The first forwarded `From:` field is treated as the real sender. This prevents Notion from listing the forwarding account (`jan.duplessis@nhs.net`) as the sender.

### Attachment behavior

Attachments are uploaded to the Notion page's `Attachments` Files & Media property. They are not appended to the email body.

The implementation uploads all files first and then performs one combined property update. This is important because repeatedly calling NotionHelper's one-step Files & Media helper would replace the previous property value rather than reliably append to it.

## 7. Notion archive

The email archive is a Notion database created under the user's `Databases` parent page.

Database title:

```text
Work Email Archive
```

Data source ID:

```text
f07c5456-62e7-4589-848d-d87fca9a483c
```

Expected properties:

| Property | Type | Purpose |
|---|---|---|
| `Subject` | Title | Email subject and page title |
| `Sender` | Rich text | Original sender extracted from forwarded `From:` |
| `Recipients` | Rich text | To/CC recipients |
| `Date Received` | Date | Received date from Resend |
| `Tags` | Multi-select | Currently `email` and `work` |
| `Attachments` | Files & Media | All email attachments |
| `Message ID` | Rich text | Resend message ID and idempotency key |
| `Source URL` | URL | Optional source/reference URL |
| `Ingested` | Checkbox | Set true after successful or already-known local RAG ingestion |

The Notion integration must have access to the database/data source. A valid ID without integration access will generally behave like a missing object.

## 8. Local Notion-to-RAG sync

### Purpose

`watcher/notion_sync.py` periodically scans the Notion `Work Email Archive` data source and ingests the page text into the local RAG database.

### Current behavior

- Runs one sync immediately at startup.
- Waits until the ten-minute interval has elapsed since the previous run.
- Queries the configured data source.
- Uses the existing `IngestionService.ingest_notion_data_source()` method.
- Retrieves page text through `NotionHelper`.
- Includes expanded child-page content according to the existing Notion settings.
- Stores each page with a canonical URI in the form `notion://page/{page_id}`.
- Uses the Notion page ID as the stable identity for duplicate detection.
- Adds the tags `email` and `work`.
- Sets the Notion `Ingested` checkbox to true after a page is successfully ingested.
- Also sets `Ingested` to true for pages already present locally, allowing the checkbox to be repaired if it was manually cleared.
- Leaves the checkbox unticked when extraction, chunking, embedding, or database storage fails.
- Does not retrieve, parse, or ingest Notion attachments.
- Keeps running after a single sync failure and records the exception in the log.

### Constants

The worker currently defines:

```python
NOTION_DATA_SOURCE_ID = "f07c5456-62e7-4589-848d-d87fca9a483c"
SYNC_INTERVAL_SECONDS = 10 * 60
```

The data source ID is intentionally explicit in the worker for now. If multiple environments are introduced, move it into configuration or an environment variable.

### Run manually

```bash
make sync-notion
```

or:

```bash
.venv/bin/python -m watcher.notion_sync
```

### Background operation

The LaunchAgent template is:

```text
launchd/com.janduplessis.jan883-rag.notion-sync.plist
```

The installed service label is:

```text
com.janduplessis.jan883-rag.notion-sync
```

It is currently loaded and running for the macOS user account. It uses the project virtual environment and has the project root as its working directory.

Logs:

```text
logs/notion-sync.log
logs/notion-sync.stdout.log
logs/notion-sync.stderr.log
```

Useful status command:

```bash
launchctl print gui/$(id -u)/com.janduplessis.jan883-rag.notion-sync
```

To stop/unload the service:

```bash
launchctl bootout gui/$(id -u)/com.janduplessis.jan883-rag.notion-sync
```

To reinstall after changing the plist or worker path:

```bash
mkdir -p logs ~/Library/LaunchAgents
cp launchd/com.janduplessis.jan883-rag.notion-sync.plist ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/com.janduplessis.jan883-rag.notion-sync 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.janduplessis.jan883-rag.notion-sync.plist
launchctl kickstart -k gui/$(id -u)/com.janduplessis.jan883-rag.notion-sync
```

## 9. Configuration and secrets

Non-secret settings are in:

```text
config/settings.toml
```

Local secrets are expected in:

```text
config/secrets.toml
```

The example file is:

```text
config/secrets.toml.example
```

The local Notion configuration must contain an enabled Notion section and an API token. The Notion database/data source must be shared with the integration used by that token.

Never commit:

- `config/secrets.toml`
- `.env` files
- Render API keys or webhook secrets
- the SQLite database under `data/`
- runtime logs under `logs/`
- private email content or downloaded attachments

The current `.gitignore` excludes these categories, including `data/`, `logs/`, `.env`, `render/.env`, and `config/secrets.toml`.

## 10. Common commands

Install/synchronise dependencies:

```bash
uv sync
```

Initialise the local database:

```bash
make init
```

View database statistics:

```bash
make stats
```

Run the Streamlit app:

```bash
make app
```

Compile-check the Python source:

```bash
.venv/bin/python -m compileall app.py local_rag watcher
```

Run the current tests:

```bash
.venv/bin/python -m pytest tests
```

## 11. Troubleshooting

### Notion sync is running but no pages are ingested

Check:

1. `logs/notion-sync.log`.
2. The local Notion token exists in `config/secrets.toml`.
3. The Notion integration has access to the `Work Email Archive` data source.
4. Ollama is running and the configured embedding model is available.
5. The pages have not already been indexed under `notion://page/{page_id}`.

### LaunchAgent is not running

Validate the plist:

```bash
plutil -lint launchd/com.janduplessis.jan883-rag.notion-sync.plist
```

Then inspect the service:

```bash
launchctl print gui/$(id -u)/com.janduplessis.jan883-rag.notion-sync
```

Read both `logs/notion-sync.stdout.log` and `logs/notion-sync.stderr.log`.

### A new Notion page is not found in the RAG search

The sync is periodic rather than event-driven. Wait for the next ten-minute cycle, or restart/kickstart the worker. Check `notion-sync.log` for the resulting counts. The page must contain text that survives Notion extraction and must be successfully embedded by Ollama.

### Render email parser deployment fails

Confirm:

- Render root directory is `render`.
- The build uses `render/requirements.txt`.
- The service starts Uvicorn on `$PORT`.
- All four required environment variables are present.
- The Notion data source is shared with the Render Notion integration.
- The Resend webhook URL points to the deployed Render service.

## 12. Git and deployment workflow

The repository's `.git` directory is currently managed outside the writable project area in this environment, so commits may need to be created from the user's normal terminal if the Codex environment cannot update the Git index.

Before pushing changes:

```bash
git status
git diff
git add PROJECT_SPEC.md watcher/notion_sync.py launchd/com.janduplessis.jan883-rag.notion-sync.plist Makefile README.md
git commit -m "Add local Notion to RAG sync worker"
git push origin master
```

If the Render parser has changed, include the relevant `render/` files in the same commit. Do not add ignored secrets, databases, logs, or private email files.

Render should redeploy from the pushed branch according to the service's existing auto-deploy configuration.

## 13. Planned next phase: Notion attachments

The next major feature is to ingest attachments while preserving the current email page as the parent record.

Recommended design:

1. Query the Notion Files & Media property for each new email page.
2. Resolve the current Notion-hosted file URL or file-upload reference.
3. Download each attachment to a controlled temporary directory.
4. Support PDF and TXT first, matching the existing local file ingestion rules.
5. Ingest each attachment through `IngestionService` with metadata linking it to:
   - the Notion page ID,
   - the email Message ID,
   - the attachment filename,
   - the parent email subject.
6. Add stable canonical URIs so the same attachment is not repeatedly ingested.
7. Delete temporary files after successful or failed processing.
8. Decide explicitly whether unsupported attachment types should be ignored, logged, or converted.

The attachment phase should not be started until the page-text sync is stable and duplicate/update behavior has been specified.

## 14. Important implementation decisions

- `argparse` is intentionally not used by either watcher. Configuration is defined in code or existing project configuration.
- The Notion worker is a long-running process rather than a ten-minute cron job so it can report status, retain the initialized database connection, and handle failures without losing the process.
- The Notion worker uses the existing ingestion service rather than duplicating chunking, embedding, and database logic.
- Duplicate detection is based on the canonical Notion page URI, not the page title or subject.
- Email sender metadata comes from the original forwarded `From:` line, not the Resend envelope sender.
- Notion attachments are placed in the Files & Media property, not inserted into the searchable email body.
- Secrets and local RAG data stay outside version control.
