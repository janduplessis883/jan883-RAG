# Notion contacts implementation plan

**Goal:** Sync the user's Notion contacts into structured SQLite records and the existing RAG index, supporting exact contact lookup and description search.

**Architecture:** Follow the calendar's source-linked record pattern. Use NotionHelper's paginated page API; Notion remains the editable source of truth. Store attachment links only and refresh expired URLs during retrieval.

**Source:** User-approved design in chat, 2 Oct 2026. Contacts data source: `202fdfd6-8a97-81dd-9a7b-000b08218e12`.

**Constraints:** Preserve existing workspace changes. Do not download attachments, crawl websites, or write contact properties back to Notion. Do not apply a calendar date cutoff. Keep identical contacts distinct by page ID. Reconcile removals only after a complete successful listing and valid schema; retain local records on partial failures.

- [x] Add integration tests with real SQLite and ingestion, substituting only external Notion and embedding calls. Cover field normalization, identical descriptions, unchanged runs, phone-only updates without re-embedding, removal, failed scans, and attachment refresh/fallback.
- [x] Add `local_rag/contacts.py`: contact parser, sync service, attachment hydration, and natural-language lookup. Add the structured table and scoped lookup methods to `local_rag/database.py`.
- [x] Add chat exact lookup and contact filtering. Hydrate semantic hits with fresh structured fields and attachment URLs. Pass those fields to answer generation and render them with links.
- [x] Add manual and scheduled sync, a contacts directory, the supplied configuration ID, CLI sync, and documentation.
- [x] Run contact tests, existing full test suite, and Streamlit page smoke tests. Check the live schema and perform a first sync if credentials and model access permit.

**Review focus:** Empty optional fields, ambiguous names, international/local phone formatting, expired URLs with unavailable Notion, and failed/partial scans must not cause fabricated values, arbitrary selection, or destructive reconciliation.

**Verification:** 68 tests passed, including 13 contacts tests and Streamlit page checks. Initial live import stored all 92 contacts and 92 vectors with zero errors; SQLite integrity was `ok`. Confirmed Contacts in the running chat selector. Reloaded the existing Notion LaunchAgent and verified its next contacts sync succeeded.
