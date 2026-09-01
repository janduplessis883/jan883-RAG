from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Callable

from bs4 import BeautifulSoup, NavigableString

from local_rag.chunking import FixedChunker, SemanticChunker
from local_rag.embeddings import OllamaClient
from local_rag.extractors import (
    extract_article,
    extract_pdf,
    extract_text_document,
    extract_urls,
    normalize_url,
    sanitize_filename,
    sha256_text,
)
from local_rag.telegram_sync import TelegramBotClient


class IngestionService:
    def __init__(self, config_manager, database) -> None:
        self.config_manager = config_manager
        self.database = database
        self.config = config_manager.load_merged()
        self.chunker = SemanticChunker(self.config["chunking"])
        self.ollama = OllamaClient(self.config["ollama"])
        self.raw_dir = self.config_manager.resolve_path(self.config["app"]["raw_files_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def ingest_url(
        self,
        url: str,
        tags: list[str] | None = None,
        chunking: dict | None = None,
    ) -> dict:
        extracted = self.preview_url(url)
        return self._store_document(
            source_type="article",
            title=extracted["title"],
            text=extracted["text"],
            tags=tags or [],
            canonical_uri=extracted["canonical_uri"],
            external_ref=None,
            metadata=extracted["metadata"],
            raw_text=extracted["raw_text"],
            raw_binary=None,
            raw_binary_name=None,
            chunking=chunking,
        )

    def preview_url(self, url: str) -> dict:
        return extract_article(url)

    def preview_notion_page(self, page_url_or_id: str) -> dict:
        notion = self._get_notion_client()
        page_id = notion.extract_page_id_from_url(page_url_or_id)
        max_child_page_depth = int(self.config.get("notion", {}).get("max_child_page_depth", 3))
        payload = notion.get_page(
            page_id,
            return_markdown=True,
            expand_child_pages=True,
            max_child_page_depth=max_child_page_depth,
        )
        markdown, child_page_ids = self._notion_markdown_with_children(payload)

        title = self._extract_notion_title(payload.get("properties", {}), fallback=f"Notion page {page_id[:8]}")
        return {
            "page_id": page_id,
            "title": title,
            "text": markdown.strip(),
            "canonical_uri": f"notion://page/{page_id}",
            "metadata": {
                "notion_page_id": page_id,
                "notion_original_input": page_url_or_id,
                "child_page_ids": child_page_ids,
                "child_page_count": len(child_page_ids),
                "max_child_page_depth": max_child_page_depth,
            },
            "raw_text": markdown,
            "properties": payload.get("properties", {}),
            "child_page_ids": child_page_ids,
        }

    def ingest_notion_page(
        self,
        page_url_or_id: str,
        tags: list[str] | None = None,
        chunking: dict | None = None,
    ) -> dict:
        preview = self.preview_notion_page(page_url_or_id)
        return self._store_document(
            source_type="notion_page",
            title=preview["title"],
            text=preview["text"],
            tags=tags or [],
            canonical_uri=preview["canonical_uri"],
            external_ref=page_url_or_id,
            metadata=preview["metadata"],
            raw_text=preview["raw_text"],
            raw_binary=None,
            raw_binary_name=None,
            chunking=chunking,
        )

    def preview_notion_data_source(self, data_source_id: str, limit: int | None = None) -> dict:
        notion = self._get_notion_client()
        normalized_id = self._normalize_notion_id(data_source_id)
        effective_limit = limit or int(self.config["notion"]["data_source_preview_limit"])
        data_source = notion.get_data_source(normalized_id)
        dataframe = notion.get_data_source_pages_as_dataframe(
            normalized_id,
            limit=effective_limit,
            include_page_ids=True,
            request_timeout=float(self.config["notion"]["request_timeout_seconds"]),
        )
        records = self._dataframe_records(dataframe)
        return {
            "data_source_id": normalized_id,
            "title": self._extract_notion_title(data_source.get("title", []), fallback=f"Data source {normalized_id[:8]}"),
            "schema": data_source.get("properties", {}),
            "dataframe": dataframe,
            "records": records,
            "page_count": len(records),
        }

    def ingest_notion_data_source(
        self,
        data_source_id: str,
        tags: list[str] | None = None,
        chunking: dict | None = None,
        limit: int | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        preview = self.preview_notion_data_source(data_source_id, limit=limit)
        total = preview["page_count"]
        results: list[dict[str, Any]] = []
        ingested = 0
        duplicates = 0
        errors = 0

        for index, row in enumerate(preview["records"], start=1):
            page_id = row.get("notion_page_id")
            label = self._row_display_title(row, fallback=str(page_id or f"row-{index}"))
            if progress_callback:
                progress_callback(index - 1, total, label)

            if not page_id:
                errors += 1
                results.append({"status": "error", "error": "Row missing notion_page_id", "row": row})
                continue

            try:
                canonical_uri = f"notion://page/{self._normalize_notion_id(str(page_id))}"
                existing = self.database.find_source_by_uri(canonical_uri)
                if existing:
                    duplicates += 1
                    results.append(
                        {
                            "status": "duplicate",
                            "notion_page_id": self._normalize_notion_id(str(page_id)),
                            "source_id": int(existing["id"]),
                            "title": existing["title"],
                            "canonical_uri": existing["canonical_uri"],
                            "skipped_content_fetch": True,
                        }
                    )
                    continue

                page_preview = self.preview_notion_page(page_id)
                merged_metadata = dict(page_preview["metadata"])
                merged_metadata["notion_data_source_id"] = preview["data_source_id"]
                merged_metadata["notion_row"] = row
                result = self._store_document(
                    source_type="notion_data_source_page",
                    title=page_preview["title"],
                    text=page_preview["text"],
                    tags=tags or [],
                    canonical_uri=page_preview["canonical_uri"],
                    external_ref=f"notion://data-source/{preview['data_source_id']}",
                    metadata=merged_metadata,
                    raw_text=page_preview["raw_text"],
                    raw_binary=None,
                    raw_binary_name=None,
                    chunking=chunking,
                )
                result["notion_page_id"] = self._normalize_notion_id(str(page_id))
                results.append(result)
                if result["status"] == "duplicate":
                    duplicates += 1
                else:
                    ingested += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                results.append({"status": "error", "page_id": page_id, "error": str(exc)})

        if progress_callback:
            progress_callback(total, total, "Completed")

        return {
            "status": "ok",
            "data_source_id": preview["data_source_id"],
            "data_source_title": preview["title"],
            "page_count": total,
            "ingested": ingested,
            "duplicates": duplicates,
            "errors": errors,
            "items": results,
        }

    def ingest_text(
        self,
        *,
        title: str,
        text: str,
        tags: list[str] | None = None,
        source_type: str = "text",
        canonical_uri: str | None = None,
        external_ref: str | None = None,
        metadata: dict | None = None,
        chunking: dict | None = None,
    ) -> dict:
        return self._store_document(
            source_type=source_type,
            title=title,
            text=text,
            tags=tags or [],
            canonical_uri=canonical_uri,
            external_ref=external_ref,
            metadata=metadata or {},
            raw_text=text,
            raw_binary=None,
            raw_binary_name=None,
            chunking=chunking,
        )

    def ingest_file(
        self,
        *,
        filename: str,
        content: bytes,
        tags: list[str] | None = None,
        source_type: str = "document",
        canonical_uri: str | None = None,
        external_ref: str | None = None,
        metadata: dict | None = None,
        chunking: dict | None = None,
    ) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            extracted = extract_pdf(content, filename)
        else:
            extracted = extract_text_document(content, filename)

        merged_metadata = dict(metadata or {})
        merged_metadata.update(extracted["metadata"])
        return self._store_document(
            source_type=source_type,
            title=extracted["title"],
            text=extracted["text"],
            tags=tags or [],
            canonical_uri=canonical_uri,
            external_ref=external_ref,
            metadata=merged_metadata,
            raw_text=extracted["raw_text"],
            raw_binary=content,
            raw_binary_name=filename,
            chunking=chunking,
        )

    def ingest_markdown_directory(
        self,
        *,
        directory: str,
        tags: list[str] | None = None,
        chunking: dict | None = None,
        recursive: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        directory_path = Path(directory).expanduser().resolve()
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        pattern = "**/*.md" if recursive else "*.md"
        markdown_files = sorted(path for path in directory_path.glob(pattern) if path.is_file())
        total = len(markdown_files)
        items: list[dict[str, Any]] = []
        ingested = 0
        skipped = 0
        duplicates = 0
        errors = 0

        for index, path in enumerate(markdown_files, start=1):
            if progress_callback:
                progress_callback(index - 1, total, path.name)

            file_path = str(path)
            logged = self.database.get_ingestion_log_entry(file_path)
            if logged and logged["status"] in {"ingested", "duplicate"}:
                skipped += 1
                items.append(
                    {
                        "status": "skipped_logged",
                        "filename": path.name,
                        "file_path": file_path,
                        "source_id": logged["source_id"],
                        "logged_at": logged["created_at"],
                    }
                )
                continue

            try:
                content = path.read_bytes()
                raw_text = content.decode("utf-8", errors="replace")
                content_hash = sha256_text(raw_text.strip())
                result = self.ingest_file(
                    filename=path.name,
                    content=content,
                    tags=tags or [],
                    source_type="markdown_directory",
                    canonical_uri=path.as_uri(),
                    external_ref=str(directory_path),
                    metadata={
                        "directory": str(directory_path),
                        "file_path": file_path,
                    },
                    chunking=chunking,
                )
                result["filename"] = path.name
                result["file_path"] = file_path
                if result["status"] == "duplicate":
                    duplicates += 1
                    log_status = "duplicate"
                    message = "Already present in vector database."
                else:
                    ingested += 1
                    log_status = "ingested"
                    message = f"Stored {result.get('chunk_count', 0)} chunks."

                self.database.record_ingestion_log(
                    file_path=file_path,
                    filename=path.name,
                    content_hash=content_hash,
                    source_id=result.get("source_id"),
                    status=log_status,
                    message=message,
                )
                items.append(result)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                self.database.record_ingestion_log(
                    file_path=file_path,
                    filename=path.name,
                    content_hash=None,
                    source_id=None,
                    status="error",
                    message=str(exc),
                )
                items.append({"status": "error", "filename": path.name, "file_path": file_path, "error": str(exc)})

        if progress_callback:
            progress_callback(total, total, "Completed")

        return {
            "status": "ok",
            "directory": str(directory_path),
            "file_count": total,
            "ingested": ingested,
            "skipped_logged": skipped,
            "duplicates": duplicates,
            "errors": errors,
            "items": items,
        }

    def sync_telegram(self) -> dict:
        if not self.config["telegram"]["enabled"]:
            return {"status": "disabled", "message": "Telegram ingestion is disabled in config."}

        token = self.config.get("telegram", {}).get("bot_token")
        if not token:
            return {"status": "error", "message": "Telegram bot token is missing."}

        client = TelegramBotClient(token)
        last_offset = self.database.get_state("telegram_offset")
        offset = int(last_offset) if last_offset else None
        limit = int(self.config["telegram"]["poll_limit"])
        updates = client.fetch_updates(offset=offset, limit=limit)

        results = []
        next_offset = offset or 0
        for update in updates:
            next_offset = max(next_offset, int(update["update_id"]) + 1)
            message = update.get("message") or update.get("edited_message") or update.get("channel_post")
            if not message:
                continue
            results.extend(self._process_telegram_message(message, client))

        if updates:
            self.database.set_state("telegram_offset", str(next_offset))

        return {"status": "ok", "processed": len(results), "items": results}

    def _process_telegram_message(self, message: dict, client: TelegramBotClient) -> list[dict]:
        results: list[dict] = []
        chat_id = message.get("chat", {}).get("id", "unknown")
        message_id = message.get("message_id", "unknown")
        message_uri = f"telegram://chat/{chat_id}/message/{message_id}"
        forwarded = "forward_origin" in message or "forward_from" in message or "forward_date" in message
        text = (message.get("text") or message.get("caption") or "").strip()
        urls = [normalize_url(url) for url in extract_urls(text)]
        metadata = {
            "telegram_chat_id": chat_id,
            "telegram_message_id": message_id,
            "forwarded": forwarded,
        }

        for url in urls:
            try:
                result = self.ingest_url(url=url, tags=["telegram"])
                result["telegram_message_uri"] = message_uri
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                results.append({"status": "error", "url": url, "error": str(exc)})

        if message.get("document"):
            document = message["document"]
            filename = document.get("file_name") or f"{document.get('file_unique_id', 'telegram_file')}.bin"
            content = client.download_file(document["file_id"])
            results.append(
                self.ingest_file(
                    filename=filename,
                    content=content,
                    tags=["telegram"],
                    source_type="telegram_document",
                    canonical_uri=f"{message_uri}/document/{document.get('file_unique_id', filename)}",
                    external_ref=message_uri,
                    metadata=metadata,
                )
            )

        if text and not urls:
            title_prefix = "Forwarded Telegram note" if forwarded else "Telegram note"
            title = f"{title_prefix} {message_id}"
            results.append(
                self.ingest_text(
                    title=title,
                    text=text,
                    tags=["telegram"],
                    source_type="telegram_forward" if forwarded else "telegram_text",
                    canonical_uri=message_uri,
                    external_ref=message_uri,
                    metadata=metadata,
                )
            )

        return results

    def _get_notion_client(self):
        if not self.config.get("notion", {}).get("enabled", False):
            raise ValueError("Notion ingestion is disabled in config.")
        token = self.config.get("notion", {}).get("api_token")
        if not token:
            raise ValueError("Notion API token is missing in config/secrets.toml.")

        from notionhelper import NotionHelper

        return NotionHelper(
            notion_token=token,
            request_timeout=float(self.config["notion"]["request_timeout_seconds"]),
        )

    def _normalize_notion_id(self, value: str) -> str:
        cleaned = (value or "").strip().replace("-", "")
        if not re.fullmatch(r"[0-9a-fA-F]{32}", cleaned):
            raise ValueError("Data source ID must be a valid Notion UUID.")
        cleaned = cleaned.lower()
        return (
            f"{cleaned[0:8]}-{cleaned[8:12]}-{cleaned[12:16]}-"
            f"{cleaned[16:20]}-{cleaned[20:32]}"
        )

    def _dataframe_records(self, dataframe) -> list[dict]:
        json_blob = dataframe.to_json(orient="records", date_format="iso")
        return json.loads(json_blob)

    def _extract_notion_title(self, properties: Any, fallback: str) -> str:
        if isinstance(properties, list):
            text = self._rich_text_to_plain_text(properties)
            return text or fallback

        if not isinstance(properties, dict):
            return fallback

        for property_value in properties.values():
            if not isinstance(property_value, dict):
                continue
            if property_value.get("type") == "title":
                text = self._rich_text_to_plain_text(property_value.get("title", []))
                if text:
                    return text

        return fallback

    def _notion_markdown_with_children(self, payload: dict) -> tuple[str, list[str]]:
        """Flatten notionhelper's expanded child pages into one searchable document."""
        root_content = self._strip_notion_html(payload.get("content", ""))

        child_page_ids: list[str] = []
        visited: set[str] = set()
        sections = [root_content.strip()] if root_content.strip() else []
        for child in payload.get("child_pages", []):
            child_markdown, child_ids, title = self._notion_child_page_section(child, visited)
            if child_markdown:
                sections.append(f"## Child page: {title}\n\n{child_markdown}")
            child_page_ids.extend(child_ids)

        return "\n\n".join(sections).strip(), list(dict.fromkeys(child_page_ids))

    def _notion_child_page_section(
        self, child: dict, visited: set[str]
    ) -> tuple[str, list[str], str]:
        child_id = child.get("id")
        page = child.get("page", {})
        if not isinstance(page, dict):
            page = {}
        if isinstance(child_id, str):
            if child_id in visited:
                return "", [], child.get("title") or child_id[:8]
            visited.add(child_id)

        title = child.get("title") or self._extract_notion_title(
            page.get("properties", {}),
            fallback=child_id[:8] if isinstance(child_id, str) else "Untitled child page",
        )
        content = self._strip_notion_html(page.get("content", ""))

        ids = [child_id] if isinstance(child_id, str) else []
        sections = [content.strip()] if content.strip() else []
        for nested_child in page.get("child_pages", []):
            nested_markdown, nested_ids, nested_title = self._notion_child_page_section(
                nested_child, visited
            )
            if nested_markdown:
                sections.append(f"### Child page: {nested_title}\n\n{nested_markdown}")
            ids.extend(nested_ids)

        return "\n\n".join(sections).strip(), list(dict.fromkeys(ids)), str(title)

    def _strip_notion_html(self, content: Any) -> str:
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=2)

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        for tag_name in ("empty-block", "database", "page"):
            for tag in soup.find_all(tag_name):
                tag.decompose()

        heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
        for tab in soup.find_all("tab"):
            # The first direct text node is the tab label, not page content.
            for child in list(tab.children):
                if isinstance(child, NavigableString) and child.strip():
                    child.extract()
                    break

            heading_text = " ".join(
                heading.get_text(" ", strip=True) for heading in tab.find_all(heading_tags)
            ).strip()
            tab_text = tab.get_text(" ", strip=True)
            if not tab_text or tab_text == heading_text:
                tab.decompose()

        for tabs in soup.find_all("tabs"):
            tabs.unwrap()

        block_tags = {
            "br", "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
            "blockquote", "pre", "tr", "table", "tab", "tabs", "page", "empty-block",
        }
        for tag in soup.find_all(block_tags):
            tag.insert_after("\n")

        cleaned_lines = []
        for line in soup.get_text(" ").splitlines():
            normalized = re.sub(r"\s+", " ", line).strip()
            normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
            if normalized:
                cleaned_lines.append(normalized)
        return "\n".join(cleaned_lines)

    def _rich_text_to_plain_text(self, items: Any) -> str:
        if not isinstance(items, list):
            return ""
        parts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            plain_text = item.get("plain_text")
            if plain_text:
                parts.append(str(plain_text))
                continue
            text_payload = item.get("text")
            if isinstance(text_payload, dict) and text_payload.get("content"):
                parts.append(str(text_payload["content"]))
        return "".join(parts).strip()

    def _row_display_title(self, row: dict, fallback: str) -> str:
        for key, value in row.items():
            if value in (None, "", []):
                continue
            if key.lower() in {"title", "name"}:
                return str(value)
        return fallback

    def _store_document(
        self,
        *,
        source_type: str,
        title: str,
        text: str,
        tags: list[str],
        canonical_uri: str | None,
        external_ref: str | None,
        metadata: dict,
        raw_text: str,
        raw_binary: bytes | None,
        raw_binary_name: str | None,
        chunking: dict | None = None,
    ) -> dict:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Extracted content is empty.")

        content_hash = sha256_text(clean_text)
        duplicate = self.database.find_duplicate(canonical_uri, content_hash)
        if duplicate:
            return {
                "status": "duplicate",
                "source_id": int(duplicate["id"]),
                "title": duplicate["title"],
                "canonical_uri": duplicate["canonical_uri"],
            }

        slug = sanitize_filename(title)[:80]
        source_dir = self.raw_dir / slug
        suffix_index = 1
        while source_dir.exists():
            suffix_index += 1
            source_dir = self.raw_dir / f"{slug}_{suffix_index}"
        source_dir.mkdir(parents=True, exist_ok=True)

        raw_text_path = source_dir / "source.txt"
        raw_text_path.write_text(raw_text, encoding="utf-8")
        raw_binary_path = None
        if raw_binary is not None and raw_binary_name:
            raw_binary_path = source_dir / sanitize_filename(raw_binary_name)
            raw_binary_path.write_bytes(raw_binary)

        chunking = chunking or {"strategy": "semantic"}
        strategy = chunking.get("strategy", "semantic")
        if strategy == "fixed":
            chunker = FixedChunker(
                chunk_size=int(chunking.get("chunk_size", self.config["chunking"]["fixed_chunk_size"])),
                overlap=int(chunking.get("overlap", self.config["chunking"]["fixed_chunk_overlap"])),
            )
        elif strategy == "semantic":
            chunker = self.chunker
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        chunks = chunker.chunk(clean_text, self.ollama.embed_texts)
        validation_enabled = bool(chunking.get("validate_chunks", False))
        validation_model = chunking.get("validation_model")
        rejected_chunk_count = 0
        if validation_enabled:
            if not validation_model:
                raise ValueError("A chunk validation model is required when LLM validation is enabled.")
            meaningful_chunks = []
            for chunk in chunks:
                if self.ollama.is_meaningful_chunk(chunk.text, model=validation_model):
                    meaningful_chunks.append(chunk)
                else:
                    rejected_chunk_count += 1
            chunks = meaningful_chunks
            if not chunks:
                raise ValueError("The LLM rejected all chunks as non-meaningful content.")

        embeddings = self.ollama.embed_texts([chunk.text for chunk in chunks])
        stored_metadata = dict(metadata)
        stored_metadata["chunking"] = {
            "strategy": strategy,
            **(
                {
                    "chunk_size": int(chunking.get("chunk_size", self.config["chunking"]["fixed_chunk_size"])),
                    "overlap": int(chunking.get("overlap", self.config["chunking"]["fixed_chunk_overlap"])),
                }
                if strategy == "fixed"
                else {}
            ),
        }
        stored_metadata["chunk_validation"] = {
            "enabled": validation_enabled,
            "model": validation_model if validation_enabled else None,
            "rejected_chunk_count": rejected_chunk_count,
        }

        source_id = self.database.insert_source(
            source_type=source_type,
            title=title,
            canonical_uri=canonical_uri,
            external_ref=external_ref,
            content_hash=content_hash,
            summary=clean_text[:280],
            tags=tags,
            metadata=stored_metadata,
            raw_text_path=str(raw_text_path),
            raw_binary_path=str(raw_binary_path) if raw_binary_path else None,
        )
        self.database.replace_chunks(source_id, chunks, embeddings)

        return {
            "status": "ingested",
            "source_id": source_id,
            "title": title,
            "chunk_count": len(chunks),
            "tags": tags,
            "canonical_uri": canonical_uri,
        }
