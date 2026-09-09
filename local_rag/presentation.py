"""Presentation transformations shared by pages and tests."""
from __future__ import annotations


def filter_sources(rows, query="", tags=(), types=(), start=None, end=None):
    return [row for row in rows
            if (not query or query.casefold() in row["title"].casefold())
            and (not tags or set(tags).issubset(row.get("tags", [])))
            and (not types or row["source_type"] in types)
            and (not start or row["created_at"][:10] >= str(start))
            and (not end or row["created_at"][:10] <= str(end))]


def ingestion_counts(result):
    items = result.get("items", [result])
    return {
        "added": result.get("ingested", sum(item.get("status") in {"ingested", "updated", "restored"} for item in items)),
        "duplicates": result.get("duplicates", sum(item.get("status") == "duplicate" for item in items)),
        "skipped": result.get("skipped_logged", sum(item.get("status", "").startswith("skipped") for item in items)),
        "failed": result.get("errors", sum(item.get("status") == "error" for item in items)),
    }


def export_conversation(messages):
    blocks = ["# Knowledge base conversation"]
    for message in messages:
        blocks.append(f"## {message['role'].capitalize()}\n\n{message['content']}")
        if message.get("error"):
            blocks.append("*Response interrupted.*")
        for i, source in enumerate(message.get("sources", []), 1):
            blocks.append(f"### [S{i}] {source['title']}\n\n{source.get('canonical_uri') or 'Local document'}\n\n{source.get('text', source.get('preview', ''))}")
    return "\n\n".join(blocks)
