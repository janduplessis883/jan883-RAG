from __future__ import annotations

from datetime import datetime
from html import escape
import re
from typing import Any

import requests


class SummaryService:
    """Create practice-manager summaries and deliver them by transactional email."""

    EXCLUDED_SOURCE_TYPES = {"notion_calendar", "notion_contact", "contact"}

    def __init__(self, config_manager, database, ollama) -> None:
        self.config_manager = config_manager
        self.database = database
        self.ollama = ollama
        self.config = config_manager.load_merged()

    def should_summarise(self, source_type: str, canonical_uri: str | None) -> bool:
        if source_type in self.EXCLUDED_SOURCE_TYPES:
            return False
        return not re.match(r"notion://(?:calendar|contact)/", canonical_uri or "")

    def process(self, *, source_id: int, source_type: str, title: str,
                text: str, canonical_uri: str | None) -> dict[str, Any]:
        if not self.should_summarise(source_type, canonical_uri):
            return {"status": "skipped", "reason": "excluded_source_type", "source_id": source_id}

        try:
            summary = self._generate_summary(title, text)
            html = self._render_email(title, summary, canonical_uri)
            email_id = self._send_email(title, html, summary)
            result = {
                "status": "sent", "source_id": source_id, "title": title,
                "email_id": email_id,
            }
        except Exception as exc:  # email must never turn a successful ingestion into a failure
            result = {
                "status": "error", "source_id": source_id, "title": title,
                "error": str(exc),
            }
        self.database.record_operation("summary_email", result)
        return result

    def _generate_summary(self, title: str, text: str) -> str:
        settings = self.config.get("summary", {})
        max_chars = int(settings.get("max_input_chars", 60000))
        source_text = text if len(text) <= max_chars else text[:max_chars // 2] + "\n\n[Middle of document omitted]\n\n" + text[-max_chars // 2:]
        model = self.config["ollama"].get("default_answer_model", "default")
        system_prompt = (
            "You are a highly practical primary-care practice manager's briefing editor. "
            "Summarise the supplied source document for someone who may not have time to read it. "
            "Use Markdown only. Be accurate, distinguish explicit source claims from sensible implications, "
            "and never invent dates, requirements, clinical advice, or policy. "
            "Organise the briefing with these headings where relevant: Executive summary, "
            "What this is about, What a practice manager needs to know, Changes/developments, "
            "Actions and decisions, Risks or unanswered questions, and Useful source details. "
            "Prioritise operational impact, governance, workforce, finance, compliance, patient access, "
            "digital systems, deadlines, named organisations, and practical next steps. "
            "If a section is not supported by the source, omit it."
        )
        user_prompt = f"Source title: {title}\n\nDocument:\n{source_text}"
        return self.ollama.chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt).strip()

    def _send_email(self, title: str, html: str, summary: str) -> str | None:
        key = self.config.get("resend", {}).get("api_key")
        if not key:
            raise ValueError("RESEND_API_KEY is missing from .streamlit/secrets.toml.")
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "from": "jan883-RAG Summary Service <hello@attribut.me>",
                "to": ["jan.duplessis@nhs.net"],
                "subject": f"RAG summary: {title}",
                "html": html,
                "text": re.sub(r"\n{3,}", "\n\n", summary),
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("id")

    def _render_email(self, title: str, markdown: str, canonical_uri: str | None) -> str:
        body = self._markdown_to_html(markdown)
        source = f'<p style="margin:16px 0 0;font-size:11px;line-height:1.6;color:#65717b;overflow-wrap:break-word">Source: <a href="{escape(canonical_uri)}" style="color:#315e63">{escape(canonical_uri)}</a></p>' if canonical_uri and canonical_uri.startswith("http") else ""
        date = datetime.now().strftime("%d %B %Y")
        return f"""<!DOCTYPE html>
<html lang=\"en\"><head><meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"><title>{escape(title)}</title></head>
<body style=\"margin:0;padding:0;background-color:#f4f5f7;color:#303744;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:1.65;\">
<table role=\"presentation\" width=\"100%\" cellspacing=\"0\" cellpadding=\"0\" border=\"0\" style=\"width:100%;background-color:#f4f5f7;\"><tr><td align=\"center\" style=\"padding:28px 12px;\">
<table role=\"presentation\" width=\"600\" cellspacing=\"0\" cellpadding=\"0\" border=\"0\" style=\"width:100%;max-width:600px;background-color:#ffffff;border:1px solid #e3e7ec;border-radius:8px;\">
<tr><td style=\"padding:24px 28px 18px;border-top:3px solid #355c60;font-family:Arial,Helvetica,sans-serif;\"><p style=\"margin:0 0 8px;font-size:10px;font-weight:bold;letter-spacing:1.5px;text-transform:uppercase;color:#526d70;\">JAN883-RAG SUMMARY</p><h1 style=\"margin:0;font-size:20px;line-height:1.3;font-weight:normal;color:#253b3e;\">{escape(title)}</h1></td></tr>
<tr><td style=\"padding:0 28px 22px;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:1.65;color:#303744;\"><p style=\"margin:0 0 14px;\">Dear Jan,</p><p style=\"margin:0;\">Your ingested document has been summarised for practice-management review. <strong style=\"color:#253b3e;\">Prepared {date}.</strong></p></td></tr>
<tr><td style=\"padding:0 28px 24px;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:1.65;color:#303744;overflow-wrap:break-word;\"><h2 style=\"margin:0 0 12px;padding-top:20px;border-top:1px solid #e7ebee;font-size:14px;line-height:1.4;font-weight:bold;color:#253b3e;\">Preparation notes</h2><div style=\"font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:1.65;color:#303744;\">{body}{source}</div></td></tr>
<tr><td style=\"padding:16px 28px 20px;border-top:1px solid #e7ebee;font-family:Arial,Helvetica,sans-serif;font-size:11px;line-height:1.6;color:#65717b;\"><p style=\"margin:0;\">Happy reading,<br><strong style=\"font-weight:bold;color:#526168;\">jan883-RAG Summary Service</strong></p><p style=\"margin:8px 0 0;font-size:10px;\">Verify important requirements against the original source.</p></td></tr>
</table></td></tr></table></body></html>"""

    def _markdown_to_html(self, markdown: str) -> str:
        lines = markdown.splitlines()
        output: list[str] = []
        in_list = False
        for raw in lines:
            line = raw.strip()
            if not line:
                if in_list:
                    output.append("</ul>")
                    in_list = False
                continue
            heading = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading:
                if in_list:
                    output.append("</ul>")
                    in_list = False
                level = min(len(heading.group(1)) + 1, 4)
                output.append(f'<h{level} style="color:#294447;line-height:1.25;margin:24px 0 8px;font-size:{22 if level == 2 else 18}px">{self._inline(heading.group(2))}</h{level}>')
                continue
            bullet = re.match(r"^[-*+]\s+(.+)$", line)
            if bullet:
                if not in_list:
                    output.append('<ul style="padding-left:24px;font-size:14px;line-height:1.45;margin-top:4px;margin-bottom:14px">')
                    in_list = True
                output.append(f"<li>{self._inline(bullet.group(1))}</li>")
                continue
            if in_list:
                output.append("</ul>")
                in_list = False
            output.append(f'<p style="font-size:14px;margin:0 0 12px">{self._inline(line)}</p>')
        if in_list:
            output.append("</ul>")
        return "".join(output)

    @staticmethod
    def _inline(value: str) -> str:
        value = escape(value)
        value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
        value = re.sub(r"`(.+?)`", r"<code>\1</code>", value)
        return value
