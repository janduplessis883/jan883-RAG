from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup
from pypdf import PdfReader
from readability import Document
import requests


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    clean_query = [(key, value) for key, value in parse_qsl(parsed.query) if key not in TRACKING_PARAMS]
    clean_path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            clean_path,
            "",
            urlencode(clean_query),
            "",
        )
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_filename(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return base or "document"


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s<>\"]+", text)


def extract_article(url: str) -> dict:
    response = requests.get(
        url,
        timeout=60,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        },
    )
    response.raise_for_status()
    html = response.text

    document = Document(html)
    soup_for_title = BeautifulSoup(html, "html.parser")
    html_title = soup_for_title.title.string.strip() if soup_for_title.title and soup_for_title.title.string else None
    title = document.short_title() or html_title or url
    summary_html = document.summary(html_partial=True)
    soup = BeautifulSoup(summary_html, "html.parser")
    paragraphs = [node.get_text(" ", strip=True) for node in soup.find_all(["p", "li", "h1", "h2", "h3"])]
    text = "\n\n".join(part for part in paragraphs if part)

    if not text.strip():
        plain_soup = BeautifulSoup(html, "html.parser")
        text = plain_soup.get_text(" ", strip=True)

    return {
        "title": title.strip(),
        "text": re.sub(r"\n{3,}", "\n\n", text).strip(),
        "canonical_uri": normalize_url(url),
        "metadata": {"fetch_url": url},
        "raw_text": html,
    }


def extract_pdf(content: bytes, filename: str) -> dict:
    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    return {
        "title": Path(filename).stem,
        "text": text.strip(),
        "canonical_uri": None,
        "metadata": {"filename": filename, "page_count": len(reader.pages)},
        "raw_text": text,
    }


def extract_text_document(content: bytes, filename: str) -> dict:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = content.decode("utf-8", errors="ignore")

    return {
        "title": Path(filename).stem,
        "text": text.strip(),
        "canonical_uri": None,
        "metadata": {"filename": filename},
        "raw_text": text,
    }
