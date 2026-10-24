"""Source-linked Notion contacts and link-only attachments."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Callable

from notionhelper import NotionHelper

from local_rag.calendar import _plain


def property_value(properties: dict, name: str):
    prop = next((value for key, value in properties.items() if key.casefold() == name.casefold()), {})
    return prop.get(prop.get('type', ''))


def text_value(value) -> str:
    if isinstance(value, list):
        return _plain(value)
    return value.strip() if isinstance(value, str) else ''


def attachment_links(properties: dict) -> list[dict]:
    files = property_value(properties, 'Files & media') or []
    result = []
    for item in files:
        kind = item.get('type')
        payload = item.get(kind, {})
        url = payload.get('url', '')
        if url.startswith(('https://', 'http://')):
            result.append({'name': item.get('name') or 'Attachment', 'url': url,
                           'type': kind, 'expiry_time': payload.get('expiry_time')})
    return result


def normalize_phone(value: str) -> str:
    # Notion retains the display value; this key handles UK local/international forms.
    digits = re.sub(r'\D', '', value.replace('(0)', ''))
    if digits.startswith('00'):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith('0'):
        digits = '44' + digits[1:]
    return digits


@dataclass
class Contact:
    page_id: str
    name: str
    address: str
    description: str
    email: str
    tel: str
    url: str
    attachments: list[dict]
    notion_url: str
    created_time: str | None
    last_edited_time: str | None
    raw_json: dict

    @classmethod
    def from_notion_page(cls, page: dict) -> 'Contact':
        properties = page.get('properties', {})
        name_prop = next((value for key, value in properties.items() if key.casefold() == 'name / company'), None)
        if not isinstance(name_prop, dict) or name_prop.get('type') != 'title':
            raise ValueError('Contact is missing the Name / Company title property.')
        address = property_value(properties, 'Address')
        if isinstance(address, dict):
            address = address.get('address') or address.get('name') or ''
        page_id = page['id'].replace('-', '')
        return cls(
            page_id=page_id, name=text_value(property_value(properties, 'Name / Company')) or 'Unnamed contact',
            address=text_value(address), description=text_value(property_value(properties, 'Description')),
            email=text_value(property_value(properties, 'Email')), tel=text_value(property_value(properties, 'Tel')),
            url=text_value(property_value(properties, 'URL')), attachments=attachment_links(properties),
            notion_url=page.get('url') or f'https://www.notion.so/{page_id}',
            created_time=page.get('created_time'), last_edited_time=page.get('last_edited_time'), raw_json=page,
        )

    def embedding_text(self) -> str:
        return f'Contact: {self.name}\nAddress: {self.address}\nDescription: {self.description}'.strip()


def contact_listing_query(query: str) -> bool:
    normalized = query.casefold().strip().rstrip('.?!')
    prefix = r"(?:please )?(?:(?:list|show|give|find)(?: me)? )?(?:all |my )?contacts?"
    fields = r"(?:an? |their )?(?:e-?mail(?: address)?|phone(?: number)?|telephone(?: number)?|tel|address|website|url|description|attachments?|files)(?:s)?"
    predicate = r"(?:without |missing |with no |(?:who|that) have no )" + fields
    return bool(re.fullmatch(prefix + r"(?: " + predicate + r")?", normalized))


def contact_matches(records: list[dict], query: str) -> list[dict]:
    """Deterministic name, email, phone and missing-field lookup over scoped rows."""
    normalized = re.sub(r"['’]s\b", '', query.casefold())
    emails = re.findall(r'[\w.+-]+@[\w.-]+\.[\w-]+', normalized)
    phones = [normalize_phone(match) for match in re.findall(r'(?:\+|00)?\d[\d\s().-]{5,}\d', normalized)]
    if emails or phones:
        return [row for row in records if row['email_normalized'] in emails or (row['tel_normalized'] and row['tel_normalized'] in phones)]
    if contact_listing_query(normalized):
        field = next((field for pattern, field in (
            (r'email|e-mail', 'email'), (r'phone|telephone|\btel\b', 'tel'),
            (r'address', 'address'), (r'website|\burl\b', 'url'),
            (r'description', 'description'), (r'attachment|files', 'attachments'),
        ) if re.search(pattern, normalized)), None)
        if field and re.search(r'\b(?:without|missing|no)\b', normalized):
            return [row for row in records if not row[field]]
        return records
    stopwords = set('what whats who is are the a an of for do does i me my we our you your know get give find please contact contacts company name number phone telephone tel email e mail address website url details information attachment attachments file files media open link links send call have has show tell can could would s'.split())
    words = set(re.findall(r'\w+', normalized)) - stopwords
    if not words:
        return []
    return [row for row in records if words.intersection(set(re.findall(r'\w+', row['name'].casefold())))]


class NotionContactsService:
    def __init__(self, config_manager, database, ingestion=None):
        self.config_manager, self.database, self.ingestion = config_manager, database, ingestion
        notion = config_manager.load_merged().get('notion', {})
        self.data_source_id = notion.get('contacts_data_source_id', '').strip()
        self.token = notion.get('api_token', '')
        self.client = NotionHelper(self.token, request_timeout=float(notion.get('request_timeout_seconds', 30))) if self.token else None

    def fetch_all_pages(self) -> list[dict]:
        if not self.client:
            raise ValueError('Configure the Notion API token before syncing contacts.')
        schema = self.client.get_data_source(self.data_source_id)
        properties = schema.get('properties', {})
        title = next((p for name, p in properties.items() if name.casefold() == 'name / company'), {})
        if title.get('type') != 'title':
            raise ValueError('The configured data source is not a contacts database: Name / Company title is missing.')
        # Fully materialize before reconciliation: a failed later page must never delete local rows.
        return list(self.client.iter_data_source_pages(self.data_source_id, page_size=100))

    def sync(self, *, full_sync=False, progress_callback: Callable | None = None) -> dict:
        if not self.data_source_id:
            return {'status': 'skipped', 'message': 'Configure notion.contacts_data_source_id to enable contact syncing.'}
        pages = self.fetch_all_pages()
        existing = {row['notion_page_id']: row for row in self.database.list_contacts(include_removed=True, data_source_id=self.data_source_id)}
        upserted = unchanged = deleted = 0
        errors = []
        if progress_callback:
            progress_callback(0, len(pages), 'Starting contacts sync')
        for index, page in enumerate(pages, 1):
            try:
                if page.get('archived') or page.get('in_trash'):
                    continue
                contact = Contact.from_notion_page(page)
                old = existing.get(contact.page_id)
                # Compare content rather than timestamp alone; URLs can renew without a page edit.
                same = old and all(old[field] == getattr(contact, field) for field in (
                    'name', 'address', 'description', 'email', 'tel', 'url', 'attachments', 'last_edited_time'))
                source = self.database.get_source(old['source_id']) if old else None
                if not full_sync and same and source and not source.get('deleted_at'):
                    unchanged += 1
                else:
                    self._upsert_contact(contact)
                    upserted += 1
            except Exception as exc:
                errors.append({'page_id': page.get('id'), 'error': str(exc)})
            if progress_callback:
                progress_callback(index, len(pages), 'Processing contacts')
        # A malformed page makes the listing unsafe to reconcile, even if transport succeeded.
        if not errors:
            active_seen = {p['id'].replace('-', '') for p in pages if not p.get('archived') and not p.get('in_trash')}
            for page_id, row in existing.items():
                if page_id not in active_seen and not row['deleted_at']:
                    self.database.set_source_removed(row['source_id'], True)
                    deleted += 1
        return {'status': 'partial' if errors else 'ok', 'page_count': len(pages), 'upserted': upserted,
                'unchanged': unchanged, 'deleted': deleted, 'errors': len(errors), 'error_details': errors[:50]}

    def _upsert_contact(self, contact: Contact):
        uri = f'notion://contact/{contact.page_id}'
        existing = self.database.find_source_by_uri(uri)
        text = contact.embedding_text()
        metadata = {'notion_page_id': contact.page_id, 'notion_data_source_id': self.data_source_id, 'contact': True}
        if existing:
            source_id = int(existing['id'])
            current = self.database.get_source(source_id)
            if current['deleted_at']:
                self.database.set_source_removed(source_id, False)
            if current['full_text'] != text:
                self.ingestion.update_source(source_id, title=contact.name, text=text, tags=['contacts'])
            self.database.update_source_metadata(source_id, {**current['metadata'], **metadata})
        else:
            result = self.ingestion.ingest_text(title=contact.name, text=text, tags=['contacts'], source_type='notion_contact',
                canonical_uri=uri, external_ref=f'notion://data-source/{self.data_source_id}', metadata=metadata,
                allow_duplicate_content=True)
            source_id = int(result['source_id'])
        self.database.upsert_contact(contact, source_id, self.data_source_id)


def _expired(link: dict, now: datetime) -> bool:
    if link.get('type') != 'file':
        return False
    try:
        expires = datetime.fromisoformat((link.get('expiry_time') or '').replace('Z', '+00:00'))
        return expires <= now + timedelta(minutes=5)
    except (TypeError, ValueError):
        return True


def hydrate_contacts(config_manager, database, records: list[dict]) -> list[dict]:
    """Return fresh attachment links; degrade to the owning Notion page if unavailable."""
    now = datetime.now(timezone.utc)
    service = None
    result = []
    for original in records:
        row = deepcopy(original)
        if any(_expired(link, now) for link in row['attachments']):
            try:
                service = service or NotionContactsService(config_manager, database)
                if not service.client:
                    raise ValueError('No Notion credentials configured')
                page = service.client.get_page(row['notion_page_id'], expand_child_pages=False)
                links = attachment_links(page['properties'])
                database.update_contact_attachments(row['notion_page_id'], links)
                row['attachments'] = links
            except Exception:
                for link in row['attachments']:
                    if _expired(link, now):
                        link['url'] = row['notion_url']
                        link['open_in_notion'] = True
        result.append(row)
    return result
