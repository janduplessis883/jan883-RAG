import pytest

from local_rag.config import ConfigManager
from local_rag.database import Database
from local_rag.ingestion import IngestionService
from local_rag.retrieval import SearchService
from local_rag.chat import ChatService


PAGE_ID = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'


def page(page_id=PAGE_ID, name='Sarah Jones', tel='+44 (0)20 7946 0123'):
    return {
        'id': page_id, 'url': 'https://www.notion.so/' + page_id.replace('-', ''),
        'last_edited_time': '2026-10-01T10:00:00Z', 'created_time': '2020-01-01T00:00:00Z',
        'properties': {
            'Name / Company': {'type': 'title', 'title': [{'plain_text': name}]},
            'Description': {'type': 'rich_text', 'rich_text': [{'plain_text': 'Kitchen renovation specialist'}]},
            'Address': {'type': 'place', 'place': {'name': 'Office', 'address': '10 High Street, London'}},
            'Email': {'type': 'email', 'email': 'Sarah@Example.com'},
            'Tel': {'type': 'phone_number', 'phone_number': tel},
            'URL': {'type': 'url', 'url': 'https://example.com'},
            'Files & media': {'type': 'files', 'files': [
                {'name': 'Brochure.pdf', 'type': 'file', 'file': {'url': 'https://files.example/expired', 'expiry_time': '2020-01-01T00:00:00Z'}},
                {'name': 'Portfolio', 'type': 'external', 'external': {'url': 'https://example.com/portfolio'}},
            ]},
        },
    }


@pytest.fixture
def services(tmp_path, monkeypatch):
    from local_rag.contacts import NotionContactsService
    config = ConfigManager(tmp_path)
    config.ensure_defaults()
    config.write_text(config.settings_path, config.read_text(config.settings_path).replace('embedding_dimensions = 768', 'embedding_dimensions = 3').replace('contacts_data_source_id = ""', 'contacts_data_source_id = "contacts"'))
    config.write_text(config.secrets_path, '[notion]\napi_token = "test-token"\n')
    db = Database(config)
    db.initialize()
    ingestion = IngestionService(config, db)
    embedded = []
    def embed(texts):
        embedded.extend(texts)
        return [[1., 0., 0.] for _ in texts]
    monkeypatch.setattr(ingestion.ollama, 'embed_texts', embed)
    service = NotionContactsService(config, db, ingestion)
    # Remote responses are the only substituted dependency besides model calls.
    monkeypatch.setattr(service, 'fetch_all_pages', lambda: [page()])
    yield config, db, ingestion, service, embedded
    db.connection.close()


def test_contact_parses_place_and_preserves_exact_fields():
    from local_rag.contacts import Contact
    contact = Contact.from_notion_page(page())
    assert contact.name == 'Sarah Jones'
    assert contact.address == '10 High Street, London'
    assert contact.email == 'Sarah@Example.com'
    assert contact.tel == '+44 (0)20 7946 0123'
    assert contact.attachments[0]['name'] == 'Brochure.pdf'
    assert contact.attachments[0]['expiry_time'] == '2020-01-01T00:00:00Z'
    assert 'Kitchen renovation specialist' in contact.embedding_text()
    assert '+44' not in contact.embedding_text()


def test_sync_keeps_identity_and_updates_phone_without_embeddings(services):
    _, db, _, service, embedded = services
    assert service.sync()['upserted'] == 1
    original = db.list_contacts()[0]
    initial_embeddings = len(embedded)
    changed = page(tel='020 7000 9999')
    changed['last_edited_time'] = '2026-10-02T10:00:00Z'
    service.fetch_all_pages = lambda: [changed]
    assert service.sync()['upserted'] == 1
    current = db.list_contacts()[0]
    assert current['source_id'] == original['source_id']
    assert current['tel'] == '020 7000 9999'
    assert len(embedded) == initial_embeddings
    assert db.search_contacts('What is Sarah’s telephone number?')[0]['tel'] == '020 7000 9999'
    assert db.search_contacts('Who owns 020-7000-9999?')[0]['name'] == 'Sarah Jones'
    assert db.search_contacts('SARAH@example.com')[0]['name'] == 'Sarah Jones'


def test_description_updates_existing_vectors_and_identical_contacts_are_distinct(services):
    _, db, _, service, embedded = services
    pages = [page(), page('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')]
    service.fetch_all_pages = lambda: pages
    assert service.sync()['errors'] == 0
    before = len(embedded)
    assert len({row['source_id'] for row in db.list_contacts()}) == 2
    assert service.sync()['unchanged'] == 2
    assert len(embedded) == before
    for item in pages:
        item['properties']['Description']['rich_text'][0]['plain_text'] = 'Bathroom renovations'
    result = service.sync()
    assert result['errors'] == 0, result
    assert len(embedded) > before
    for row in db.list_contacts():
        assert 'Bathroom renovations' in db.get_source(row['source_id'])['full_text']


def test_failed_or_malformed_scan_preserves_contacts_and_removal_hides_vectors(services):
    _, db, _, service, _ = services
    service.sync()
    def failed():
        raise RuntimeError('Notion unavailable')
    service.fetch_all_pages = failed
    with pytest.raises(RuntimeError):
        service.sync()
    assert len(db.list_contacts()) == 1
    service.fetch_all_pages = lambda: [{'id': 'other', 'properties': {}}]
    assert service.sync()['errors'] == 1
    assert len(db.list_contacts()) == 1
    service.fetch_all_pages = lambda: []
    assert service.sync()['deleted'] == 1
    assert db.list_contacts() == []
    assert db.contact_source_ids() == []
    assert db.search_lexical('Sarah', 10) == []


def test_lookup_respects_scopes_ambiguity_and_missing_fields(services):
    _, db, _, service, _ = services
    other = page('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'Sarah Smith')
    other['properties']['Email']['email'] = None
    service.fetch_all_pages = lambda: [page(), other]
    service.sync()
    assert len(db.search_contacts('Sarah')) == 2
    assert db.search_contacts('Sarah', source_ids=[]) == []
    assert len(db.search_contacts('Sarah', source_ids=[db.list_contacts()[0]['source_id']])) == 1
    assert [r['name'] for r in db.search_contacts('List all contacts without an email address')] == ['Sarah Smith']
    assert db.search_contacts('What is the telephone number for Nobody?') == []
    assert db.search_contacts('Who owns +44 20 7946 0123?')[0]['name'] == 'Sarah Jones'
    assert len(db.search_contacts('List all contacts')) == 2


def test_expired_attachment_refresh_and_fallback(services, monkeypatch):
    from local_rag.contacts import hydrate_contacts
    config, db, _, service, _ = services
    service.sync()
    updated = page()
    updated['properties']['Files & media']['files'][0]['file'] = {'url': 'https://files.example/fresh', 'expiry_time': '2099-01-01T00:00:00Z'}
    monkeypatch.setattr('notionhelper.NotionHelper.get_page', lambda *a, **kw: updated)
    row = hydrate_contacts(config, db, db.list_contacts())[0]
    assert row['attachments'][0]['url'] == 'https://files.example/fresh'
    assert db.list_contacts()[0]['attachments'][0]['url'] == 'https://files.example/fresh'
    service.sync()
    def fail(*a, **kw):
        raise RuntimeError('offline')
    monkeypatch.setattr('notionhelper.NotionHelper.get_page', fail)
    row = hydrate_contacts(config, db, db.list_contacts())[0]
    assert row['attachments'][0]['url'] == row['notion_url']
    assert row['attachments'][0]['open_in_notion'] is True
    assert row['attachments'][1]['url'] == 'https://example.com/portfolio'


def test_chat_exact_contacts_and_semantic_contacts_use_structured_context(services):
    config, db, _, service, _ = services
    # Use an external URL only, so retrieval requires no remote refresh.
    current = page()
    current['properties']['Files & media']['files'] = current['properties']['Files & media']['files'][1:]
    service.fetch_all_pages = lambda: [current]
    service.sync()
    search = SearchService(config, db)
    search.ollama.embed_texts = lambda texts: [[1., 0., 0.] for _ in texts]
    chat = ChatService(config, search)
    records = chat.retrieve_sources('What is Sarah’s telephone number?', include_contacts=True, include_calendar=False, include_knowledge_base=False)
    assert records[0]['retrieval_mode'] == 'contact_exact'
    assert records[0]['contact']['tel'] == '+44 (0)20 7946 0123'
    assert chat.retrieve_sources('Sarah', include_contacts=False, include_calendar=False, include_knowledge_base=True) == []
    semantic = chat.retrieve_sources('kitchen renovations', include_contacts=True, include_calendar=False, include_knowledge_base=False)
    assert semantic[0]['contact']['email'] == 'Sarah@Example.com'
    system, prompt, _, _ = chat._answer_request(question='Sarah phone?', sources=records, related_questions=None, source_limit=None, history=None)
    assert '+44 (0)20 7946 0123' in prompt
    assert 'https://example.com/portfolio' in prompt
    assert 'multiple contacts' in system.lower()


def test_direct_lookup_does_not_depend_on_embedding_service(services):
    config, db, _, service, _ = services
    current = page()
    current['properties']['Files & media']['files'] = []
    service.fetch_all_pages = lambda: [current]
    service.sync()
    search = SearchService(config, db)
    def offline(*args, **kwargs):
        raise RuntimeError('Model offline')
    search.ollama.embed_texts = offline
    chat = ChatService(config, search)
    records = chat.retrieve_sources('What is Sarah’s telephone number?', include_contacts=True)
    assert records[0]['contact']['name'] == 'Sarah Jones'
    assert chat.retrieve_sources('What is Nobody’s telephone number?', include_contacts=True, include_calendar=False, include_knowledge_base=False) == []


def test_contact_listing_is_not_limited_by_vector_top_k(services):
    config, db, _, service, _ = services
    pages = [page(f'{n:032x}', f'Person {n}') for n in range(1, 5)]
    for current in pages:
        current['properties']['Files & media']['files'] = []
    service.fetch_all_pages = lambda: pages
    service.sync()
    chat = ChatService(config, SearchService(config, db))
    sources = chat.retrieve_sources('List all contacts', source_limit=1, include_calendar=False, include_knowledge_base=False)
    assert len(sources) == 4


def test_contacts_selector_and_directory_render(services, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from local_rag.backup import BackupService
    import app_pages.common as common
    config, db, ingestion, service, _ = services
    current = page()
    current['properties']['Files & media']['files'] = current['properties']['Files & media']['files'][1:]
    service.fetch_all_pages = lambda: [current]
    service.sync()
    search = SearchService(config, db)
    chat = ChatService(config, search)
    monkeypatch.setattr(common, 'model_health', lambda settings: {'available': True, 'models': ['test']})
    runtime = (config, db, ingestion, search, chat, BackupService(config))
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(root / 'app_pages/chat.py')
    app.session_state['runtime'] = runtime
    app.run()
    assert not app.exception
    selector = next(item for item in app.multiselect if item.label == 'Chat with')
    assert 'Contacts' in selector.options
    selector.set_value(['Contacts']).run()
    assert not app.exception
    directory = AppTest.from_file(root / 'app_pages/contacts.py')
    directory.session_state['runtime'] = runtime
    directory.run()
    assert not directory.exception
    assert 'Sarah Jones' in directory.dataframe[0].value['Name / Company'].tolist()


def test_qualified_contact_questions_do_not_return_the_whole_directory(services):
    _, db, _, service, _ = services
    service.sync()
    assert db.search_contacts('List contacts who specialise in plumbing') == []


def test_empty_optional_fields_and_text_addresses():
    from local_rag.contacts import Contact
    current = page()
    current['properties']['Address'] = {'type': 'rich_text', 'rich_text': [{'plain_text': '20 Main Road'}]}
    current['properties']['Email']['email'] = None
    current['properties']['Tel']['phone_number'] = None
    current['properties']['Files & media']['files'] = []
    contact = Contact.from_notion_page(current)
    assert contact.address == '20 Main Road'
    assert contact.email == ''
    assert contact.tel == ''
    assert contact.attachments == []


def test_contact_directory_answers_include_every_match_without_model_calls(services):
    config, db, _, service, _ = services
    pages = [page(f'{n:032x}', f'Person {n}') for n in range(1, 5)]
    for current in pages:
        current['properties']['Files & media']['files'] = []
    service.fetch_all_pages = lambda: pages
    service.sync()
    chat = ChatService(config, SearchService(config, db))
    sources = chat.retrieve_sources('List all contacts', source_limit=1, include_calendar=False, include_knowledge_base=False)
    def offline(*args, **kwargs):
        raise RuntimeError('Model unavailable')
    chat.ollama.chat = offline
    chat.ollama.chat_stream = offline
    answer = chat.answer(question='List all contacts', model_name='test', sources=sources)['answer']
    assert '4 contacts' in answer
    assert all(f'Person {n}' in answer for n in range(1, 5))
    streamed = ''.join(chat.answer_stream(question='List all contacts', model_name='test', sources=sources))
    assert streamed == answer
