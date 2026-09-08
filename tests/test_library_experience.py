import json
import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from local_rag.backup import BackupService
from local_rag.chat import ChatService
from local_rag.chunking import SemanticChunk
from local_rag.config import ConfigManager
from local_rag.database import Database
from local_rag.ingestion import IngestionService
from local_rag.presentation import export_conversation, filter_sources, ingestion_counts
from local_rag.retrieval import SearchService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = ConfigManager(tmp_path / "workspace")
    config.ensure_defaults()
    settings = config.read_text(config.settings_path).replace('embedding_dimensions = 768', 'embedding_dimensions = 3')
    settings = settings.replace('/Volumes/BackupHD/jan883-RAG-Backup/', str(tmp_path / 'backups'))
    config.write_text(config.settings_path, settings)
    database = Database(config)
    database.initialize()
    ingestion = IngestionService(config, database)
    retrieval = SearchService(config, database)
    chat = ChatService(config, retrieval)
    monkeypatch.setattr(ingestion.ollama, 'embed_texts', lambda texts: [[1., 0., 0.] for _ in texts])
    monkeypatch.setattr(retrieval.ollama, 'embed_texts', lambda texts: [[1., 0., 0.] for _ in texts])
    import app_pages.common as common
    monkeypatch.setattr(common, 'model_health', lambda settings: {'available': True, 'models': ['test']})
    yield config, database, ingestion, retrieval, chat, BackupService(config)
    database.connection.close()


def add(runtime, title='Original', text='Original searchable passage', tags=None):
    return runtime[2].ingest_text(title=title, text=text, tags=tags or ['work'], source_type='text',
                                  chunking={'strategy':'fixed'})['source_id']


def page(runtime, name):
    at = AppTest.from_file(ROOT / 'app_pages' / f'{name}.py', default_timeout=15)
    at.session_state['runtime'] = runtime
    at.run()
    assert not at.exception
    return at


def test_trash_excludes_dense_lexical_and_restores(runtime):
    sid = add(runtime)
    db = runtime[1]
    assert runtime[3].search('Original')
    db.set_source_removed(sid, True)
    assert not db.library_sources()
    assert db.library_sources(removed=True)[0]['id'] == sid
    assert not db.search_lexical('Original', 8)
    assert not db.search_candidates([1., 0., 0.], 8)
    assert not runtime[3].search('Original')
    db.set_source_removed(sid, False)
    assert runtime[3].search('Original')


def test_update_replaces_text_and_fts(runtime):
    sid = add(runtime)
    runtime[2].update_source(sid, title='Replacement', text='New unique zebra evidence', tags=['personal'])
    db = runtime[1]
    assert db.get_source(sid)['full_text'] == 'New unique zebra evidence'
    assert db.get_source(sid)['tags'] == ['personal']
    assert db.search_lexical('zebra', 8)
    assert not db.search_lexical('Original', 8)


def test_failed_update_rolls_back_all_indexes(runtime):
    sid = add(runtime)
    db = runtime[1]
    with pytest.raises(ValueError):
        db.update_document(sid, title='Broken', text='Broken', tags=[], content_hash='new',
                           chunks=[SemanticChunk('Broken', 6)], embeddings=[])
    assert db.get_source(sid)['title'] == 'Original'
    assert db.search_lexical('Original', 8)
    assert db.search_candidates([1., 0., 0.], 8)


def test_duplicate_update_rolls_back(runtime):
    first = add(runtime)
    add(runtime, 'Second', 'Different document')
    with pytest.raises(sqlite3.IntegrityError):
        runtime[2].update_source(first, title='Conflict', text='Different document', tags=[])
    assert runtime[1].get_source(first)['title'] == 'Original'
    assert runtime[1].search_lexical('Original', 8)


def test_filtered_search_scopes_before_ranking(runtime):
    add(runtime)
    target = add(runtime, 'Second', 'Another searchable passage', ['personal'])
    assert all(row['source_id'] == target for row in runtime[3].search('passage', source_ids=[target]))
    assert runtime[3].search('passage', source_ids=[]) == []


def test_conversations_and_preferences_survive_reopen(runtime):
    config, db, *_ = runtime
    messages = [{'role':'user', 'content':'Question'}, {'role':'assistant','content':'Answer [S1]',
                 'sources':[{'title':'Evidence','text':'Complete passage', 'canonical_uri':None}]}]
    db.save_conversation('one', messages)
    db.set_state('search_preferences', json.dumps({'tags':['work']}))
    reopened = Database(config)
    try:
        reopened.initialize()
        assert reopened.load_conversation('one') == messages
        assert json.loads(reopened.get_state('search_preferences')) == {'tags':['work']}
        assert 'Complete passage' in export_conversation(reopened.load_conversation('one'))
    finally:
        reopened.connection.close()


def test_backup_verified_and_contains_database(runtime):
    add(runtime)
    result = runtime[5].run_backup()
    assert result['verified_at']
    assert runtime[1].last_successful_operation('backup')
    with sqlite3.connect(Path(result['destination']) / result['database']) as snapshot:
        assert snapshot.execute('SELECT COUNT(*) FROM sources').fetchone()[0] == 1
        assert snapshot.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_filters_and_counts():
    rows = [{'title':'Meeting', 'tags':['work','notes'], 'source_type':'text', 'created_at':'2026-09-05 12:00'}]
    assert filter_sources(rows, 'meet', ['work'], ['text'], '2026-09-05', '2026-09-05') == rows
    assert filter_sources(rows, tags=['other']) == []
    assert ingestion_counts({'ingested':12,'duplicates':3,'errors':1}) == {'added':12,'duplicates':3,'failed':1,'skipped':0}


@pytest.mark.parametrize('name', ['library', 'search', 'chat', 'ingest', 'dashboard', 'admin'])
def test_pages_render(runtime, name):
    add(runtime)
    page(runtime, name)


def test_chat_default_and_example_saved(runtime, monkeypatch):
    sid = add(runtime)
    chat = runtime[4]
    monkeypatch.setattr(chat, 'generate_related_questions', lambda prompt: [])
    monkeypatch.setattr(chat, 'answer_stream', lambda **kwargs: iter(['A grounded answer [S1].']))
    at = AppTest.from_file(ROOT / 'app.py', default_timeout=15)
    at.session_state['runtime'] = runtime
    at.run()
    assert not at.exception
    assert at.title[0].value == 'Ask your knowledge base'
    next(b for b in at.button if b.label == 'What decisions were made in the meeting notes?').click().run()
    assert not at.exception
    history = runtime[1].list_conversations()
    assert len(history) == 1
    messages = runtime[1].load_conversation(history[0]['id'])
    assert messages[-1]['content'] == 'A grounded answer [S1].'
    assert messages[-1]['sources'][0]['source_id'] == sid
    next(b for b in at.button if b.label == 'New conversation').click().run()
    assert not at.exception
    assert not at.session_state['chat_messages']
    at.selectbox(key='conversation_picker').select(history[0]['id']).run()
    assert at.session_state['chat_messages'] == messages


def test_search_results_survive_rerun(runtime):
    add(runtime)
    at = page(runtime, 'search')
    at.text_input[0].set_value('Original')
    next(b for b in at.button if b.label == 'Search').click().run()
    assert not at.exception
    assert runtime[1].get_state('search_results')
    fresh = page(runtime, 'search')
    assert fresh.text_input[0].value == 'Original'
    assert any(s.value == 'Original' for s in fresh.subheader)


def test_import_error_can_retry(runtime, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError('Temporary failure')
    monkeypatch.setattr(runtime[2], 'ingest_text', fail)
    at = page(runtime, 'ingest')
    next(w for w in at.text_area if w.label == 'Text').set_value('A document')
    next(b for b in at.button if b.label == 'Ingest text').click().run()
    assert not at.exception
    assert any('1 failed' in w.value for w in at.warning)
    monkeypatch.setattr(runtime[2], 'ingest_text', lambda **kwargs: {'status':'ingested','title':'A document'})
    next(b for b in at.button if b.label == 'Retry failed import').click().run()
    assert not at.exception
    assert any('Added 1 documents' in w.value for w in at.success)
