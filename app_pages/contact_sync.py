"""Session-owned background contacts sync shared by Admin and Contacts."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import streamlit as st

from app_pages.common import runtime
from watcher.notion_sync import sync_contacts_once


@st.cache_resource
def _contacts_executor():
    return ThreadPoolExecutor(max_workers=1)


def _run_contacts_sync(config, database, ingestion, state):
    def progress(completed, total, label):
        with state['lock']:
            state.update(completed=completed, total=total, label=label)
    try:
        result = sync_contacts_once(config, database, ingestion, full_sync=True, progress_callback=progress)
        with state['lock']:
            state['result'] = result
    except Exception as exc:
        with state['lock']:
            state['error'] = str(exc)
        database.record_operation('sync:notion_contacts', {'status': 'error', 'error': str(exc)})


@st.fragment(run_every='1s')
def contacts_sync_panel():
    config, database, ingestion, *_ = runtime()
    job = st.session_state.get('contacts_sync_state')
    state = {}
    if job:
        with job['lock']:
            state = dict(job)
    future = st.session_state.get('contacts_sync_future')
    running = future is not None and not future.done()
    if running:
        total = state.get('total') or 0
        completed = state.get('completed', 0)
        st.progress(completed / total if total else 0,
                    text=f'{completed} of {total} contacts processed' if total else 'Fetching contacts…')
    elif state.get('error'):
        st.error(f"Contacts sync failed: {state['error']}")
    elif state.get('result'):
        result = state['result']
        if result['status'] == 'skipped':
            st.info(result['message'])
        elif result.get('errors'):
            st.warning(f"Contacts sync: {result['upserted']} stored, {result['errors']} errors.")
            with st.expander('Contacts sync errors'):
                st.json(result.get('error_details', []))
        else:
            st.success(f"Contacts synced: {result['upserted']} stored, {result['unchanged']} unchanged, {result['deleted']} removed.")
    if st.button('Sync contacts', icon=':material/sync:', disabled=running):
        job = {'lock': Lock(), 'completed': 0, 'total': None, 'result': None, 'error': None}
        st.session_state['contacts_sync_state'] = job
        st.session_state['contacts_sync_future'] = _contacts_executor().submit(
            _run_contacts_sync, config, database, ingestion, job)
        st.rerun()
