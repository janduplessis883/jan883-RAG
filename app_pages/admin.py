import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import re
from threading import Lock

from app_pages.common import render_runtime_sidebar, runtime
from app_pages.ingest_feedback import run_import, show_import
from watcher.notion_sync import sync_calendar_once
from app_pages.contact_sync import contacts_sync_panel


@st.cache_resource
def _calendar_executor():
    return ThreadPoolExecutor(max_workers=1)


render_runtime_sidebar()
config, database, ingestion, _, _, backup = runtime()


def _run_full_calendar_sync(config, database, ingestion, state):
    # The worker must retain this session's job state across full-page reruns.
    # It must never read Streamlit session state from its background thread.
    def progress(completed, total, label):
        with state["lock"]:
            state.update(completed=completed, total=total, label=label)
    try:
        result = sync_calendar_once(config, database, ingestion, full_sync=True, progress_callback=progress)
        with state["lock"]:
            state["result"] = result
    except Exception as exc:
        with state["lock"]:
            state["error"] = str(exc)


@st.fragment(run_every="1s")
def calendar_sync_panel():
    job = st.session_state.get("calendar_sync_state")
    state = {}
    if job is not None:
        with job["lock"]:
            state = dict(job)
    future = st.session_state.get("calendar_sync_future")
    running = future is not None and not future.done()
    if running:
        completed = state.get("completed", 0)
        total = state.get("total")
        remaining = max(total - completed, 0) if total else None
        progress_label = (
            f"{completed} of {total} processed · {remaining} events remaining — {state.get('label', '')}"
            if remaining is not None
            else state.get("label") or "Fetching calendar events…"
        )
        st.progress((completed / total) if total else 0, text=progress_label)
    elif state.get("result"):
        result = state["result"]
        if result.get("status") == "skipped":
            st.info(result["message"])
        elif result.get("errors"):
            st.progress(1.0, text="Calendar sync finished with item errors")
            st.warning(f"Calendar sync completed with {result['errors']} item errors: {result['upserted']} stored, {result['deleted']} deleted.")
            with st.expander("Calendar sync errors"):
                st.json(result.get("error_details", []))
        else:
            st.progress(1.0, text="Calendar sync complete")
            st.success(f"Calendar sync complete: {result['upserted']} stored, {result['deleted']} deleted.")
        st.session_state.pop("calendar_sync_future", None)
    elif state.get("error"):
        st.error(f"Calendar sync failed: {state['error']}")
        st.session_state.pop("calendar_sync_future", None)

    if st.button("Calendar master sync", icon=":material/calendar_month:", disabled=running):
        job = {"lock": Lock(), "completed": 0, "total": None,
               "label": "Fetching calendar events…", "result": None, "error": None}
        st.session_state["calendar_sync_state"] = job
        st.session_state["calendar_sync_future"] = _calendar_executor().submit(
            _run_full_calendar_sync, config, database, ingestion, job
        )
        st.rerun()


st.title("Admin")
st.caption("Maintenance tasks and local configuration.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Notion calendar")
    st.caption("Refreshes events from the past 30 days and all upcoming events.")
    calendar_sync_panel()
    st.subheader("Notion contacts")
    st.caption("Refreshes all contacts and their attachment links.")
    contacts_sync_panel()
    if st.button("Poll Telegram", icon=":material/sync:"):
        with st.spinner("Polling Telegram updates..."):
            run_import("telegram", "sync_telegram")
    show_import("telegram")
    if st.button("Run backup", icon=":material/backup:"):
        with st.spinner("Creating backup snapshot..."):
            try:
                result = backup.run_backup()
                st.success("Backup created. Database integrity check passed.")
                st.caption(result["destination"])
            except Exception as exc:
                st.error(f"Backup failed: {exc}")
    if st.button("Rebuild vector index", icon=":material/build:"):
        with st.spinner("Rebuilding vector index..."):
            ingestion.database.rebuild_vector_index()
        st.success("Vector index rebuilt.")
    if st.button("Rebuild FTS index", icon=":material/build:"):
        with st.spinner("Rebuilding FTS index..."):
            ingestion.database.rebuild_fts_index()
        st.success("FTS index rebuilt.")

with col2:
    st.subheader("Sync frequency")
    current_settings = config.load_merged()
    sync_minutes = st.number_input(
        "Notion sync interval (minutes)",
        min_value=1,
        max_value=1440,
        value=int(current_settings.get("notion", {}).get("sync_interval_minutes", 60)),
        step=1,
        help="The watcher reads this interval between email, calendar and contacts sync cycles.",
    )
    if st.button("Save sync frequency", icon=":material/schedule:"):
        settings_text = config.read_text(config.settings_path)
        replacement = f"sync_interval_minutes = {int(sync_minutes)}"
        if re.search(r"(?m)^sync_interval_minutes\s*=.*$", settings_text):
            settings_text = re.sub(r"(?m)^sync_interval_minutes\s*=.*$", replacement, settings_text)
        else:
            settings_text = re.sub(r"(?m)^(\[notion\]\s*)$", rf"\1{replacement}\n", settings_text)
        config.validate_toml(settings_text)
        config.write_text(config.settings_path, settings_text)
        st.session_state.pop("runtime", None)
        st.success(f"Notion sync frequency saved: every {int(sync_minutes)} minutes.")

    st.write("Settings file")
    settings_text = st.text_area("config/settings.toml", value=config.read_text(config.settings_path), height=300)
    if st.button("Save settings", icon=":material/save:"):
        config.validate_toml(settings_text)
        config.write_text(config.settings_path, settings_text)
        st.session_state.pop("runtime", None)
        st.success("Settings saved.")

    st.write("Secrets file")
    secrets_text = st.text_area("config/secrets.toml", value=config.read_text(config.secrets_path), height=220)
    if st.button("Save secrets", icon=":material/save:"):
        config.validate_toml(secrets_text)
        config.write_text(config.secrets_path, secrets_text)
        st.session_state.pop("runtime", None)
        st.success("Secrets saved.")
