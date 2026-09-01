import streamlit as st

from app_pages.common import render_runtime_sidebar, runtime


render_runtime_sidebar()
config, _, ingestion, _, _, backup = runtime()

st.title("Admin")
st.caption("Maintenance tasks and local configuration.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Poll Telegram", icon=":material/sync:"):
        with st.spinner("Polling Telegram updates..."):
            st.json(ingestion.sync_telegram())
    if st.button("Run backup", icon=":material/backup:"):
        with st.spinner("Creating backup snapshot..."):
            st.json(backup.run_backup())
    if st.button("Rebuild vector index", icon=":material/build:"):
        with st.spinner("Rebuilding vector index..."):
            ingestion.database.rebuild_vector_index()
        st.success("Vector index rebuilt.")
    if st.button("Rebuild FTS index", icon=":material/build:"):
        with st.spinner("Rebuilding FTS index..."):
            ingestion.database.rebuild_fts_index()
        st.success("FTS index rebuilt.")

with col2:
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
