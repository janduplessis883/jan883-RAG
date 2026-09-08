import streamlit as st
from app_pages.common import model_health, render_runtime_sidebar, runtime

st.title("Health")
st.caption("Model availability, imports, sync activity, and backup verification.")
render_runtime_sidebar()
config, database, _, _, _, _ = runtime()
if st.button("Refresh health", icon=":material/refresh:"):
    model_health.clear()
settings = config.load_merged()
health = model_health(settings["ollama"])
left, right = st.columns(2)
with left.container(border=True):
    st.subheader("oMLX / Model Service")
    if health["available"]:
        st.success("Available")
        expected = {settings["ollama"]["default_answer_model"], settings["ollama"]["embedding_model"]}
        missing = expected - set(health["models"])
        if missing:
            st.warning("Configured models not listed: " + ", ".join(sorted(missing)))
    else:
        st.error("Unavailable")
        st.write("Start your model service and check its address in Admin settings.")
    st.caption("Model-list connectivity check, cached for up to 30 seconds; generation is not tested.")
with right.container(border=True):
    st.subheader("Library")
    st.metric("Documents", database.get_stats()["source_count"])
    failures = [row for row in database.list_ingestion_log(limit=10000) if row["status"] == "error"]
    st.metric("Unresolved file import failures", len(failures))

st.subheader("Last successful sync")
cols = st.columns(3)
for col, kind, label in zip(cols, ("sync:notion", "sync:telegram", "sync:folder"), ("Notion", "Telegram", "Folder watcher")):
    latest = database.last_successful_operation(kind)
    with col.container(border=True):
        st.write(f"**{label}**")
        st.write(latest["created_at"] + " UTC" if latest else "No successful run recorded")
st.caption("Sync records appear after the updated workers run. A successful run does not prove a worker is currently running.")

st.subheader("Last verified backup")
backup = database.last_successful_operation("backup")
if backup:
    st.success(f"Database integrity verified: {backup['details']['verified_at']}")
    st.caption(backup["details"]["destination"])
    st.caption(backup["details"]["verification"])
else:
    st.info("No verified backup recorded. Create one from Admin.")

st.subheader("Recent operations")
runs = database.operation_history()
if runs:
    st.dataframe([{"Time (UTC)": row["created_at"], "Operation": row["kind"], "Status": row["status"]} for row in runs], hide_index=True)
    failed_runs = [row for row in runs if row["status"] == "error"]
    if failed_runs:
        with st.expander(f"Failed operations ({len(failed_runs)} in the last 50 runs)"):
            for row in failed_runs:
                st.write(f"**{row['kind']} · {row['created_at']} UTC**")
                details = row["details"]
                st.text(details.get("error") or details.get("message") or f"{details.get('errors', 0)} items failed")
            st.caption("Retry an import on Ingest, or rerun the relevant sync. Historical failures remain in this log.")
else:
    st.caption("No operations recorded yet.")
if failures:
    with st.expander("File imports that need attention"):
        st.dataframe(failures, hide_index=True)
