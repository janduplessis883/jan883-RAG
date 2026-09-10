from pathlib import Path
import os
import signal
import subprocess
import sys

import requests
import streamlit as st
from app_pages.common import model_health, render_runtime_sidebar, runtime


API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8005


def _api_process():
    process = st.session_state.get("api_process")
    if process is not None and process.poll() is not None:
        st.session_state.pop("api_process", None)
        return None
    return process


def _api_health(port: int):
    try:
        response = requests.get(f"http://{API_HOST}:{port}/health", timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def _api_pid(port: int):
    """Find a listener on the selected port, including APIs started externally."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    for line in result.stdout.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


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

st.subheader("Local API")
api_process = _api_process()
api_port = int(st.number_input(
    "API port",
    min_value=1024,
    max_value=65535,
    value=DEFAULT_API_PORT,
    step=1,
    disabled=api_process is not None,
    help="Choose an unused local port for the API, such as 8005.",
))
api_pid = api_process.pid if api_process is not None else _api_pid(api_port)
api_health = _api_health(api_port)
api_running = api_pid is not None or api_health is not None
with st.container(border=True):
    if api_health is not None:
        st.success(f"Running at http://{API_HOST}:{api_port}")
        st.caption("The API health endpoint responded successfully.")
    elif api_process is not None:
        st.info(f"Starting at http://{API_HOST}:{api_port}…")
    else:
        st.warning("Stopped")

    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("Start API", icon=":material/play_arrow:", disabled=api_running, width="stretch"):
            root_dir = Path(__file__).resolve().parents[1]
            try:
                st.session_state["api_process"] = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "api:app",
                        "--host",
                        API_HOST,
                        "--port",
                        str(api_port),
                    ],
                    cwd=root_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                st.success("API start requested. Refresh health in a moment.")
            except OSError as exc:
                st.error(f"Could not start the API: {exc}")
    with stop_col:
        if st.button("Stop API", icon=":material/stop:", disabled=api_pid is None, width="stretch"):
            if api_process is not None:
                api_process.terminate()
                try:
                    api_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    api_process.kill()
                    api_process.wait(timeout=5)
            else:
                os.kill(api_pid, signal.SIGTERM)
            st.session_state.pop("api_process", None)
            st.success("API stopped.")

st.subheader("Last successful sync")
cols = st.columns(4)
for col, kind, label in zip(cols, ("sync:notion", "sync:notion_calendar", "sync:telegram", "sync:folder"), ("Notion email", "Notion calendar", "Telegram", "Folder watcher")):
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
