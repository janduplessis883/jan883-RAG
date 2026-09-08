"""Run UI ingestion actions with retained results and repeatable retries."""
import streamlit as st
from app_pages.common import runtime
from local_rag.presentation import ingestion_counts


def run_import(key, method, **kwargs):
    _, database, ingestion, _, _, _ = runtime()
    # Store arguments, not a closure over a previous Streamlit script run.
    arguments = {k: v for k, v in kwargs.items() if k != "progress_callback"}
    st.session_state[f"import_request_{key}"] = (method, arguments)
    try:
        result = getattr(ingestion, method)(**kwargs)
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    database.record_operation(f"import:{key}", result)
    st.session_state[f"import_result_{key}"] = result
    return result


def show_import(key):
    result = st.session_state.get(f"import_result_{key}")
    if result is None:
        return
    counts = ingestion_counts(result)
    if result.get("status") == "disabled":
        st.info(result.get("message", "This integration is disabled."))
    else:
        message = f"Added {counts['added']} documents; skipped {counts['duplicates']} duplicates"
        if counts["skipped"]:
            message += f" and {counts['skipped']} previously imported files"
        message += f"; {counts['failed']} failed."
        (st.warning if counts["failed"] else st.success)(message)
    with st.expander("Import details"):
        items = result.get("items", [result])
        st.dataframe([{"Document": item.get("title") or item.get("filename") or item.get("page_id", "Import"),
                       "Status": item.get("status", "ok"),
                       "Tags": ", ".join(item.get("tags", [])),
                       "Details": item.get("error") or item.get("message") or f"{item.get('chunk_count', 0)} passages"}
                      for item in items], hide_index=True)
    if counts["failed"] and st.button("Retry failed import", key=f"retry_{key}", icon=":material/refresh:"):
        method, arguments = st.session_state[f"import_request_{key}"]
        with st.spinner("Retrying import; already imported documents will be skipped..."):
            run_import(key, method, **arguments)
        st.rerun()
