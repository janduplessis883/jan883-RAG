import streamlit as st

from app_pages.common import render_runtime_sidebar, runtime, source_link
from local_rag.presentation import filter_sources

st.title("Library")
st.caption("Browse your documents, read the full text, and keep your knowledge base up to date.")
render_runtime_sidebar()
config, database, ingestion, _, _, _ = runtime()
if notice := st.session_state.pop("library_notice", None):
    st.success(notice)
removed = st.toggle("Show Trash", help="Removed documents are excluded from search. Restore them here.")
rows = database.library_sources(removed=removed)
query = st.text_input("Find a document by title", key="library_query")
left, middle, right = st.columns(3)
with left:
    tags = st.multiselect("Tags (match all)", sorted({t for row in rows for t in row["tags"]}))
with middle:
    types = st.multiselect("Document types", sorted({row["source_type"] for row in rows}))
with right:
    dates = st.date_input("Date added", value=(), help="Optional inclusive start and end dates.")
start = dates[0] if dates else None
end = dates[1] if len(dates) == 2 else None
filtered = filter_sources(rows, query, tags, types, start, end)
st.caption(f"{len(filtered)} of {len(rows)} documents")
if not filtered:
    st.info("No documents here yet." if not rows else "No documents match these filters.")
    st.stop()
page_count = max(1, (len(filtered) + 19) // 20)
page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
visible = filtered[(page - 1) * 20:page * 20]
st.dataframe([{ "Title": row["title"], "Type": row["source_type"], "Tags": ", ".join(row["tags"]),
                "Added": row["created_at"]} for row in visible], hide_index=True)
selected_id = st.selectbox("Open document", [r["id"] for r in visible],
                           format_func=lambda sid: next(r["title"] for r in visible if r["id"] == sid))
source = database.get_source(selected_id)
if not source:
    st.info("This document is no longer available.")
    st.stop()
st.subheader(source["title"])
st.caption(f"Added {source['created_at']} · Updated {source['updated_at'] or 'Never'}")
uri = source.get("canonical_uri")
if uri and source_link(uri).startswith(("http://", "https://")):
    st.link_button("Open original", source_link(uri))
with st.expander("Full document text", expanded=True):
    st.markdown(source["full_text"])
st.download_button("Download text", source["full_text"], file_name=f"document-{selected_id}.txt", mime="text/plain")
if removed:
    if st.button("Restore document", icon=":material/restore_from_trash:"):
        database.set_source_removed(selected_id, False)
        st.session_state["library_notice"] = "Document restored to the library and search."
        st.rerun()
else:
    with st.expander("Edit document", icon=":material/edit:"):
        st.caption("Changes update the local copy and rebuild its search passages. The original file or website is untouched.")
        with st.form(f"edit_source_{selected_id}"):
            title = st.text_input("Title", value=source["title"])
            tag_text = st.text_input("Tags, comma-separated", value=", ".join(source["tags"]))
            text = st.text_area("Document text", value=source["full_text"], height=350)
            save = st.form_submit_button("Save and re-index", type="primary")
        if save:
            try:
                with st.spinner("Updating search passages..."):
                    ingestion.update_source(selected_id, title=title, text=text, tags=config.parse_tags(tag_text))
                st.session_state["library_notice"] = "Document updated and re-indexed."
                st.rerun()
            except Exception as exc:
                st.error(f"Could not update this document. Your saved copy is unchanged. {exc}")
    if st.button("Move to Trash", icon=":material/delete:"):
        database.set_source_removed(selected_id, True)
        st.session_state["library_notice"] = "Document moved to Trash. You can restore it at any time."
        st.rerun()
