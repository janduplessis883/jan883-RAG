import streamlit as st

from app_pages.common import render_runtime_sidebar, runtime, source_link
from local_rag.presentation import filter_sources

PAGE_SIZE = 10

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
page_count = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
page = (
    st.slider("Page", min_value=1, max_value=page_count, value=1, step=1)
    if page_count > 1
    else 1
)
visible = filtered[(page - 1) * PAGE_SIZE:page * PAGE_SIZE]
tag_options = sorted({tag for row in rows for tag in row["tags"]})
tag_colors = st.get_option("theme.chartCategoricalColors") or "auto"
st.dataframe(
    [
        {
            "Title": row["title"],
            "Type": row["source_type"],
            "Tags": row["tags"],
            "Added": row["created_at"],
        }
        for row in visible
    ],
    column_config={
        "Tags": st.column_config.MultiselectColumn(
            "Tags",
            options=tag_options,
            color=tag_colors,
            disabled=True,
        ),
    },
    hide_index=True,
)
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
        if st.button(
            "Generate LLM tags",
            key=f"generate_tags_{selected_id}",
            icon=":material/auto_awesome:",
            help="Review the document beginning with the tag-generation model and replace its tags.",
        ):
            with st.spinner("Generating tags with gemma-4-26b-a4b-it-4bit..."):
                generated_tags = ingestion.suggest_tags(
                    title=source["title"],
                    text=source["full_text"],
                )
            if generated_tags:
                database.update_source_tags(selected_id, generated_tags)
                st.session_state["library_notice"] = (
                    "Generated tags: " + ", ".join(generated_tags)
                )
                st.rerun()
            else:
                st.warning("The LLM did not return any usable tags. The document was not changed.")
        with st.form(f"edit_source_{selected_id}"):
            title = st.text_input("Title", value=source["title"])
            tag_text = st.text_input("Tags, comma-separated", value=", ".join(source["tags"]))
            text = st.text_area("Document text", value=source["full_text"], height=350)
            save = st.form_submit_button(
                "Save and re-index",
                type="primary",
                key=f"save_source_{selected_id}",
            )
        if save:
            try:
                with st.spinner("Updating search passages..."):
                    result = ingestion.update_source(
                        selected_id,
                        title=title,
                        text=text,
                        tags=config.parse_tags(tag_text),
                    )
                st.session_state["library_notice"] = (
                    f"Document saved and re-indexed successfully ({result.get('chunk_count', 0)} passages)."
                )
                st.toast(st.session_state["library_notice"], icon=":material/check_circle:")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not update this document. Your saved copy is unchanged. {exc}")
    if st.button("Move to Trash", icon=":material/delete:"):
        database.set_source_removed(selected_id, True)
        st.session_state["library_notice"] = "Document moved to Trash. You can restore it at any time."
        st.rerun()
