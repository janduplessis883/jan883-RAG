import json
from datetime import date
import streamlit as st
from app_pages.common import render_runtime_sidebar, render_search_results, runtime
from local_rag.presentation import filter_sources

st.title("Search")
st.caption("Find passages and narrow your search to the documents that matter.")
render_runtime_sidebar()
_, database, _, retrieval, _, _ = runtime()
saved = json.loads(database.get_state("search_preferences") or "{}")
rows = database.library_sources()
all_tags = sorted({tag for row in rows for tag in row["tags"]})
all_types = sorted({row["source_type"] for row in rows})
with st.form("search"):
    query = st.text_input("Natural language query", value=saved.get("query", ""))
    left, right = st.columns(2)
    tags = left.multiselect("Tags (match all)", all_tags, default=[t for t in saved.get("tags", []) if t in all_tags])
    types = right.multiselect("Document types", all_types, default=[t for t in saved.get("types", []) if t in all_types])
    dates = st.date_input("Date added", value=tuple(date.fromisoformat(d) for d in saved.get("dates", [])))
    limit = st.slider("Results", min_value=3, max_value=20, value=saved.get("limit", 8))
    submitted = st.form_submit_button("Search", icon=":material/search:", type="primary")
if submitted:
    saved = dict(query=query, tags=tags, types=types, dates=[str(d) for d in dates], limit=limit)
    database.set_state("search_preferences", json.dumps(saved))
    if not query.strip():
        st.warning("Enter a question or search phrase.")
    else:
        scoped = filter_sources(rows, tags=tags, types=types, start=dates[0] if dates else None,
                                end=dates[1] if len(dates) == 2 else None)
        try:
            with st.spinner("Searching your documents..."):
                results = retrieval.search(query=query, limit=limit, source_ids=[r["id"] for r in scoped]) if scoped else []
            database.set_state("search_results", json.dumps({"query": query, "results": results}))
        except Exception:
            st.error("Search could not reach the model service. Check Health and submit again. Previous results are kept below.")
previous = json.loads(database.get_state("search_results") or "{}")
if previous:
    st.caption(f"Saved results for: {previous['query']}. Run Search again to refresh after document changes.")
    # Never show a removed document from an older saved result set.
    active = {row["id"] for row in rows}
    results = [item for item in previous["results"] if item["source_id"] in active]
    render_search_results(results)
    st.download_button("Export search results", json.dumps(results, indent=2), "search-results.json", "application/json")
else:
    st.info("Search results and filters will be saved locally for your next visit.")
