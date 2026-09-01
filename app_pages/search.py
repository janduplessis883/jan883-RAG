import streamlit as st

from app_pages.common import render_runtime_sidebar, render_search_results, runtime


render_runtime_sidebar()
_, _, _, retrieval, _, _ = runtime()

st.title("Search")
st.caption("Find relevant passages across your ingested sources.")

with st.form("search"):
    query = st.text_input("Natural language query")
    limit = st.slider("Results", min_value=3, max_value=20, value=8)
    submitted = st.form_submit_button("Search", icon=":material/search:")
if submitted and query.strip():
    with st.spinner("Searching..."):
        results = retrieval.search(query=query, limit=limit)
    render_search_results(results)
