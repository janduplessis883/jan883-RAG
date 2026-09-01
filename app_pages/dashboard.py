import streamlit as st

from app_pages.common import render_runtime_sidebar, runtime, source_link


render_runtime_sidebar()
_, database, _, _, _, _ = runtime()

st.title("Dashboard")
st.caption("A quick view of your local knowledge base.")

stats = database.get_stats()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sources", stats["source_count"])
col2.metric("Chunks", stats["chunk_count"])
col3.metric("Embeddings", stats["embedding_count"])
if "fts_count" in stats:
    col4.metric("FTS5 entries", stats["fts_count"])

st.subheader("Recent sources")
for row in database.list_sources(limit=10):
    with st.container(border=True):
        st.write(f"**{row['title']}**")
        st.caption(f"{row['source_type']} | {row['created_at']}")
        if row["canonical_uri"]:
            st.markdown(f"[Open source]({source_link(row['canonical_uri'])})")
        st.write(row["summary"] or "No summary available.")
