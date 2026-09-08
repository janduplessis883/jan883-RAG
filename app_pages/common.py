from __future__ import annotations

import re
import streamlit as st
from local_rag.embeddings import OllamaClient

PREFILLED_MARKDOWN_DIRECTORIES = [
    "/Users/janduplessis/Documents/notion_partner_meeting_md_files",
    "/Users/janduplessis/Documents/notion_staff_meeting_md_files",
]


def runtime():
    return st.session_state["runtime"]


def source_link(canonical_uri: str) -> str:
    match = re.fullmatch(r"notion://page/([0-9a-fA-F-]+)", canonical_uri)
    return f"https://app.notion.com/p/{match.group(1).replace('-', '')}?source=copy_link" if match else canonical_uri


def render_runtime_details():
    config, _, _, _, _, _ = runtime()
    merged = config.load_merged()
    st.caption(f"Database: {merged['app']['database_path']}")
    st.caption(f"Embedding model: {merged['ollama']['embedding_model']}")
    st.caption(f"Default answer model: {merged['ollama']['default_answer_model']}")
    if st.button("Refresh app state", icon=":material/refresh:"):
        st.session_state.pop("runtime", None)
        st.rerun()


def render_runtime_sidebar():
    with st.sidebar.expander("Advanced settings", icon=":material/tune:"):
        render_runtime_details()


@st.cache_data(ttl=30, max_entries=8, show_spinner=False)
def model_health(settings):
    try:
        client = OllamaClient({**settings, "request_timeout_seconds": 3})
        return {"available": True, "models": client.list_models()}
    except Exception:
        return {"available": False, "models": []}


def get_answer_models(chat, configured_models, default_model):
    health = model_health(chat.config["ollama"])
    return list(dict.fromkeys(health["models"] or [default_model, *configured_models]))


def render_source_content(item, *, markdown=False):
    st.caption(f"{item['source_type']} · Passage {item.get('chunk_index', 0) + 1}")
    uri = item.get("canonical_uri")
    if uri and source_link(uri).startswith(("https://", "http://")):
        st.link_button("Open original", source_link(uri), icon=":material/open_in_new:")
    text = item.get("text", item.get("preview", ""))
    if markdown:
        st.markdown(text)
    else:
        st.text(text)
    if item.get("tags"):
        st.caption("Tags: " + ", ".join(item["tags"]))


def render_search_results(results):
    if not results:
        st.info("No matching passages. Try a broader question or fewer filters.")
        return
    for index, item in enumerate(results, 1):
        with st.container(border=True):
            st.subheader(item["title"])
            st.caption(f"[S{index}] · {item['source_type']}")
            st.write(item.get("preview", item.get("text", "")[:800]))
            with st.expander("Read full supporting passage", icon=":material/menu_book:"):
                render_source_content(item)
            with st.expander("Retrieval diagnostics"):
                st.caption(f"Ranking score: {item.get('rrf_score', 0):.4f} · Similarity: {item.get('similarity')}")
                st.caption("Diagnostic ranking values, not confidence percentages.")


def render_chat_sources(sources):
    if not sources:
        return
    st.caption("Verify an answer: open the matching citation below. Passages are saved with this conversation.")
    with st.container(horizontal=True):
        for index, item in enumerate(sources, 1):
            with st.popover(f"[S{index}] {item['title'][:45]}"):
                st.markdown(f"**{item['title']}**")
                render_source_content(item, markdown=True)


def render_chunking_controls(key_prefix: str, config: dict) -> dict:
    merged_config = config.load_merged() if hasattr(config, "load_merged") else config
    chunking_config = merged_config["chunking"]
    strategy = st.selectbox(
        "Chunking strategy",
        options=["semantic", "fixed"],
        format_func=lambda value: "Semantic" if value == "semantic" else "Fixed size",
        key=f"{key_prefix}_chunking_strategy",
        help="Fixed-size chunks are useful for Notion pages with long transcripts or unusual markup.",
    )
    if strategy == "semantic":
        chunking = {"strategy": "semantic"}
    else:
        size_col, overlap_col = st.columns(2)
        with size_col:
            chunk_size = st.number_input(
                "Chunk size (characters)",
                min_value=100,
                max_value=10000,
                value=int(chunking_config.get("fixed_chunk_size", 1200)),
                step=100,
                key=f"{key_prefix}_fixed_chunk_size",
            )
        with overlap_col:
            overlap = st.number_input(
                "Overlap (characters)",
                min_value=0,
                max_value=max(int(chunk_size) - 1, 0),
                value=min(
                    int(chunking_config.get("fixed_chunk_overlap", 200)),
                    max(int(chunk_size) - 1, 0),
                ),
                step=50,
                key=f"{key_prefix}_fixed_chunk_overlap",
            )
        chunking = {"strategy": "fixed", "chunk_size": int(chunk_size), "overlap": int(overlap)}

    validate_chunks = st.checkbox(
        "Validate chunks with LLM",
        value=False,
        key=f"{key_prefix}_validate_chunks",
        help="Makes one chat-model call per chunk and skips obvious extraction noise. This can slow ingestion.",
    )
    chunking["validate_chunks"] = validate_chunks
    if validate_chunks:
        ollama_config = merged_config["ollama"]
        models = list(dict.fromkeys(ollama_config.get("answer_models", [])))
        default_model = ollama_config.get("default_answer_model", "")
        if default_model and default_model not in models:
            models.insert(0, default_model)
        chunking["validation_model"] = st.selectbox(
            "Chunk validation model",
            models,
            index=models.index(default_model) if default_model in models else 0,
            key=f"{key_prefix}_validation_model",
        )
    return chunking
