from __future__ import annotations

import re

import streamlit as st


PREFILLED_MARKDOWN_DIRECTORIES = [
    "/Users/janduplessis/Documents/notion_partner_meeting_md_files",
    "/Users/janduplessis/Documents/notion_staff_meeting_md_files",
]


def runtime():
    return st.session_state["runtime"]


def source_link(canonical_uri: str) -> str:
    """Return a browser-friendly URL for a stored source identifier."""
    notion_match = re.fullmatch(r"notion://page/([0-9a-fA-F-]+)", canonical_uri)
    if notion_match:
        notion_id = notion_match.group(1).replace("-", "")
        return f"https://app.notion.com/p/{notion_id}?source=copy_link"
    return canonical_uri


def render_runtime_sidebar() -> None:
    config, _, _, _, _, _ = runtime()
    merged = config.load_merged()
    with st.sidebar:
        st.header("Runtime")
        st.write(f"Database: `{merged['app']['database_path']}`")
        st.write(f"Embedding model: `{merged['ollama']['embedding_model']}`")
        st.write(f"Default answer model: `{merged['ollama']['default_answer_model']}`")
        if st.button("Refresh app state", icon=":material/refresh:"):
            st.session_state.pop("runtime", None)
            st.rerun()


def render_search_results(results: list[dict]) -> None:
    if not results:
        st.info("No results found yet.")
        return

    for index, item in enumerate(results, start=1):
        with st.container(border=True):
            st.subheader(f"{index}. {item['title']}")
            score_bits = [f"RRF score: {item['rrf_score']:.4f}"]
            if item.get("similarity") is not None:
                score_bits.append(f"Similarity: {item['similarity']:.3f}")
            else:
                score_bits.append("Exact/lexical match")
            st.caption(
                f"{' | '.join(score_bits)} | Source type: {item['source_type']} | "
                f"Chunk {item['chunk_index'] + 1} | Citation: [S{index}]"
            )
            if item["canonical_uri"]:
                st.markdown(f"[Open source]({source_link(item['canonical_uri'])})")
            st.write(item["preview"])
            if item["tags"]:
                st.caption(f"Tags: {', '.join(item['tags'])}")


def render_chat_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources", expanded=False, icon=":material/source:"):
        render_search_results(sources)


def get_answer_models(chat, configured_models: list[str], default_model: str) -> list[str]:
    try:
        models = chat.ollama.list_models()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load current Ollama models; using configured list. {exc}")
        models = configured_models

    if not models and default_model:
        models = [default_model, *models]
    return list(dict.fromkeys(models))


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
