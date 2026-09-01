import streamlit as st
from uuid import uuid4

from app_pages.common import get_answer_models, render_chat_sources, render_runtime_sidebar, runtime


_, _, _, _, chat, _ = runtime()
merged = chat.config

st.title("Chat")
st.caption("Ask questions about your local knowledge base.")

answer_models = get_answer_models(
    chat,
    configured_models=merged["ollama"]["answer_models"],
    default_model=merged["ollama"]["default_answer_model"],
)
st.session_state.setdefault("chat_messages", [])
chat_sidebar = st.sidebar.container()
with chat_sidebar:

    st.title(":material/chat: Chat")
    if st.button("New chat", icon=":material/add_comment:", width="stretch"):
        st.session_state["chat_messages"] = []
        st.rerun()
    st.space(20)
    selected_model = st.selectbox(
        "Answer model",
        answer_models,
        index=answer_models.index(merged["ollama"]["default_answer_model"])
        if merged["ollama"]["default_answer_model"] in answer_models
        else 0,
        key="chat_answer_model",
    )
    source_limit = st.slider(
        "Sources to reference",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
        key="chat_source_limit",
        help="Number of relevant knowledge-base chunks to retrieve for each question.",
    )
    multi_query_enabled = st.toggle(
        "Multi-Query RAG",
        value=True,
        key="chat_multi_query_enabled",
        help="Generate three related searches with the default model before retrieving context.",
    )
    hybrid_retrieval_enabled = st.toggle(
        "Hybrid retrieval (FTS5 + dense)",
        value=bool(merged["retrieval"].get("hybrid_enabled", True)),
        key="chat_hybrid_retrieval_enabled",
        help="Combine exact lexical matches with dense semantic search using Reciprocal Rank Fusion. Disable to use dense embeddings only.",
    )

st.sidebar.divider()
render_runtime_sidebar()
for message in st.session_state["chat_messages"]:
    role = message["role"]
    avatar = ":material/auto_awesome:" if role == "assistant" else ":material/person:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            render_chat_sources(message["sources"])

if not st.session_state["chat_messages"]:
    st.info("Ask a question about the documents in your knowledge base.")

if prompt := st.chat_input("Ask your knowledge base"):
    previous_messages = list(st.session_state["chat_messages"])
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    related_questions = []
    streamed_answer = ""
    assistant_message = st.chat_message("assistant", avatar=":material/auto_awesome:")
    answer_placeholder = assistant_message.empty()
    session_id = st.session_state.setdefault("chat_session_id", str(uuid4()))
    with chat.langfuse.trace(
        "chat-response",
        session_id=session_id,
        input_data={"question": prompt, "model": selected_model},
        tags=["streamlit", "rag-chat"],
    ):
        with chat_sidebar.status("Processing question...", expanded=True) as progress_status:
            progress_status.write("Step 1/4: Preparing search questions...")
            if multi_query_enabled:
                related_questions = chat.generate_related_questions(prompt)
                progress_status.write("Generated search variations:")
                for related_question in related_questions:
                    progress_status.write(f"- {related_question}")
            else:
                progress_status.write("Related-question expansion is disabled; using the original question.")

            progress_status.write("Step 2/4: Retrieving knowledge-base context...")
            if hybrid_retrieval_enabled:
                progress_status.write("2a. Searching dense embeddings for semantic matches...")
                progress_status.write("2b. Searching SQLite FTS5/BM25 for exact names, dates, and identifiers...")
                progress_status.write("2c. Fusing dense and lexical rankings with Reciprocal Rank Fusion (k=60)...")
            else:
                progress_status.write("Dense-only mode enabled; skipping SQLite FTS5/BM25 retrieval.")
            sources = chat.retrieve_sources(
                question=prompt,
                related_questions=related_questions,
                source_limit=source_limit,
                hybrid=hybrid_retrieval_enabled,
            )
            progress_status.write(f"Retrieved {len(sources)} source(s).")
            progress_status.write("Step 3/4: Preparing retrieved context...")
            progress_status.write("Step 4/4: Generating final answer...")
            for text_delta in chat.answer_stream(
                question=prompt,
                model_name=selected_model,
                history=previous_messages,
                source_limit=source_limit,
                related_questions=related_questions,
                sources=sources,
            ):
                streamed_answer += text_delta
                answer_placeholder.markdown(streamed_answer)
            progress_status.write("Final answer generated.")
            progress_status.update(
                label="Question answered", state="complete", expanded=False
            )

    if sources:
        with assistant_message:
            render_chat_sources(sources)

    st.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "content": streamed_answer,
            "sources": sources,
        }
    )
    chat.langfuse.flush()
    if chat.langfuse.last_trace_url:
        st.caption(f"Langfuse trace: {chat.langfuse.last_trace_url}")
