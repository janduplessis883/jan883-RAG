from uuid import uuid4
import streamlit as st

from app_pages.common import get_answer_models, render_chat_sources, render_runtime_details, runtime
from local_rag.presentation import export_conversation


_, database, _, _, chat, _ = runtime()
merged = chat.config
st.html("""
<div class="rag-banner" role="banner" aria-label="jan883-RAG">
    <span>jan883-RAG</span>
</div>
<style>
    .rag-banner {
        align-items: center;
        animation: rag-gradient 14s ease-in-out infinite;
        background: linear-gradient(120deg, #252f3d 0%, #263b4d 48%, #3a5367 100%);
        background-size: 200% 200%;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 16px;
        box-shadow: 0 14px 28px rgba(17, 24, 39, 0.24);
        box-sizing: border-box;
        color: #ffffff;
        display: flex;
        min-height: 116px;
        padding: 0.5rem 2.5rem;
        transition: box-shadow 160ms ease, transform 160ms ease;
        width: 100%;
    }

    @keyframes rag-gradient {
        0%, 100% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
    }

    .rag-banner span {
        font-size: clamp(2.25rem, 6vw, 4.5rem);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1;
    }

    .rag-banner:hover {
        box-shadow: 0 6px 12px rgba(17, 24, 39, 0.2);
        transform: translateY(4px);
    }

    .rag-banner:active {
        box-shadow: 0 3px 6px rgba(17, 24, 39, 0.18);
        transform: translateY(6px);
    }

    @media (prefers-reduced-motion: reduce) {
        .rag-banner {
            animation: none;
            transition: none;
        }
    }
</style>
""")
st.caption("Answers grounded in your documents, with passages you can verify.")
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("chat_session_id", str(uuid4()))
if pending := st.session_state.pop("pending_conversation", None):
    st.session_state["conversation_picker"] = pending
with st.sidebar:
    if st.button("New conversation", icon=":material/add_comment:", width="stretch"):
        st.session_state["chat_messages"] = []
        st.session_state["chat_session_id"] = str(uuid4())
        st.session_state["conversation_picker"] = None
    st.divider()
    conversations = database.list_conversations()
    titles = {row["id"]: row["title"] for row in conversations}
    options = [None, *titles]
    st.session_state.setdefault("conversation_picker", None)
    if st.session_state["conversation_picker"] not in options:
        st.session_state["conversation_picker"] = None
    selected = st.selectbox("Conversation history", options, key="conversation_picker",
                            format_func=lambda cid: "New conversation" if cid is None else titles[cid])
    if selected and selected != st.session_state["chat_session_id"]:
        st.session_state["chat_messages"] = database.load_conversation(selected)
        st.session_state["chat_session_id"] = selected
    st.divider()
    with st.expander("Advanced settings", icon=":material/tune:"):
        models = get_answer_models(chat, merged["ollama"]["answer_models"], merged["ollama"]["default_answer_model"])
        selected_model = st.selectbox("Answer model", models,
            index=models.index(merged["ollama"]["default_answer_model"]) if merged["ollama"]["default_answer_model"] in models else 0)
        source_limit = st.slider("Passages to reference", 1, 20, 8)
        multi_query = st.toggle("Multi-query retrieval", value=True,
                               help="Explore related wording before retrieving passages. Adds a model call.")
        hybrid = st.toggle("Hybrid retrieval", value=bool(merged["retrieval"].get("hybrid_enabled", True)),
                           help="Combine keyword and semantic matches.")
        chat_sources = st.multiselect(
            "Chat with",
            options=["Knowledge Base", "Calendar"],
            default=["Knowledge Base", "Calendar"],
            help="Choose which sources should be searched for this conversation.",
        )
        include_knowledge_base = "Knowledge Base" in chat_sources
        include_calendar = "Calendar" in chat_sources
        render_runtime_details()

messages = st.session_state["chat_messages"]
if messages:
    st.download_button("Export conversation", export_conversation(messages),
                       file_name=f"conversation-{st.session_state['chat_session_id'][:8]}.md", mime="text/markdown")
for message in messages:
    role = message["role"]
    avatar = ":material/auto_awesome:" if role == "assistant" else ":material/person:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        if message.get("error"):
            st.warning("This answer was interrupted. Retry below to generate it again.")
        render_chat_sources(message.get("sources", []))

example = None
if not messages:
    st.subheader("Start with a question")
    for question in (
        "What decisions were made in the meeting notes?",
        "Which documents mention outstanding action items?",
        "What do my documents say about project planning?",
    ):
        if st.button(question, icon=":material/arrow_forward:"):
            example = question
    if database.get_stats()["source_count"] == 0:
        st.info("Your library is empty. Add a document on the Ingest page to get started.")
retry = False
if messages and (messages[-1].get("error") or messages[-1]["role"] == "user"):
    retry = st.button("Retry last question", icon=":material/refresh:")
prompt = st.chat_input("Ask about your documents") or example
if retry:
    if messages[-1].get("error"):
        messages.pop()
    prompt = messages.pop()["content"]
if prompt:
    previous = [m for m in messages if not m.get("error")]
    messages.append({"role": "user", "content": prompt})
    database.save_conversation(st.session_state["chat_session_id"], messages)
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)
    sources = []
    answer = ""
    failure = False
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        answer_slot = st.empty()
        try:
            with chat.langfuse.trace("chat-response", session_id=st.session_state["chat_session_id"],
                                     input_data={"question": prompt, "model": selected_model}, tags=["streamlit", "rag-chat"]):
                related = []
                with st.sidebar:
                    with st.status("Finding supporting passages...", type="step", expanded=True) as status:
                        if multi_query:
                            try:
                                related = chat.generate_related_questions(prompt)
                            except Exception:
                                status.write("Search expansion unavailable; using your original question.")
                        sources = chat.retrieve_sources(
                            prompt,
                            related_questions=related,
                            source_limit=source_limit,
                            hybrid=hybrid,
                            include_calendar=include_calendar,
                            include_knowledge_base=include_knowledge_base,
                        )
                        status.update(
                            label=f"Found {len(sources)} supporting passages",
                            state="complete",
                            expanded=False,
                        )

                if not sources:
                    answer = "I couldn’t find supporting passages in your library. Try a more specific question or add relevant documents."
                    answer_slot.markdown(answer)
                else:
                    for delta in chat.answer_stream(question=prompt, model_name=selected_model, history=previous,
                                                    sources=sources, related_questions=related, source_limit=source_limit):
                        answer += delta
                        answer_slot.markdown(answer)
                    if not answer.strip():
                        raise ValueError("The model returned an empty response.")
        except Exception:
            failure = True
            if not answer:
                answer = "I couldn’t complete this answer. Check the model service on the Health page, then retry."
            answer_slot.markdown(answer)
        render_chat_sources(sources)
    messages.append({"role": "assistant", "content": answer, "sources": sources, "error": failure})
    database.save_conversation(st.session_state["chat_session_id"], messages)
    try:
        chat.langfuse.flush()
    except Exception:
        pass
    # Refresh the history picker and export with the newly persisted answer.
    st.session_state["pending_conversation"] = st.session_state["chat_session_id"]
    st.rerun()
