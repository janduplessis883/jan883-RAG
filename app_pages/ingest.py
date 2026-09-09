import streamlit as st
from app_pages.ingest_feedback import run_import, show_import

from app_pages.common import (
    PREFILLED_MARKDOWN_DIRECTORIES,
    render_chunking_controls,
    render_runtime_sidebar,
    runtime,
)


render_runtime_sidebar()
config, database, ingestion, _, _, _ = runtime()
st.title("Ingest")
st.caption("Add URLs, notes, documents, Markdown directories, and Notion content.")

for key in ("url_preview", "notion_page_preview", "notion_data_source_preview"):
    st.session_state.setdefault(key, None)

with st.expander("Ingest URL", expanded=False, icon=":material/link:"):
    url_chunking = render_chunking_controls("url", config)
    with st.form("ingest_url"):
        url = st.text_input("Article URL")
        preview_url = st.form_submit_button("Preview extracted text", icon=":material/preview:")
        submit_url = st.form_submit_button("Ingest URL", type="primary", icon=":material/download:")
    if preview_url and url.strip():
        with st.spinner("Loading URL..."):
            preview = ingestion.preview_url(url.strip())
            st.session_state["url_preview"] = preview
            st.session_state["url_preview_text"] = preview["text"]
    if st.session_state["url_preview"]:
        preview = st.session_state["url_preview"]
        with st.container(border=True):
            st.write("**Preview**")
            st.caption(f"Normalized URL: {preview['canonical_uri'] or 'n/a'}")
            st.write(f"**Title:** {preview['title']}")
            st.text_area("Extracted text", value=preview["text"], height=320, key="url_preview_text")
            st.caption(f"Characters extracted: {len(preview['text'])}")
    if submit_url and url.strip():
        with st.spinner("Generating tags and ingesting URL..."):
            result = run_import("url", "ingest_url",
                url=url.strip(), chunking=url_chunking
            )
        st.session_state["url_preview"] = None


    show_import("url")

with st.expander("Ingest text or Markdown", expanded=False, icon=":material/article:"):
    text_chunking = render_chunking_controls("text", config)
    st.caption("Paste plain text or Markdown. Headings, lists, links, and emphasis remain searchable and render correctly in the Library.")
    with st.form("ingest_text"):
        title = st.text_input("Title")
        text = st.text_area("Text", height=240)
        submit_text = st.form_submit_button("Ingest text", icon=":material/note_add:")
    if submit_text and text.strip():
        with st.spinner("Generating tags and ingesting text..."):
            run_import("text", "ingest_text",
                    title=title.strip() or "Untitled note",
                    text=text,
                    source_type="text",
                    chunking=text_chunking,
                )

    show_import("text")

with st.expander("Upload PDF or document", expanded=False, icon=":material/upload_file:"):
    upload_chunking = render_chunking_controls("upload", config)
    uploaded = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])
    if uploaded and st.button("Ingest uploaded file", icon=":material/upload_file:"):
        with st.spinner("Generating tags and ingesting file..."):
            run_import("upload", "ingest_file",
                    filename=uploaded.name,
                    content=uploaded.getvalue(),
                    source_type="document",
                    chunking=upload_chunking,
                )

    show_import("upload")

with st.expander("Ingest Markdown directory", expanded=False, icon=":material/folder_open:"):
    directory_chunking = render_chunking_controls("directory", config)
    selected_directory = st.selectbox("Directory", ["Custom path", *PREFILLED_MARKDOWN_DIRECTORIES])
    with st.form("ingest_markdown_directory"):
        custom_directory = st.text_input(
            "Custom directory path", disabled=selected_directory != "Custom path"
        )
        recursive_directory = st.checkbox("Include subdirectories", value=False)
        submit_directory = st.form_submit_button(
            "Ingest Markdown directory", icon=":material/folder_open:"
        )
    if submit_directory:
        directory = custom_directory.strip() if selected_directory == "Custom path" else selected_directory
        if not directory:
            st.warning("Enter a directory path or choose one from the list.")
        else:
            progress = st.progress(0.0, text="Preparing Markdown directory import...")
            status = st.empty()

            def update_directory_progress(current: int, total: int, label: str) -> None:
                ratio = 1.0 if total == 0 else min(current / total, 1.0)
                progress.progress(ratio, text=f"Processing {current}/{total}: {label}")
                status.caption(f"Current file: {label}")

            with st.spinner("Generating tags and embedding Markdown files..."):
                result = run_import("directory", "ingest_markdown_directory",
                    directory=directory,
                    chunking=directory_chunking,
                    recursive=recursive_directory,
                    progress_callback=update_directory_progress,
                )
            progress.progress(1.0, text="Markdown directory import complete.")



    show_import("directory")

with st.expander("Ingest Notion page", expanded=False, icon=":material/description:"):
    notion_page_chunking = render_chunking_controls("notion_page", config)
    with st.form("ingest_notion_page"):
        notion_page_input = st.text_input("Notion page URL or page ID")
        preview_notion_page = st.form_submit_button(
            "Preview Notion Markdown", icon=":material/preview:"
        )
        submit_notion_page = st.form_submit_button(
            "Ingest Notion page", icon=":material/download:"
        )
    if preview_notion_page and notion_page_input.strip():
        with st.spinner("Loading Notion page..."):
            preview = ingestion.preview_notion_page(
                notion_page_input.strip()
            )
            st.session_state["notion_page_preview"] = preview
            st.session_state["notion_page_preview_text"] = preview["text"]
    if st.session_state["notion_page_preview"]:
        preview = st.session_state["notion_page_preview"]
        with st.container(border=True):
            st.write("**Notion page preview**")
            st.caption(f"Page ID: {preview['page_id']}")
            st.caption(f"Canonical URI: {preview['canonical_uri']}")
            st.write(f"**Title:** {preview['title']}")
            st.text_area(
                "Extracted Markdown",
                value=preview["text"],
                height=320,
                key="notion_page_preview_text",
            )
    if submit_notion_page and notion_page_input.strip():
        with st.spinner("Generating tags and ingesting Notion page..."):
            run_import("notion_page", "ingest_notion_page",
                    page_url_or_id=notion_page_input.strip(),
                    chunking=notion_page_chunking,
                )
        st.session_state["notion_page_preview"] = None

    show_import("notion_page")

with st.expander("Ingest Notion data source", expanded=False, icon=":material/table_view:"):
    notion_data_source_chunking = render_chunking_controls("notion_data_source", config)
    with st.form("ingest_notion_data_source"):
        notion_data_source_id = st.text_input("Notion data source ID")
        notion_data_source_limit = st.number_input(
            "Preview/import limit (0 means use configured preview limit)",
            min_value=0,
            value=0,
            step=1,
        )
        preview_notion_data_source = st.form_submit_button(
            "Preview data source", icon=":material/preview:"
        )
        submit_notion_data_source = st.form_submit_button(
            "Ingest data source", icon=":material/download:"
        )
    if preview_notion_data_source and notion_data_source_id.strip():
        with st.spinner("Loading data source..."):
            st.session_state["notion_data_source_preview"] = ingestion.preview_notion_data_source(
                notion_data_source_id.strip(), limit=int(notion_data_source_limit) or None
            )
    if st.session_state["notion_data_source_preview"]:
        preview = st.session_state["notion_data_source_preview"]
        with st.container(border=True):
            st.write("**Notion data source preview**")
            st.caption(f"Data source ID: {preview['data_source_id']}")
            st.write(f"**Title:** {preview['title']}")
            st.write(f"Pages in preview: {preview['page_count']}")
            st.dataframe(preview["dataframe"])
    if submit_notion_data_source and notion_data_source_id.strip():
        progress = st.progress(0.0, text="Preparing Notion data source import...")
        status = st.empty()

        def update_progress(current: int, total: int, label: str) -> None:
            ratio = 1.0 if total == 0 else min(current / total, 1.0)
            progress.progress(ratio, text=f"Extracting {current}/{total}: {label}")
            status.caption(f"Current page: {label}")

        with st.spinner("Generating tags and ingesting Notion data source..."):
            result = run_import("notion_data_source", "ingest_notion_data_source",
                data_source_id=notion_data_source_id.strip(),
                chunking=notion_data_source_chunking,
                limit=int(notion_data_source_limit) or None,
                progress_callback=update_progress,
            )
        progress.progress(1.0, text="Notion data source import complete.")
        st.session_state["notion_data_source_preview"] = None


    show_import("notion_data_source")
