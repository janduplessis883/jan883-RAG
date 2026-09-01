from pathlib import Path

import streamlit as st

from local_rag.backup import BackupService
from local_rag.chat import ChatService
from local_rag.config import ConfigManager
from local_rag.database import Database
from local_rag.ingestion import IngestionService
from local_rag.retrieval import SearchService


ROOT_DIR = Path(__file__).resolve().parent


def get_runtime():
    config = ConfigManager(ROOT_DIR)
    config.ensure_defaults()
    database = Database(config)
    database.initialize()
    ingestion = IngestionService(config, database)
    retrieval = SearchService(config, database)
    chat = ChatService(config, retrieval)
    backup = BackupService(config)
    return config, database, ingestion, retrieval, chat, backup


def main() -> None:
    st.set_page_config(page_title="Personal Local RAG", page_icon=":material/menu_book:", layout="wide")
    if "runtime" not in st.session_state:
        st.session_state["runtime"] = get_runtime()

    page = st.navigation(
        [
            st.Page("app_pages/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
            st.Page("app_pages/ingest.py", title="Ingest", icon=":material/download:"),
            st.Page("app_pages/search.py", title="Search", icon=":material/search:"),
            st.Page("app_pages/chat.py", title="Chat", icon=":material/chat:"),
            st.Page("app_pages/admin.py", title="Admin", icon=":material/settings:"),
        ],
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
