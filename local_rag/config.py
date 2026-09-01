from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib


DEFAULT_SETTINGS = """[app]
workspace_dir = "."
data_dir = "data"
database_path = "data/rag.sqlite3"
raw_files_dir = "data/raw"
backup_root = "/Volumes/BackupHD/jan883-RAG-Backup/"

[ollama]
api_key = "12345"
base_url = "http://127.0.0.1:8000/v1"
embedding_model = "embeddinggemma-300m-4bit"
answer_models = ["gemma-4-26b-a4b-it-4bit", "gpt-oss-20b-MXFP4-Q8"]
default_answer_model = "gemma-4-26b-a4b-it-4bit"
embedding_dimensions = 768
request_timeout_seconds = 120
embed_batch_size = 64

[chunking]
target_chars = 1200
min_chars = 500
max_chars = 1800
hard_split_chars = 2200
semantic_merge_threshold = 0.72
fixed_chunk_size = 1200
fixed_chunk_overlap = 200

[retrieval]
hybrid_enabled = true
top_k = 8
candidate_k = 24
lexical_k = 24
max_context_chunks = 8
min_similarity = 0.20
dedupe_by_source = true
rrf_k = 60
expand_radius = 1

[langfuse]
base_url = "http://localhost:3000"
capture_content = false
environment = "local"

[telegram]
enabled = true
poll_limit = 25
allowed_document_extensions = ["pdf", "txt", "md"]

[notion]
enabled = true
request_timeout_seconds = 30
data_source_preview_limit = 200
max_child_page_depth = 3
"""

DEFAULT_SECRETS = """[telegram]
bot_token = ""

[notion]
api_token = ""

[langfuse]
public_key = ""
secret_key = ""
"""


class ConfigManager:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.config_dir = self.root_dir / "config"
        self.settings_path = self.config_dir / "settings.toml"
        self.secrets_path = self.config_dir / "secrets.toml"

    def ensure_defaults(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings_path.exists():
            self.write_text(self.settings_path, DEFAULT_SETTINGS)
        if not self.secrets_path.exists():
            self.write_text(self.secrets_path, DEFAULT_SECRETS)

        merged = self.load_merged()
        (self.root_dir / merged["app"]["data_dir"]).mkdir(parents=True, exist_ok=True)
        (self.root_dir / merged["app"]["raw_files_dir"]).mkdir(parents=True, exist_ok=True)

    def load_merged(self) -> dict:
        settings = self.load_toml(self.settings_path)
        secrets = self.load_toml(self.secrets_path)
        merged = self._deep_merge(settings, secrets)

        # Streamlit's secrets file is the deployment-facing source of secrets.
        # Map its conventional uppercase Langfuse variables into the same config
        # shape used by the rest of the application. These values override the
        # local config/secrets.toml values when both are present.
        streamlit_secrets = self.load_toml(self.root_dir / ".streamlit" / "secrets.toml")
        langfuse = streamlit_secrets.get("langfuse", {})
        langfuse_overrides = {
            "public_key": streamlit_secrets.get("LANGFUSE_PUBLIC_KEY", langfuse.get("public_key")),
            "secret_key": streamlit_secrets.get("LANGFUSE_SECRET_KEY", langfuse.get("secret_key")),
            "base_url": streamlit_secrets.get(
                "LANGFUSE_BASE_URL",
                langfuse.get("base_url", langfuse.get("url")),
            ),
        }
        merged["langfuse"] = self._deep_merge(
            merged.get("langfuse", {}),
            {key: value for key, value in langfuse_overrides.items() if value},
        )
        return merged

    def load_toml(self, path: Path) -> dict:
        if not path.exists():
            return {}
        return tomllib.loads(path.read_text(encoding="utf-8"))

    def validate_toml(self, text: str) -> dict:
        return tomllib.loads(text)

    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def parse_tags(self, raw: str) -> list[str]:
        return [item.strip() for item in raw.split(",") if item.strip()]

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.root_dir / path).resolve()

    def _deep_merge(self, left: dict, right: dict) -> dict:
        merged = deepcopy(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
