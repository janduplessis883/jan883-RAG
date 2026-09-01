from __future__ import annotations

from typing import Any

import requests


class TelegramBotClient:
    def __init__(self, token: str, timeout: int = 60) -> None:
        self.token = token
        self.timeout = timeout
        self.api_base = f"https://api.telegram.org/bot{token}"
        self.file_base = f"https://api.telegram.org/file/bot{token}"

    def fetch_updates(self, offset: int | None, limit: int) -> list[dict[str, Any]]:
        params = {"timeout": 1, "limit": limit}
        if offset is not None:
            params["offset"] = offset
        response = requests.get(f"{self.api_base}/getUpdates", params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return payload.get("result", [])

    def download_file(self, file_id: str) -> bytes:
        file_response = requests.get(
            f"{self.api_base}/getFile",
            params={"file_id": file_id},
            timeout=self.timeout,
        )
        file_response.raise_for_status()
        file_path = file_response.json()["result"]["file_path"]
        content_response = requests.get(f"{self.file_base}/{file_path}", timeout=self.timeout)
        content_response.raise_for_status()
        return content_response.content

