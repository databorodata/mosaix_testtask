"""Пути, ключи и адреса сервисов. Ключи читаются из .env в корне репозитория и нигде не печатаются."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
BASE_XLSX = DATA / "base.xlsx"
KSR_XLSX = DATA / "ksr" / "KSR_v43_2026-05-21.xlsx"

SEARCH_URL = "https://searchapi.api.cloud.yandex.net/v2/web/search"
LLM_URL = "https://ai.api.cloud.yandex.net/v1/chat/completions"


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in os.environ.items():
        if key.startswith("YC_"):
            env[key] = value
    return env


_ENV = _load_env(ROOT / ".env")
YC_API_KEY = _ENV.get("YC_API_KEY", "")
YC_FOLDER_ID = _ENV.get("YC_FOLDER_ID", "")


def require_keys() -> None:
    if not YC_API_KEY or not YC_FOLDER_ID:
        raise SystemExit("Нет YC_API_KEY или YC_FOLDER_ID: положите их в .env в корне репозитория")
