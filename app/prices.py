"""Цены в одном месте.

Модели: прайс Yandex AI Studio на 09.09.2026, рубли за 1000 токенов (вход, выход).
Поиск: прайс Yandex Search API, проверен 16–17.10.2026, рубли за 1000 запросов с НДС.
"""

from __future__ import annotations

LLM_RUB_PER_1K: dict[str, tuple[float, float]] = {
    "aliceai-llm-flash": (0.10, 0.20),
    "qwen3.6-35b-a3b": (0.20, 0.30),
    "yandexgpt-5-lite": (0.20, 0.20),
    "deepseek-v4-flash": (0.30, 0.50),
    "aliceai-llm": (0.50, 1.20),
    "yandexgpt-5.1": (0.80, 0.80),
    "yandexgpt-5-pro": (1.20, 1.20),
}

SEARCH_RUB_PER_1K = {"sync_day": 488.0, "sync_night": 366.0, "deferred_day": 30.5, "deferred_night": 25.41}


def llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in LLM_RUB_PER_1K:
        raise KeyError(f"нет цены для модели {model!r}: добавьте её в LLM_RUB_PER_1K")
    inp, out = LLM_RUB_PER_1K[model]
    return (prompt_tokens * inp + completion_tokens * out) / 1000


def search_cost(n_queries: int, mode: str = "sync_day") -> float:
    return n_queries * SEARCH_RUB_PER_1K[mode] / 1000
