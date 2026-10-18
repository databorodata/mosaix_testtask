"""Карточка позиции по наименованию и отфильтрованным пассажам.

Модель в Yandex AI Studio, ответ по строгой JSON-схеме, логирование данных у провайдера выключено.
Результат кэшируется по версии промпта, модели и входу."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request

from . import config
from .prices import llm_cost

PROMPT_VERSION = "card-v2"
CACHE_DIR = config.CACHE / "cards"

SYSTEM = """Ты инженер-сметчик в энергостроительстве. По наименованию позиции из сметы и фрагментам из поисковой выдачи составь карточку позиции для поиска аналогов.

Правила:
1. Используй только факты из наименования и из фрагментов. Ничего не добавляй из общих знаний: ни напряжений, ни материалов, ни производителей, которых нет во входе. Не пиши чисел, которых нет во входе: ни количества, ни напряжения, ни размеров.
2. Числа переписывай точно, с единицами.
3. Фрагменты могут описывать соседнюю модель, другой размер или другой товар. Такие фрагменты не используй.
4. canonical_name: каноническое наименование: тип изделия, марка или модель, ключевые размеры. Все обозначения из наименования переноси дословно, даже длинные: марки, коды, номера, типы вроде «Т-1(2)», «K5E-K6E», полное обозначение модели.
5. description: от 60 до 100 слов, не короче 60: что это за изделие, для чего применяется, ключевые характеристики с числами и единицами. Каждое дробное число пиши в двух видах, например «1,5 мм² (1.5 мм2)».
6. synonyms: 5–10 вариантов того, как эту же позицию пишут снабженцы, сметчики и магазины: сокращения, другой порядок слов, марка латиницей и кириллицей, слитно и через пробел. Только варианты этой же позиции, не аналоги.
7. attributes: характеристики «название, значение, единица» только из входа; если единицы нет, пустая строка.
8. manufacturer и model_code: только если они есть во входе, иначе пустая строка.
9. kind: «товар», «услуга» или «комплект».
10. confidence: «высокая», если фрагменты прямо про эту позицию; «средняя», если фрагменты про категорию; «низкая», если опирался только на наименование.
11. sources_used: номера фрагментов, которые использовал."""

SCHEMA = {
    "type": "object",
    "properties": {
        "canonical_name": {"type": "string"},
        "kind": {"type": "string", "enum": ["товар", "услуга", "комплект"]},
        "group": {"type": "string"},
        "description": {"type": "string"},
        "synonyms": {"type": "array", "items": {"type": "string"}},
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["name", "value", "unit"],
                "additionalProperties": False,
            },
        },
        "manufacturer": {"type": "string"},
        "model_code": {"type": "string"},
        "confidence": {"type": "string", "enum": ["высокая", "средняя", "низкая"]},
        "sources_used": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["canonical_name", "kind", "group", "description", "synonyms", "attributes",
                 "manufacturer", "model_code", "confidence", "sources_used"],
    "additionalProperties": False,
}


def user_text(name: str, ntype_title: str, passages: list[dict]) -> str:
    lines = [f"Наименование: {name}", f"Тип наименования: {ntype_title}", "Фрагменты из выдачи:"]
    if not passages:
        lines.append("Фрагментов нет.")
    for k, p in enumerate(passages, 1):
        lines.append(f"[{k}] {p['domain']} | {p['title']} | {p['text']}")
    return "\n".join(lines)


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean_strings(obj):
    """Убирает управляющие символы: модель превращает «500\\800» из наименования в символ забоя."""
    if isinstance(obj, str):
        return _CONTROL.sub("", obj)
    if isinstance(obj, list):
        return [_clean_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean_strings(v) for k, v in obj.items()}
    return obj


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise


def _post(body: dict) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Api-Key {config.YC_API_KEY}",
        "OpenAI-Project": config.YC_FOLDER_ID,
        "x-data-logging-enabled": "false",
        "Content-Type": "application/json",
    }
    last = "нет ответа"
    for attempt in range(4):
        req = urllib.request.Request(config.LLM_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "ignore")[:400]
            last = f"HTTP {e.code}: {text}"
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(last) from e
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(last)


def make_card(name: str, ntype_title: str, passages: list[dict], model: str) -> dict:
    user = user_text(name, ntype_title, passages)
    key = hashlib.sha1(f"{PROMPT_VERSION}\n{model}\n{user}".encode("utf-8")).hexdigest()[:16]
    path = CACHE_DIR / model / f"{key}.json"
    if path.exists():
        res = json.loads(path.read_text(encoding="utf-8"))
        res["cached"] = True
        return res
    body = {
        "model": f"gpt://{config.YC_FOLDER_ID}/{model}/latest",
        "temperature": 0.1,
        "max_tokens": 1500,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "card", "strict": True, "schema": SCHEMA}},
    }
    if model.startswith("qwen"):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.monotonic()
    card, error, usage = None, None, {}
    try:
        data = _post(body)
        usage = data.get("usage") or {}
        content = (data["choices"][0]["message"].get("content") or "").strip()
        card = _clean_strings(_parse_json(content))
    except Exception as e:  # ошибка попадает в журнал прогона, строка не теряется
        error = str(e)[:400]
    pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    res = {
        "card": card, "error": error, "model": model, "prompt_version": PROMPT_VERSION,
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
        "cost_rub": round(llm_cost(model, pt, ct), 4),
        "latency_ms": int((time.monotonic() - started) * 1000), "cached": False,
    }
    if card is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res
