"""Выбор ресурса КСР моделью из кандидатов локального поиска.

Отраслевой приём «выбор из списка»: модель видит позицию, её карточку и 10 ресурсов КСР и отвечает
по строгой схеме. Код проверяет ответ: если числа выбранного ресурса не встречаются в строке или
карточке, «тот же ресурс» понижается до «той же группы»."""

from __future__ import annotations

import hashlib
import json

from . import config
from .cards import _parse_json, _post
from .ksr import Index
from .normalize import numbers
from .prices import llm_cost
from .verify import card_text

PROMPT_VERSION = "ksr-select-v1"
CACHE_DIR = config.CACHE / "ksr_select"
SYSTEM = """Ты инженер-сметчик. Даны позиция из сметы, её карточка и список ресурсов из Классификатора строительных ресурсов (КСР). Найди в списке ресурс для этой позиции.

Правила:
1. «тот же ресурс»: тот же тип изделия и те же ключевые характеристики: размеры, сечение, диаметр, класс, марка по ГОСТ. Производитель и артикул не важны, в КСР их нет.
2. «та же группа»: тот же тип изделия, но ресурса с теми же характеристиками в списке нет; укажи ближайший.
3. «нет»: в списке нет ресурса того же типа изделия. Не выбирай ресурс другого типа, даже если совпадают слова или числа.
4. choice: номер ресурса из списка, 0 если «нет».
5. reason: одно короткое предложение, почему."""
SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer"},
        "relation": {"type": "string", "enum": ["тот же ресурс", "та же группа", "нет"]},
        "reason": {"type": "string"},
    },
    "required": ["choice", "relation", "reason"],
    "additionalProperties": False,
}


def _empty(status: str, reason: str, cost: float = 0.0) -> dict:
    return {"status": status, "code": "", "name": "", "unit": "", "group": "", "path": "", "reason": reason, "cost_rub": cost}


def select(index: Index, name: str, card: dict, ntype: str, model: str = "qwen3.6-35b-a3b", k: int = 10) -> dict:
    if ntype == "service":
        return _empty("service", "услуги ищутся в сборниках расценок, не в КСР")
    hits = index.search(" ".join([card.get("canonical_name", ""), name]), k=k)
    if not hits:
        return _empty("none", "локальный поиск не нашёл кандидатов")
    cands = [index.pos[i] for i, _ in hits]
    lines = [f"Позиция: {name}", f"Карточка: {card.get('canonical_name', '')}. {card.get('description', '')}", "Ресурсы КСР:"]
    for n, p in enumerate(cands, 1):
        lines.append(f"[{n}] {p['code']} | {p['name']} | {p['unit']} | {p['group'] or p['section'] or ''}")
    user = "\n".join(lines)
    key = hashlib.sha1(f"{PROMPT_VERSION}\n{model}\n{user}".encode("utf-8")).hexdigest()[:16]
    path = CACHE_DIR / model / f"{key}.json"
    cost = 0.0
    if path.exists():
        ans = json.loads(path.read_text(encoding="utf-8"))
    else:
        body = {
            "model": f"gpt://{config.YC_FOLDER_ID}/{model}/latest",
            "temperature": 0.0,
            "max_tokens": 300,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "ksr_choice", "strict": True, "schema": SCHEMA}},
        }
        if model.startswith("qwen"):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            data = _post(body)
            usage = data.get("usage") or {}
            ans = _parse_json((data["choices"][0]["message"].get("content") or "").strip())
            cost = llm_cost(model, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
        except Exception as e:  # строка не теряется: статус «ошибка» с причиной
            return _empty("error", str(e)[:200])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ans, ensure_ascii=False), encoding="utf-8")
    choice, rel, reason = int(ans.get("choice", 0)), ans.get("relation", "нет"), ans.get("reason", "")
    if rel == "нет" or not 1 <= choice <= len(cands):
        return _empty("none", reason, round(cost, 4))
    p = cands[choice - 1]
    status = "code" if rel == "тот же ресурс" else "group"
    if status == "code":
        pn = numbers(p["name"])
        if pn and not pn <= (numbers(name) | numbers(card_text(card))):
            status = "group"
            reason += " [код понижен до группы: числа ресурса не совпали со строкой]"
    return {
        "status": status, "code": p["code"] if status == "code" else "", "name": p["name"],
        "unit": p["unit"] if status == "code" else "", "group": p["group"] or p["section"] or "",
        "path": " / ".join(x for x in (p["book"], p["section"], p["group"]) if x), "reason": reason,
        "cost_rub": round(cost, 4),
    }
