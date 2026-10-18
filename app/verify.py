"""Проверка карточки кодом: модели на слово не верим.

- каждое число из наименования должно быть в карточке;
- каждое число карточки должно встречаться в наименовании или в пассажах;
- коды и марки из наименования не должны потеряться;
- длина описания в разумных пределах."""

from __future__ import annotations

from .normalize import code_key, codes, numbers


def card_text(card: dict) -> str:
    attrs = " ".join(f"{a.get('name', '')} {a.get('value', '')} {a.get('unit', '')}" for a in card.get("attributes", []))
    return " ".join([card.get("canonical_name", ""), card.get("description", ""), card.get("model_code", ""), attrs])


def verify(name: str, card: dict, passages: list[dict]) -> dict:
    text = card_text(card)
    name_nums = numbers(name)
    card_nums = numbers(text)
    src_nums = numbers(" ".join(f"{p.get('title', '')} {p['text']}" for p in passages))
    missing_nums = sorted(name_nums - card_nums)
    invented = sorted(card_nums - name_nums - src_nums)
    card_key = code_key(text + " " + " ".join(card.get("synonyms", [])))
    missing_codes = sorted(c for c in codes(name) if code_key(c) not in card_key)
    issues = []
    if invented:
        issues.append("числа не из входа: " + ", ".join(invented))
    if missing_codes:
        issues.append("потеряны коды: " + ", ".join(missing_codes))
    if missing_nums:
        issues.append("числа наименования не в карточке: " + ", ".join(missing_nums))
    words = len(card.get("description", "").split())
    if not 40 <= words <= 130:
        issues.append(f"описание {words} слов")
    status = "fail" if (invented or missing_codes) else ("warn" if issues else "ok")
    return {"status": status, "issues": issues, "desc_words": words, "invented": invented}
