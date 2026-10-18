"""Нормализация наименований. Один модуль для строк базы и для запросов."""

from __future__ import annotations

import re

_FOLD = str.maketrans("АВЕКМНОРСТХУЁ×", "ABEKMHOPCTXYEX")
_KEY_SEP = re.compile(r"[\s\-_/.,()\[\]«»\"']")
_PARAM_TOKEN = re.compile(r"^(?:[DdHhLlBbRr]|Ду|Dу|Д|Н|Л|ø|Ø)=?\d+(?:[.,]\d+)?(?:мм|mm|м|m|см|cm)?$")
_DASHES = re.compile(r"[‐-―−]")
_ITEM_NO = re.compile(r"^\s*\d+(?:\.\d+)*[.)]\s+")
_SPACES = re.compile(r"\s+")
_UNIT_POWER = re.compile(r"\b(мм|см|дм|км|м|mm|cm|m)[23²³]\b", re.I)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9./()\-+*_]*")
_UNIT_TOKEN = re.compile(
    r"^\d+(?:[.,]\d+)?(?:мм2?|мм²|mm2?|см|cm|м[23]?|m[23]?|км|km|кг|kg|т|t|л|l|шт|в|v|кв|kv|ква|kva|квт|kw|вт|w|"
    r"мва|mva|а|a|ка|ka|гц|hz|мпа|mpa|мп|mp|мпикс|мкм|°с|с|ч|мс|ок|ос|мк)$",
    re.I,
)
_DIMS = re.compile(r"^\d+(?:[.,]\d+)?[xх×*]\d", re.I)
_CLASS_TOKEN = re.compile(r"^(?:ip|ik|ухл|хл|cat|nema)\d*[a-zа-я]?$", re.I)
_STOP_CODES = {"ГОСТ", "ТУ", "ПВХ", "ИБП"}


def clean(name: object) -> str:
    """Пробелы, тире и номера пунктов; смысл строки не меняется."""
    s = str(name or "").replace(" ", " ")
    s = _DASHES.sub("-", s)
    s = _ITEM_NO.sub("", s)
    return _SPACES.sub(" ", s).strip()


def fold(s: str) -> str:
    """Скелет для сравнения кодов: латинские и русские двойники совпадают, регистр не важен."""
    return (s or "").upper().translate(_FOLD)


def code_key(s: str) -> str:
    """Ключ кода для сравнения: скелет без разделителей, «XNV-6081RE» и «xnv 6081re» совпадают."""
    return _KEY_SEP.sub("", fold(s))


def num_key(raw: str) -> str:
    value = float(raw.replace(",", "."))
    return ("%f" % value).rstrip("0").rstrip(".") or "0"


def numbers(text: str) -> set[str]:
    """Числа в одном виде: 1,5 и 1.5 совпадают; степень у единиц (мм2, м3) числом не считается."""
    t = _UNIT_POWER.sub(r"\1", text or "")
    return {num_key(m) for m in _NUM.findall(t)}


def codes(text: str) -> list[str]:
    """Коды изделий и марки: токены с буквами и цифрами, кроме единиц, размеров и классов защиты,
    плюс буквенные марки вроде КВВГЭнг(A)-LS."""
    out: list[str] = []
    for tok in _TOKEN.findall(text or ""):
        t = tok.strip("().,;:-+*/")
        if len(t) < 4:
            continue
        has_digit = any(c.isdigit() for c in t)
        has_alpha = any(c.isalpha() for c in t)
        if has_digit and has_alpha:
            if _UNIT_TOKEN.match(t) or _DIMS.match(t) or _CLASS_TOKEN.match(t) or _PARAM_TOKEN.match(t):
                continue
            out.append(t)
        elif has_alpha:
            upper = sum(1 for c in t if c.isupper())
            if upper >= 3 and ("нг" in t or "-" in t or "(" in t) and t.upper() not in _STOP_CODES:
                out.append(t)
    return out


def levenshtein(a: str, b: str, cap: int = 3) -> int:
    """Расстояние правки с ранним выходом: больше cap значит «далеко»."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def search_text(name: str, limit: int = 380) -> str:
    """Текст запроса в поиск: очищенное наименование; у комплектов только часть до «в составе»."""
    s = clean(name)
    low = s.lower()
    if "в составе" in low:
        s = s[: low.index("в составе")].strip(" :")
    return s[:limit]
