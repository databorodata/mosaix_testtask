"""Характеристики из наименования: вид изделия, сечение кабеля, размеры, мощность, напряжение,
ток, разрешение камеры, класс защиты, масса, коды и марки.

Одни и те же правила для строк базы и для запросов. Модель здесь не участвует."""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache

import pymorphy3

from .normalize import clean, codes, fold, levenshtein

_MORPH = pymorphy3.MorphAnalyzer()

_DEC = re.compile(r"(?<=\d),(?=\d)")
_TIMES = re.compile(r"(?<=\d)\s*[xх×*Х]\s*(?=\d)")
CABLE = re.compile(r"кабел|провод|ВВГ|КВВГ|АВВГ|ВБШв|ПвВ|ПвП|ППГ|NYM|СИП|ПуГВ|ПВ[1-5]|КИП|КПС|ОКГТ|ОКСН|"
                   r"витая пар|UTP|FTP|МКЭШ|АПв|ВВГЭ|КГ-|КГ\b", re.I)
_GRID = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:x(\d+(?:\.\d+)?))?(?:x(\d+(?:\.\d+)?))?"
                   r"\s*(мм2|мм²|кв\.?\s*мм|мм|mm|см|cm|м|m)?(?!\s*%)")
_NA = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s+на\s+(\d+(?:\.\d+)?)(?![\d.])")
_DIAM = re.compile(r"(?:(?<![A-Za-zА-Яа-яЁё])(?:d|D|Ø|ø|Ду|ДУ|Dу|DN|Dn)\s*[=:]?\s*"
                   r"|диаметр\w*\s*[:=]?\s*(?:(?:внутренн|наружн|номинальн|условн)\w*\s+)*[:=]?\s*)"
                   r"(\d+(?:\.\d+)?)(?!\s*(?:\d|/\d|\"|”))\s*(мм|mm|см|м)?")
_THICK = re.compile(r"(?:толщин\w*(?:\s+(?:стенки|листа|слоя))?\s*[:=]?\s*|(?<![A-Za-zА-Яа-яЁё])[sS]\s*[-=]\s*)"
                    r"(\d+(?:\.\d+)?)\s*(мм|mm)?")
_HEIGHT = re.compile(r"(?:(?<![A-Za-zА-Яа-яЁё])[hH]\s*[-=:]\s*|высот\w*\s*[:=]?\s*)(\d+(?:\.\d+)?)\s*(мм|mm|м)?")
_LENGTH = re.compile(r"(?:(?<![A-Za-zА-Яа-яЁё])L\s*[-=:]\s*|длин\w*\s*[:=]?\s*)(\d+(?:\.\d+)?)\s*(мм|mm|м)?")
_WIDTH = re.compile(r"(?:ширин\w*\s*[:=]?\s*)(\d+(?:\.\d+)?)\s*(мм|mm|м)?")
_MM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(мм|mm|см|cm)(?![\wА-Яа-я²2])")
_KVA = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(кВА|kVA|КВА|МВА|MVA)(?![\wА-Яа-я])")
_KW = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(кВт|kW|КВт|Вт|W)(?![\wА-Яа-я])")
_VOLT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(кВ|kV|КВ|В|V)(?![\wА-Яа-яAАтT])")
_VDC = re.compile(r"(?<!\d)(\d+)\s*DC(?![A-Za-z])")
_AMP = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(кА|kA|А|A)(?![\wА-Яа-я])")
_MP = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:Мп|MP|Mp|мп|Мпикс|мегапикс\w*)(?![\wА-Яа-я])")
_IP = re.compile(r"\bIP\s*-?\s*(\d{2})\b", re.I)
_MASS = re.compile(r"(свыше|более|больше|от|до|менее)\s*(\d+(?:\.\d+)?)(?:\s*до\s*(\d+(?:\.\d+)?))?\s*(т|тонн\w*|кг)(?![\wА-Яа-я])", re.I)
_WORD = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?")
# Код, набранный через пробел: «ФБС 12.4.3 т», «АС 95/16», «SL 400».
_SPLIT_CODE = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])([А-ЯЁA-Z]{2,6})\s*[-.]?\s*(\d+(?:[.\-/]\d+)*)"
                         r"(?:\s*[-.]?\s*([А-ЯЁA-Zа-яёa-z])(?![A-Za-zА-Яа-яЁё0-9]))?")
# Хвост кода через пробел: «3стп-10 70-120».
_CODE_TAIL = re.compile(r"([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9.\-/]*\d)\s+(\d+(?:[-/.]\d+)+)(?![\d.,])")
_NOT_CODE = {"ПВХ", "ПНД", "ПЭ", "ПП", "ПЭТ", "ГОСТ", "ТУ", "ИБП", "ЖБИ", "DN", "IP", "ВЛ", "КВЛ", "ЛЭП", "ПС", "РУ", "ЗРУ",
             "ОРУ", "КРУ", "КРУЭ", "СШ", "ВН", "СН", "НН"}
_UNIT_AFTER = re.compile(r"\s*(?:мм|mm|см|м\b|m\b|кВ|kV|В\b|V\b|А\b|A\b|кА|Вт|кВт|кВА|кг|т\b|Мп|MP|%|шт|л\b|°)")
_PARAM_CODE = re.compile(r"^(?:[hHsSlLbBdD]-?\d+|IP\d+|IK\d+|УХЛ\d*|ХЛ\d*)$", re.I)

# Сокращения и синонимы видов живут в rules/synonyms.yaml: их правит специалист, код не трогается.
# Здесь только загрузка; симметрия синонимов достраивается автоматически.
def _load_synonyms() -> tuple[dict[str, str], dict[str, set[str]]]:
    import yaml
    path = Path(__file__).resolve().parent.parent / "rules" / "synonyms.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    abbr = {str(k).lower(): str(v) for k, v in (data.get("abbr") or {}).items()}
    syn: dict[str, set[str]] = {}
    for kind, others in (data.get("kinds") or {}).items():
        group = {str(kind)} | {str(x) for x in (others or [])}
        for w in group:
            syn.setdefault(w, set()).update(group - {w})
    return abbr, syn


ABBR, KIND_SYN = _load_synonyms()
_NOT_KIND = {"тип", "марка", "серия", "размер", "диаметр", "толщина", "сечение", "длина", "ширина", "высота",
             "масса", "мощность", "напряжение", "ток", "класс", "цвет", "исполнение", "артикул", "производитель",
             "модель", "прим", "объём", "объем", "расход", "номер", "основа", "назначение", "вид", "шт", "мм",
             "ква", "квт", "мва", "вт", "кв", "см", "кг", "мпа", "соед", "исп"}


@dataclass
class Attrs:
    kind: str = ""
    nouns: list[str] = field(default_factory=list)
    heads: list[str] = field(default_factory=list)
    section: set[tuple] = field(default_factory=set)
    sizes: set[float] = field(default_factory=set)
    power_kva: set[float] = field(default_factory=set)
    power_kw: set[float] = field(default_factory=set)
    voltage: set[float] = field(default_factory=set)
    current: set[float] = field(default_factory=set)
    resolution: set[float] = field(default_factory=set)
    ip: set[str] = field(default_factory=set)
    mass: set[str] = field(default_factory=set)
    codes: dict[str, str] = field(default_factory=dict)  # ключ кода -> как написано
    raw: dict[str, list[str]] = field(default_factory=dict)  # как написано во входе, для объяснения


def prep(text: str) -> str:
    s = clean(text)
    s = _DEC.sub(".", s)
    return _TIMES.sub("x", s)


def _mm(value: float, unit: str | None) -> float:
    u = (unit or "").lower()
    if u in ("см", "cm"):
        return round(value * 10, 3)
    if u in ("м", "m"):
        return round(value * 1000, 3)
    return round(value, 3)


def mark_key(code: str) -> str:
    """Ключ кода с поправкой на запись категории: «нг(А)-LS» и «нг-LS» одно и то же."""
    s = fold(code)
    s = re.sub(r"HГ\s*\(\s*A\s*\)", "HГ", s)
    s = re.sub(r"HГ-A(?=-)", "HГ", s)
    return re.sub(r"[\s\-_/.,()\[\]«»\"']", "", s)


@lru_cache(maxsize=50000)
def _noun(word: str) -> tuple[str, bool]:
    """Нормальная форма существительного и признак «может быть главным словом»:
    есть разбор в именительном или винительном падеже. «лотка» в «заглушка лотка» главным не бывает."""
    w = word.lower().replace("ё", "е")
    head = w.split("-")[0] if "-" in w and w != "кабель-канал" else w
    if len(head) < 3:
        return "", False
    parses = [p for p in _MORPH.parse(head)[:4] if p.tag.POS == "NOUN"]
    if not parses:
        return "", False
    return parses[0].normal_form, any(p.tag.case in ("nomn", "accs") for p in parses)


def _nouns(text: str, limit: int, heads_only: bool) -> list[str]:
    out: list[str] = []
    for w in _WORD.findall(text or "")[:limit]:
        low = w.lower()
        if low in ABBR:
            out.extend(_nouns(ABBR[low], 3, heads_only)[:1])
            continue
        if low == "кабель-канал":
            out.append("кабель-канал")
            continue
        if w.isupper() and len(w) <= 5:
            continue  # сокращение без расшифровки не считаем видом изделия
        n, is_head = _noun(w)
        if n and n not in _NOT_KIND and (is_head or not heads_only):
            out.append(n)
    return out


def nouns(text: str, limit: int = 14) -> list[str]:
    return _nouns(text, limit, heads_only=False)


def heads(text: str, limit: int = 14) -> list[str]:
    """Существительные, которые могут быть видом изделия: без зависимых слов в родительном падеже."""
    return _nouns(text, limit, heads_only=True)


def kind_match(q_kind: str, cand_nouns: list[str]) -> bool:
    if not q_kind:
        return True
    names = set(cand_nouns)
    if q_kind in names:
        return True
    if KIND_SYN.get(q_kind, set()) & names:
        return True
    return any(len(q_kind) >= 4 and len(n) >= 4 and (q_kind in n or n in q_kind) for n in names)


def parse(text: str, extra_nouns: list[str] | None = None) -> Attrs:
    s = prep(text)
    a = Attrs()
    a.nouns = nouns(s)
    a.heads = heads(s)
    if extra_nouns:
        a.heads += [n for n in extra_nouns if n not in a.heads]
    a.kind = a.nouns[0] if a.nouns else ""

    def note(key: str, raw: str) -> None:
        a.raw.setdefault(key, []).append(raw.strip())

    cable = bool(CABLE.search(s))
    for m in _GRID.finditer(s):
        if re.match(r"[\d.]*\s*%", s[m.end():]) or re.match(r"[\d.x]*%", s[m.start():m.end() + 6].split(" ")[0] + " "):
            continue  # допуски вида «±2х2,5%» это не размер
        nums = [float(x) for x in m.groups()[:4] if x]
        unit = m.group(5)
        if cable and nums[0] == int(nums[0]) and nums[0] <= 61 and len(nums) <= 3 and nums[-1] <= 1000:
            a.section.add(tuple(nums))
            note("section", m.group(0))
        else:
            for x in nums:
                a.sizes.add(_mm(x, unit if unit not in ("мм2", "мм²") else None))
            note("sizes", m.group(0))
    for m in _NA.finditer(s):
        a.sizes.update({float(m.group(1)), float(m.group(2))})
        note("sizes", m.group(0))
    for rx in (_DIAM, _THICK, _HEIGHT, _LENGTH, _WIDTH):
        for m in rx.finditer(s):
            a.sizes.add(_mm(float(m.group(1)), m.group(2)))
            note("sizes", m.group(0))
    for m in _MM.finditer(s):
        a.sizes.add(_mm(float(m.group(1)), m.group(2)))
        note("sizes", m.group(0))
    for m in _KVA.finditer(s):
        v = float(m.group(1)) * (1000 if m.group(2).upper() in ("МВА", "MVA") else 1)
        a.power_kva.add(v)
        note("power_kva", m.group(0))
    for m in _KW.finditer(s):
        v = float(m.group(1)) / (1000 if m.group(2) in ("Вт", "W") else 1)
        a.power_kw.add(round(v, 4))
        note("power_kw", m.group(0))
    for m in _VOLT.finditer(s):
        v = float(m.group(1)) * (1000 if m.group(2).lower() in ("кв", "kv") else 1)
        a.voltage.add(v)
        note("voltage", m.group(0))
    for m in _VDC.finditer(s):
        a.voltage.add(float(m.group(1)))
        note("voltage", m.group(0))
    for m in _AMP.finditer(s):
        v = float(m.group(1)) * (1000 if m.group(2).lower() in ("ка", "ka") else 1)
        a.current.add(v)
        note("current", m.group(0))
    for m in _MP.finditer(s):
        a.resolution.add(float(m.group(1)))
        note("resolution", m.group(0))
    for m in _IP.finditer(s):
        a.ip.add(m.group(1))
        note("ip", m.group(0))
    for m in _MASS.finditer(s):
        word, x, y = m.group(1).lower(), m.group(2), m.group(3)
        unit_t = m.group(4).lower().startswith("т")
        fx = float(x) * (1 if unit_t else 0.001)
        if y:
            val = f"{fx:g}-{float(y) * (1 if unit_t else 0.001):g}"
        elif word in ("свыше", "более", "больше"):
            val = f">{fx:g}"
        elif word in ("до", "менее"):
            val = f"<={fx:g}"
        else:
            val = f">={fx:g}"
        a.mass.add(val)
        note("mass", m.group(0))
    found = list(codes(s))
    for m in _SPLIT_CODE.finditer(s):
        plain = re.fullmatch(r"\d+(?:\.\d+)?", m.group(2)) is not None
        if m.group(1).upper() in _NOT_CODE or (plain and _UNIT_AFTER.match(s[m.end(2):])):
            continue  # «ПВХ 25 мм», «ИБП 12В», «ВЛ 35 кВ» это не коды
        found.append(m.group(1) + m.group(2) + (m.group(3) or ""))
        if m.group(3):
            found.append(m.group(1) + m.group(2))
    for m in _CODE_TAIL.finditer(s):
        if any(ch.isalpha() for ch in m.group(1)):
            found.append(m.group(1) + "-" + m.group(2))
    for c in found:
        if _PARAM_CODE.match(c.strip()):
            continue
        k = mark_key(c)
        if len(k) >= 4 and any(ch.isdigit() for ch in k) or (len(k) >= 4 and c in codes(s)):
            a.codes.setdefault(k, c)
    return a


TITLES = {
    "section": "сечение", "sizes": "размеры", "power_kva": "мощность, кВА", "power_kw": "мощность, кВт",
    "voltage": "напряжение", "current": "ток", "resolution": "разрешение, Мп", "ip": "класс защиты IP",
    "mass": "масса", "codes": "код или марка", "kind": "вид изделия",
}


def fmt(key: str, a: Attrs) -> str:
    if key == "codes":
        return ", ".join(a.codes.values())
    if a.raw.get(key):
        return ", ".join(dict.fromkeys(a.raw[key]))
    return ""


def code_relation(q: Attrs, c: Attrs) -> tuple[str, str]:
    """match: код запроса есть у кандидата; near: отличается на 1–2 знака; none: нет."""
    if not q.codes:
        return "", ""
    best, best_c = "none", ""
    for qk in q.codes:
        for ck, raw in c.codes.items():
            if qk == ck:
                return "match", raw
            if len(qk) >= 5 and (qk in ck or ck in qk):
                best, best_c = "near", raw
            elif best != "near" and len(qk) >= 5 and levenshtein(qk, ck, cap=2) <= 2:
                best, best_c = "near", raw
    return best, best_c
