"""Привязка строки к КСР: локальный поиск по наименованиям классификатора.

Если совпали все числа строки и её буквенная марка, строка получает код КСР и единицу.
Иначе получает только группу ближайшего ресурса или статус «нет в справочниках»."""

from __future__ import annotations

import collections
import json
import math
import re

import openpyxl

from . import config
from .normalize import codes, fold, numbers
from .verify import card_text

CACHE_FILE = config.CACHE / "ksr_v43_positions.json"
_HDR = re.compile(r"^(Книга|Часть|Раздел|Группа)\s+([0-9.]+):\s*(.+)$")
_OKPD = re.compile(r"^(\d{2}\.\d{2}\.\d{2}\.\d{3})\.(.+?)(?:-000)?$")
_LEVELS = ["book", "part", "section", "group"]
_LEVEL_OF = {"Книга": "book", "Часть": "part", "Раздел": "section", "Группа": "group"}
_ENDINGS = re.compile(r"(ами|ями|ого|его|ому|ему|ыми|ими|ах|ях|ов|ев|ам|ям|ой|ей|ый|ий|ая|яя|ое|ее|ые|ие|ую|юю|ом|ем|у|ю|а|я|ы|и|е|о|ь)$")
SYNONYMS = {
    "ибп": "источник бесперебойного питания",
    "ups": "источник бесперебойного питания",
    "гофра": "труба гофрированная",
    "гофр": "гофрированная",
    "камера": "видеокамера",
}
_ADJ_END = re.compile(r"(ая|яя|ый|ий|ой|ое|ее|ые|ие|ую|юю|ого|его|ому|ему|ым|им|ых|их)$")


def head_noun(text: str) -> str:
    """Главное существительное грубо: первое слово длиннее трёх букв без окончания прилагательного."""
    for w in re.findall(r"[а-яё]+", (text or "").lower()):
        if len(w) > 3 and not _ADJ_END.search(w):
            return _ENDINGS.sub("", w)[:5]
    return ""


def load_positions() -> list[dict]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    wb = openpyxl.load_workbook(config.KSR_XLSX, read_only=True)
    pos: dict[str, dict] = {}
    for ws in wb.worksheets:
        cur = dict.fromkeys(_LEVELS)
        for r in ws.iter_rows(values_only=True):
            if not r or not r[0]:
                continue
            a = str(r[0]).strip()
            name = r[1] if len(r) > 1 else None
            unit = r[2] if len(r) > 2 else None
            m = _HDR.match(a)
            if m:
                lvl = _LEVEL_OF[m.group(1)]
                cur[lvl] = m.group(3).strip()
                for deeper in _LEVELS[_LEVELS.index(lvl) + 1:]:
                    cur[deeper] = None
                continue
            if "-" in a and name:
                mm = _OKPD.match(a)
                code, okpd = (mm.group(2), mm.group(1)) if mm else (a, None)
                p = pos.setdefault(code, {"code": code, "name": str(name).strip(), "unit": unit, "okpd2": None, **cur})
                if okpd:
                    p["okpd2"] = okpd
    out = list(pos.values())
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def tokens(text: str) -> list[str]:
    s = (text or "").lower().replace("ё", "е")
    for short, full in SYNONYMS.items():
        s = re.sub(rf"(?<![а-яa-z]){re.escape(short)}(?![а-яa-z])", full, s)
    s = re.sub(r"(?<=\d)[xх×*](?=\d)", "x", s)
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    out = []
    for w in re.findall(r"[a-zа-я]+|\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)*", s):
        if w.isalpha() and len(w) > 4:
            w = _ENDINGS.sub("", w)
        out.append(w)
    return out


class Index:
    def __init__(self, positions: list[dict]):
        self.pos = positions
        docs = [tokens(p["name"]) for p in positions]
        self.dl = [len(d) for d in docs]
        self.avgdl = sum(self.dl) / len(docs)
        self.inv: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for i, d in enumerate(docs):
            for w, c in collections.Counter(d).items():
                self.inv[w].append((i, c))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - len(lst) + 0.5) / (len(lst) + 0.5)) for w, lst in self.inv.items()}

    @classmethod
    def load(cls) -> "Index":
        return cls(load_positions())

    def search(self, text: str, k: int = 20, k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
        scores: dict[int, float] = collections.defaultdict(float)
        for w in set(tokens(text)):
            for i, c in self.inv.get(w, ()):
                scores[i] += self.idf[w] * c * (k1 + 1) / (c + k1 * (1 - b + b * self.dl[i] / self.avgdl))
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    def bind(self, name: str, card: dict, ntype: str = "") -> dict:
        empty = {"code": "", "name": "", "unit": "", "group": "", "path": "", "score": 0.0}
        if ntype == "service":
            return {"status": "service", **empty}  # услуги ищутся в сборниках расценок, не в КСР
        hits = self.search(" ".join([card.get("canonical_name", ""), name]))
        noun = head_noun(card.get("canonical_name", "")) or head_noun(name)
        if not hits or not noun:
            return {"status": "none", **empty}
        row_nums = numbers(name)
        allowed = row_nums | numbers(card_text(card))
        marks = {fold(c) for c in codes(name) if not any(ch.isdigit() for ch in c)}
        same_noun = [(i, s) for i, s in hits if head_noun(self.pos[i]["name"]) == noun]
        chosen = None
        for i, s in same_noun:
            p = self.pos[i]
            pn = numbers(p["name"])
            nums_ok = bool(pn) and pn <= allowed and bool(pn & row_nums)
            marks_ok = all(m in fold(p["name"]) for m in marks) if marks else True
            if nums_ok and marks_ok:
                chosen = (i, s, "code")
                break
        if chosen is None:
            if not same_noun:
                return {"status": "none", **empty}
            chosen = (*same_noun[0], "group")
        i, s, status = chosen
        p = self.pos[i]
        return {
            "status": status,
            "code": p["code"] if status == "code" else "",
            "name": p["name"],
            "unit": p["unit"] if status == "code" else "",
            "group": p["group"] or p["section"] or "",
            "path": " / ".join(x for x in (p["book"], p["section"], p["group"]) if x),
            "score": round(s, 2),
        }
