"""Что будет, если: поиск без векторов (КСР, КСР со словами) и правка правила про код.

Запуск: .venv/bin/python -m scripts.eval_whatif
Правка правила про код здесь только имитируется: замороженный поиск не меняется.
Цифры по правке НЕ слепые: ошибку нашли на этом же контрольном наборе.
"""

from __future__ import annotations

import json
import re

from app import attrs as attrs_mod
from app import config
from app.attrs import parse
from app.searcher import Engine
from scripts.eval_alt import checked, lists_for, metrics, rrf


def soft_code_judge(eng: Engine):
    orig = eng.judge

    def judge(q, c, nouns):
        bonus, status, h = orig(q, c, nouns)
        code = [d for d in h.differs if d.startswith("код из запроса")]
        if code and status != "не подходит":
            bonus += 0.5  # снимаем «жёсткое отличие», штраф −0,2 остаётся
            h.differs = [d for d in h.differs if not d.startswith("код из запроса")]
            h.notes += code
            if not h.differs:
                status = "характеристики совпали" if h.matched and not h.missing else "похожая"
        return bonus, status, h
    return judge


def full_search(eng: Engine, q: str, judge=None) -> list[int]:
    if judge:
        eng.judge = judge
    res = eng.search(q, "rules")
    return [h.i for h in res.hits] if res.found else []


def main() -> None:
    eng = Engine(with_ksr=True)
    sets = {name: json.loads((config.ROOT / f"eval/{name}.json").read_text(encoding="utf-8"))["queries"]
            for name in ("dev_set", "control_set")}
    rows_by = {x["row"]: x["i"] for x in eng.rows}
    for s, queries in sets.items():
        out = {}
        for q in queries:
            L = lists_for(eng, q["query"])
            for title, lists in (("Только КСР", [L["ksr"]]),
                                 ("КСР + слова в наименованиях", [L["ksr"], L["bm_names"]]),
                                 ("КСР + слова в наименованиях и карточках", [L["ksr"], L["bm_names"], L["bm_cards"]])):
                pool = rrf(lists)
                out.setdefault(title, {})[q["id"]] = pool[:5]
                out.setdefault(title + " + наши проверки", {})[q["id"]] = checked(eng, q["query"], pool, L["cos_all"])[1]
        print(f"\n{s}: вариант | верный первым | в пятёрке | первой другая характеристика | «не найдено» без ответа | ложное «не найдено»")
        for v, res in out.items():
            print(f"  {v} | {metrics(queries, res, eng)}")
    # имитация правки: код из запроса не найден → штраф, а не запрет; «ф» как знак диаметра
    base_judge = eng.judge
    for title, patch_f in (("Правка: код не найден = штраф, не запрет", False),
                           ("Правка: то же + «ф» как знак диаметра", True)):
        if patch_f:
            attrs_mod._DIAM = re.compile(attrs_mod._DIAM.pattern.replace("(?:d|D|Ø|ø|", "(?:d|D|Ø|ø|ф|Ф|"))
            attrs_mod._PARAM_CODE = re.compile(attrs_mod._PARAM_CODE.pattern.replace("(?:[hHsSlLbBdD]-?\\d+", "(?:[hHsSlLbBdDфФ]-?\\d+"))
            eng.attrs_plain = [parse(x["clean"]) for x in eng.rows]
        eng.judge = base_judge
        judge = soft_code_judge(eng)
        for s, queries in sets.items():
            res = {q["id"]: full_search(eng, q["query"], judge) for q in queries}
            print(f"{s} · {title} | {metrics(queries, res, eng)}")


if __name__ == "__main__":
    main()
