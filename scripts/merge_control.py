"""Собирает контрольный набор из частей A и B и проверяет формат, не показывая запросы.

Запуск: python3 -m scripts.merge_control [--parts a,b --prefix control_set_ --out control_set]
Большой набор: python3 -m scripts.merge_control --parts a,b,c,d,e --prefix big_ --out big_set
Печатает только числа и ошибки формата.
"""

from __future__ import annotations

import argparse
import json

from app import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a,b")
    ap.add_argument("--prefix", default="control_set_")
    ap.add_argument("--out", default="control_set")
    args = ap.parse_args()
    PARTS = args.parts.split(",")
    ev = config.ROOT / "eval"
    rows = {r["row"] for r in json.loads((config.DATA / "full" / "rows_all.json").read_text(encoding="utf-8"))}
    queries, errors = [], []
    for p in PARTS:
        d = json.loads((ev / f"{args.prefix}{p}.json").read_text(encoding="utf-8"))
        for q in d["queries"]:
            q["part"] = p
            queries.append(q)
    ids = [q["id"] for q in queries]
    if len(set(ids)) != len(ids):
        errors.append("повторяются id")
    for q in queries:
        for key in ("source", "same", "substitute", "wrong_key"):
            bad = [r for r in q.get(key, []) if r not in rows]
            if bad:
                errors.append(f"{q['id']}: в {key} нет таких строк базы ({len(bad)} шт.)")
        if set(q.get("same", [])) & set(q.get("wrong_key", [])) or set(q.get("substitute", [])) & set(q.get("wrong_key", [])):
            errors.append(f"{q['id']}: верный ответ попал и в другую характеристику")
        if q["type"] == "answer" and not q.get("same"):
            errors.append(f"{q['id']}: у запроса с ответом пустой same")
        if q["type"] == "no_answer" and (q.get("same") or q.get("substitute")):
            errors.append(f"{q['id']}: у запроса без ответа есть верные строки")
        if q["type"] not in ("answer", "no_answer"):
            errors.append(f"{q['id']}: неизвестный тип {q['type']}")
    n_a = sum(q["type"] == "answer" for q in queries)
    n_n = len(queries) - n_a
    out = {"version": 1, "created": "2026-10-17", "author": f"агенты-разметчики вслепую к поиску, части {args.parts}",
           "row_ref": "номер строки Excel, поле row в data/full/rows_all.json", "queries": queries}
    (ev / f"{args.out}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [(ev / f"{args.prefix}{p}.md").read_text(encoding="utf-8") for p in PARTS if (ev / f"{args.prefix}{p}.md").exists()]
    (ev / f"{args.out}.md").write_text("\n\n".join(md), encoding="utf-8")
    print(f"запросов {len(queries)}: с ответом {n_a}, без ответа {n_n}")
    print(f"в среднем строк: same {sum(len(q.get('same', [])) for q in queries) / max(n_a, 1):.1f}, "
          f"substitute {sum(len(q.get('substitute', [])) for q in queries) / max(n_a, 1):.1f}, "
          f"wrong_key {sum(len(q.get('wrong_key', [])) for q in queries) / len(queries):.1f}")
    print("ошибок формата нет" if not errors else "ошибки:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
