"""Какие запросы изменили выдачу после правки правил, словарей или кода поиска.

Прогоняет основной вариант main по всем наборам запросов и сравнивает с сохранённой выдачей.
Это проверка «что изменило правило» для специалиста и страховка при починке на защите.

Запуск:
  .venv/bin/python -m scripts.whatchanged --save   # записать эталонную выдачу eval/baseline_main.json
  .venv/bin/python -m scripts.whatchanged          # сравнить текущую выдачу с эталонной
"""

from __future__ import annotations

import argparse
import json

from app import config
from app.searcher import Engine

SETS = ["dev_set", "control_set", "big_set", "final_set", "final_gemini_set"]
BASE = config.ROOT / "eval" / "baseline_main.json"


def run(eng: Engine) -> dict:
    out = {}
    for s in SETS:
        qs = json.loads((config.ROOT / f"eval/{s}.json").read_text(encoding="utf-8"))["queries"]
        for q in qs:
            r = eng.search(q["query"], "main")
            out[f"{s}:{q['id']}"] = {"query": q["query"], "found": r.found,
                                     "top": [[h.row, h.status] for h in r.hits[:5]]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    cur = run(Engine())
    if args.save:
        BASE.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"эталонная выдача: {len(cur)} запросов → {BASE}")
        return
    base = json.loads(BASE.read_text(encoding="utf-8"))
    changed = [k for k in base if base[k] != cur.get(k)]
    print(f"запросов с изменённой выдачей: {len(changed)} из {len(base)}")
    for k in changed:
        b, c = base[k], cur[k]
        print(f"\n{k} «{b['query'][:70]}»")
        print(f"  было:  found={b['found']} top={b['top']}")
        print(f"  стало: found={c['found']} top={c['top']}")


if __name__ == "__main__":
    main()
