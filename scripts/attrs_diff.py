"""Снимок разбора характеристик по всей базе и сравнение с ним.

Зачем: любая правка app/attrs.py или словарей rules/synonyms.yaml может незаметно поменять разбор
соседних строк. Снимок показывает, какие строки базы изменили разбор, до запуска замера.

Запуск:
  .venv/bin/python -m scripts.attrs_diff --save   # записать снимок tests/snapshots/attrs_base.jsonl
  .venv/bin/python -m scripts.attrs_diff          # сравнить текущий разбор со снимком
"""

from __future__ import annotations

import argparse
import json

from app import config
from app.attrs import parse
from app.normalize import clean

SNAP = config.ROOT / "tests" / "snapshots" / "attrs_base.jsonl"


def render(name: str) -> dict:
    a = parse(clean(name))
    return {"kind": a.kind, "heads": a.heads[:6],
            "section": sorted(map(list, a.section)), "sizes": sorted(a.sizes),
            "power_kva": sorted(a.power_kva), "power_kw": sorted(a.power_kw),
            "voltage": sorted(a.voltage), "current": sorted(a.current),
            "resolution": sorted(a.resolution), "ip": sorted(a.ip), "mass": sorted(a.mass),
            "codes": sorted(a.codes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    rows = json.loads((config.DATA / "full" / "rows_all.json").read_text(encoding="utf-8"))
    cur = {r["row"]: render(r["name"]) for r in rows}
    if args.save:
        SNAP.parent.mkdir(parents=True, exist_ok=True)
        SNAP.write_text("\n".join(json.dumps({"row": k, **v}, ensure_ascii=False) for k, v in cur.items()) + "\n",
                        encoding="utf-8")
        print(f"снимок: {len(cur)} строк → {SNAP}")
        return
    base = {d["row"]: {k: v for k, v in d.items() if k != "row"}
            for d in map(json.loads, SNAP.read_text(encoding="utf-8").splitlines())}
    names = {r["row"]: r["name"] for r in rows}
    diff = [row for row in base if base[row] != cur[row]]
    print(f"строк с изменённым разбором: {len(diff)} из {len(base)}")
    for row in diff[:40]:
        print(f"\n#{row} {names[row][:90]}")
        for k in base[row]:
            if base[row][k] != cur[row][k]:
                print(f"  {k}: {base[row][k]} → {cur[row][k]}")
    if len(diff) > 40:
        print(f"... и ещё {len(diff) - 40}")


if __name__ == "__main__":
    main()
