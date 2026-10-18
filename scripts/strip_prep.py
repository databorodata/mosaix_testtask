"""prep_runtime.jsonl: результат подготовки без текстов веб-выдачи.

Поиску нужны наименование, цена, тип строки, карточка и привязка к КСР. Пассажи чужих сайтов
(filter.passages) в репозиторий и образ не кладём — остаётся только их число.
"""
import json

from app import config

src = config.DATA / "full" / "prep_all.jsonl"
dst = config.DATA / "full" / "prep_runtime.jsonl"
n = 0
with dst.open("w", encoding="utf-8") as out:
    for line in src.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        f = d.get("filter") or {}
        d["filter"] = {"kind": f.get("kind"), "n_passages": len(f.get("passages") or []),
                       "trade_cut": f.get("trade_cut"), "n_kept_total": f.get("n_kept_total")}
        for card in (d.get("cards") or {}).values():
            card.pop("error", None)
        n += 1
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"{n} строк → {dst}")
