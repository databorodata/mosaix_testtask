"""Подготовка строк: тип наименования, поиск, фильтр пассажей, карточки моделями, проверка, привязка к КСР.

Запуск (дневная отладка):
  python3 -m scripts.run_prep --rows data/debug/rows_50.json \
      --models qwen3.6-35b-a3b,yandexgpt-5.1 --out data/debug/prep_50.jsonl
Карточка по первой модели идёт в привязку к КСР; вторая модель нужна для сравнения.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from concurrent.futures import ThreadPoolExecutor

from app import config, ksr, ksr_select, prices
from app.cards import make_card
from app.nametype import NAMES, classify
from app.normalize import search_text
from app.passages import filter_docs
from app.search import search
from app.verify import verify


def _progress(it, total: int, label: str, every: int = 100):
    started = time.time()
    for i, x in enumerate(it, 1):
        if i % every == 0 or i == total:
            print(f"{label}: {i}/{total}, {time.time() - started:.0f} с", flush=True)
        yield x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--models", default="qwen3.6-35b-a3b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--search-workers", type=int, default=3)
    args = ap.parse_args()
    config.require_keys()
    rows = json.loads((config.ROOT / args.rows).read_text(encoding="utf-8"))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    started = time.time()

    def do_search(r: dict) -> tuple[dict, str, list[dict], bool, str | None]:
        query = search_text(r["name"])
        try:
            docs, cached = search(query)
            return r, query, docs, cached, None
        except Exception as e:  # строка не теряется: идёт дальше без выдачи, с пометкой
            return r, query, [], True, str(e)[:300]

    recs: list[dict] = []
    new_queries = 0
    with ThreadPoolExecutor(args.search_workers) as ex:
        found = list(_progress(ex.map(do_search, rows), len(rows), "поиск"))
    for r, query, docs, cached, search_error in found:
        new_queries += 0 if cached else 1
        recs.append({
            "row": r["row"], "name": r["name"], "price": r.get("price"), "label": r.get("label"),
            "anchor": r.get("anchor", False), "ntype": classify(r["name"]), "query": query,
            "n_docs": len(docs), "search_error": search_error, "filter": filter_docs(r["name"], docs), "cards": {},
        })
    print(f"поиск: {len(recs)} строк, новых запросов {new_queries}, {time.time() - started:.0f} с")

    by_row = {rec["row"]: rec for rec in recs}

    def job(task: tuple[dict, str]) -> tuple[int, str, dict]:
        rec, model = task
        return rec["row"], model, make_card(rec["name"], NAMES[rec["ntype"]], rec["filter"]["passages"], model)

    tasks = [(rec, m) for rec in recs for m in models]
    with ThreadPoolExecutor(args.workers) as ex:
        for row, model, res in _progress(ex.map(job, tasks), len(tasks), "карточки"):
            rec = by_row[row]
            if res.get("card"):
                res["verify"] = verify(rec["name"], res["card"], rec["filter"]["passages"])
            else:
                res["verify"] = {"status": "error", "issues": [res.get("error") or "нет карточки"], "desc_words": 0, "invented": []}
            rec["cards"][model] = res
    print(f"карточки: {len(recs) * len(models)} вызовов, {time.time() - started:.0f} с")

    index = ksr.Index.load()
    for rec in recs:
        rec["ksr_rule"] = index.bind(rec["name"], rec["cards"][models[0]].get("card") or {}, rec["ntype"])

    def pick(rec: dict) -> tuple[int, dict]:
        return rec["row"], ksr_select.select(index, rec["name"], rec["cards"][models[0]].get("card") or {}, rec["ntype"])

    with ThreadPoolExecutor(args.workers) as ex:
        for row, res in _progress(ex.map(pick, recs), len(recs), "КСР"):
            by_row[row]["ksr"] = res

    out = config.ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n=== сводка")
    print("тип наименования:", dict(collections.Counter(r["ntype"] for r in recs)))
    print("контекст из выдачи:", dict(collections.Counter(r["filter"]["kind"] for r in recs)),
          "| отброшено документов-соседей:", sum(len(r["filter"]["neighbor_docs"]) for r in recs),
          "| ошибок поиска:", sum(1 for r in recs if r["search_error"]))
    print(f"поиск: {new_queries} новых запросов, {prices.search_cost(new_queries):.2f} ₽ по синхронному тарифу")
    for m in models:
        res = [r["cards"][m] for r in recs]
        fresh = [x for x in res if not x.get("cached")]
        print(f"{m}: проверка {dict(collections.Counter(x['verify']['status'] for x in res))}, "
              f"вход в среднем {sum(x['usage']['prompt_tokens'] for x in res) / len(res):.0f} токенов, "
              f"выход {sum(x['usage']['completion_tokens'] for x in res) / len(res):.0f}, "
              f"новые вызовы {len(fresh)} на {sum(x['cost_rub'] for x in fresh):.2f} ₽, "
              f"задержка {sum(x['latency_ms'] for x in fresh) / max(1, len(fresh)) / 1000:.1f} с")
    print("КСР, выбор моделью:", dict(collections.Counter(r["ksr"]["status"] for r in recs)),
          f"на {sum(r['ksr'].get('cost_rub', 0) for r in recs):.2f} ₽")
    print("КСР, правило:", dict(collections.Counter(r["ksr_rule"]["status"] for r in recs)))
    print(f"всего {time.time() - started:.0f} с; результат: {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
