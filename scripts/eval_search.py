"""Замер поиска на наборе запросов по заданию: лестница вариантов против обычного поиска по словам.

Запуск: .venv/bin/python -m scripts.eval_search --set eval/control_set.json --out eval/control

Считает по каждому варианту:
- hit@1 и hit@5: доля запросов с ответом, где верная строка на первом месте и в первой пятёрке;
- wrong@1: сколько раз на первом месте строка с другой важной характеристикой (поле wrong_key);
- «не найдено» на запросах без ответа: сколько раз система честно так сказала;
- ложное «не найдено» на запросах с ответом.
Пишет results.md (сводка), details.md (выдача по каждому запросу) и results.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

from app import config
from app.searcher import TAU, VARIANTS, Engine

TITLES = {
    "words_raw": "1. Поиск по словам (BM25 по сырым наименованиям)",
    "words_norm": "2. + нормализация записи",
    "cards": "3. + карточки",
    "vectors": "4. + векторы",
    "rules": "5. + характеристики, правила, «не найдено»",
    "rules_soft": "5в. как 5, несовпадение кода не отбрасывает",
    "rerank_plain": "Р1. реранкер вместо проверок",
    "rerank": "Р2. реранкер вместо ручных весов, проверки 5в",
    "rules_nocard": "5б. как 5, но без карточек",
    "main": "И. итог: обучаемое решение с реранкером, статусы и типы строк",
    "main_light": "Иб. итог без реранкера",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--model", default="e5-base")
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()
    set_path = config.ROOT / args.set
    data = json.loads(set_path.read_text(encoding="utf-8"))
    queries = data["queries"]
    variants = [v for v in args.variants.split(",") if v]
    eng = Engine(model=args.model, tau=args.tau, with_ksr=True)
    names = {x["row"]: x["name"] for x in eng.rows}

    summary, per_query = {}, {v: [] for v in variants}
    for v in variants:
        hit1 = hit5 = wrong1 = nf_ok = nf_false = 0
        rr = 0.0
        n_a = sum(q["type"] == "answer" for q in queries)
        n_n = len(queries) - n_a
        for q in queries:
            res = eng.search(q["query"], v)
            top = [h.row for h in res.hits][:5]
            gold = set(q.get("same", [])) | set(q.get("substitute", []))
            wrong = set(q.get("wrong_key", []))
            first = top[0] if top else None
            rec = {"id": q["id"], "query": q["query"], "type": q["type"], "found": res.found, "top": top,
                   "first_ok": bool(first in gold), "in5": bool(gold & set(top)), "first_wrong_key": bool(first in wrong),
                   "reason": res.reason,
                   "hits": [{"row": h.row, "name": h.name, "score": h.score, "cos": h.cos, "status": h.status,
                             "matched": h.matched, "differs": h.differs, "missing": h.missing, "notes": h.notes,
                             "channels": h.channels} for h in res.hits],
                   "rejected": [{"row": h.row, "name": h.name, "cos": h.cos, "status": h.status, "differs": h.differs,
                                 "notes": h.notes} for h in res.rejected[:3]]}
            per_query[v].append(rec)
            if first in wrong:
                wrong1 += 1
            if q["type"] == "answer":
                hit1 += rec["first_ok"]
                hit5 += rec["in5"]
                nf_false += not top
                for rank, r in enumerate(top, 1):
                    if r in gold:
                        rr += 1 / rank
                        break
            else:
                nf_ok += not top
        summary[v] = {"n_answer": n_a, "n_no_answer": n_n, "hit1": hit1, "hit5": hit5, "mrr5": round(rr / n_a, 3) if n_a else 0,
                      "wrong1": wrong1, "not_found_ok": nf_ok, "not_found_false": nf_false,
                      "hit1_ci": [round(x, 2) for x in wilson(hit1, n_a)], "hit5_ci": [round(x, 2) for x in wilson(hit5, n_a)]}

    stamp = {"set": args.set, "set_sha": sha(set_path), "rules_sha": sha(config.ROOT / "rules" / "rules.yaml"),
             "code_sha": {p.name: sha(p) for p in sorted((config.ROOT / "app").glob("*.py"))},
             "model": args.model, "tau": args.tau, "at": datetime.datetime.now().isoformat(timespec="seconds")}
    out = config.ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"stamp": stamp, "summary": summary, "per_query": per_query},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")

    n_a = summary[variants[0]]["n_answer"]
    n_n = summary[variants[0]]["n_no_answer"]
    L = [f"# Замер поиска: {args.set}", "",
         f"Запросов {len(queries)}: с ответом {n_a}, без ответа {n_n}. Модель векторов {args.model}, порог сходства {args.tau}. "
         f"Хэш набора {stamp['set_sha']}, правил {stamp['rules_sha']}, время {stamp['at']}.", "",
         "| Вариант | Верный ответ первым | Верный ответ в пятёрке | MRR@5 | Первым стоит позиция с другой важной характеристикой | «Не найдено» на запросах без ответа | Ложное «не найдено» |",
         "|---|---|---|---|---|---|---|"]
    for v in variants:
        s = summary[v]
        L.append(f"| {TITLES.get(v, v)} | {s['hit1']} из {n_a} ({s['hit1'] / n_a:.0%}; {s['hit1_ci'][0]:.0%}–{s['hit1_ci'][1]:.0%}) | "
                 f"{s['hit5']} из {n_a} ({s['hit5'] / n_a:.0%}; {s['hit5_ci'][0]:.0%}–{s['hit5_ci'][1]:.0%}) | {s['mrr5']} | "
                 f"{s['wrong1']} | {s['not_found_ok']} из {n_n} | {s['not_found_false']} из {n_a} |")
    L += ["", f"В скобках доля и 95 % доверительный интервал Уилсона: на {n_a} запросах с ответом он широкий, "
          "малые разницы между вариантами не доказаны.", ""]
    main_v = next((v for v in ("main", "rules") if v in variants), variants[-1])
    L += [f"## По запросам, вариант «{TITLES.get(main_v, main_v)}» против поиска по словам", "",
          "| id | Запрос | Тип | Поиск по словам: первая строка | Наш поиск: первая строка | Итог |", "|---|---|---|---|---|---|"]
    base = {r["id"]: r for r in per_query[variants[0]]}
    for r in per_query[main_v]:
        b = base[r["id"]]

        def cell(x: dict) -> str:
            if not x["top"]:
                return "не найдено"
            return f"{x['top'][0]} {names[x['top'][0]][:60]}"

        if r["type"] == "answer":
            verdict = "верно первым" if r["first_ok"] else ("верно в пятёрке" if r["in5"] else ("ложное «не найдено»" if not r["top"] else "мимо"))
        else:
            verdict = "честное «не найдено»" if not r["top"] else "показал лишнее"
        if r["first_wrong_key"]:
            verdict += ", первым другая характеристика"
        L.append(f"| {r['id']} | {r['query']} | {'с ответом' if r['type'] == 'answer' else 'без ответа'} | {cell(b)} | {cell(r)} | {verdict} |")
    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    D = [f"# Выдача по запросам: {args.set}", ""]
    for i, q in enumerate(queries):
        D += [f"## {q['id']}. {q['query']}", "",
              f"Тип: {q['type']}. Верные: {sorted(set(q.get('same', [])) | set(q.get('substitute', [])))}. "
              f"Другая характеристика: {q.get('wrong_key', [])}. Ключ: {q.get('key', '')}.", ""]
        for v in variants:
            r = per_query[v][i]
            D.append(f"**{TITLES.get(v, v)}**" + ("" if r["found"] else f" · не найдено: {r['reason']}"))
            for h in r["hits"]:
                mark = "✓" if h["row"] in (set(q.get("same", [])) | set(q.get("substitute", []))) else ("✗ характеристика" if h["row"] in set(q.get("wrong_key", [])) else "·")
                extra = "; ".join(filter(None, ["совпало: " + ", ".join(h["matched"]) if h["matched"] else "",
                                                "отличается: " + ", ".join(h["differs"]) if h["differs"] else "",
                                                "нет данных: " + ", ".join(h["missing"]) if h["missing"] else "",
                                                "; ".join(h["notes"])]))
                D.append(f"- {mark} {h['row']} {h['name'][:110]} · {h['status'] or ''} · {h['score']} · cos {h['cos']}" + (f" · {extra}" if extra else ""))
            for h in r["rejected"]:
                mark = "✓" if h["row"] in (set(q.get("same", [])) | set(q.get("substitute", []))) else "·"
                D.append(f"  - отклонено {mark} {h['row']} {h['name'][:90]} · cos {h['cos']} · {'; '.join(h['differs'] + h['notes'])[:200]}")
            D.append("")
    (out / "details.md").write_text("\n".join(D), encoding="utf-8")
    print("\n".join(L[:8 + len(variants)]))
    print(f"\nфайлы: {out.relative_to(config.ROOT)}/results.md, details.md, results.json")


if __name__ == "__main__":
    main()
