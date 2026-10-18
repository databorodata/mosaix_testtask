"""Альтернативы проверкам кодом на тех же наборах: поиск только по КСР и векторам; реранкер вместо проверок.

Запуск: .venv/bin/python -m scripts.eval_alt
Замороженный поиск не меняется: здесь другие способы собрать выдачу из тех же индексов.
Порог «не найдено» для реранкера подбирается только на наборе настройки eval/dev_set.json.
"""

from __future__ import annotations

import collections
import json

import numpy as np

from app import config
from app.attrs import kind_match, parse
from app.ksr import tokens as norm_tokens
from app.normalize import clean
from app.searcher import FIELDS, POOL, RRF_K, Engine

RERANKER = "BAAI/bge-reranker-v2-m3"


def rrf(lists: list[list[int]]) -> list[int]:
    fused: dict[int, float] = collections.defaultdict(float)
    for lst in lists:
        for rank, i in enumerate(lst):
            fused[i] += 1.0 / (RRF_K + rank + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])]


def lists_for(eng: Engine, q: str) -> dict[str, list[int]]:
    qv = eng.qvec(q)
    cos_name = eng.cos(qv, ["name"])
    cos_card = eng.cos(qv, ["canonical", "synonyms", "description"])
    cos_all = eng.cos(qv, FIELDS)
    return {
        "ksr": [i for i, _ in eng.ksr_channel(q)],
        "dense_name": [int(i) for i in np.argsort(-cos_name)[:100]],
        "dense_card": [int(i) for i in np.argsort(-cos_card)[:100]],
        "dense_all": [int(i) for i in np.argsort(-cos_all)[:100]],
        "bm_names": [i for i, _ in eng.bm_norm.search(norm_tokens(clean(q)), 100)],
        "bm_cards": [i for i, _ in eng.bm_card.search(norm_tokens(q), 100)],
        "cos_all": cos_all,
    }


def checked(eng: Engine, q: str, pool: list[int], cos: np.ndarray) -> tuple[bool, list[int]]:
    """Те же проверки, что в замороженном поиске: вид, правила, характеристики, коды."""
    a = parse(q)
    top = None
    judged, clean_hits = [], 0
    for rank, i in enumerate(pool[:POOL]):
        base = 1.0 / (rank + 1) ** 0.25  # порядок пула сохраняется, как доля от лучшего
        bonus, status, _ = eng.judge(a, eng.attrs_plain[i], eng.attrs_plain[i].heads + eng.extra_nouns[i])
        if status == "не подходит" or cos[i] < eng.tau:
            continue
        judged.append((base + bonus, i))
        clean_hits += status in ("характеристики совпали", "похожая")
    judged.sort(key=lambda x: -x[0])
    return bool(clean_hits), [i for _, i in judged[:5]] if clean_hits else []


def hard_filter(eng: Engine, q: str, order: list[int]) -> list[int]:
    """Только жёсткие условия: вид изделия и правила специалиста из файла. Без весов и сравнения кодов."""
    a = parse(q)
    rules = {r["attribute"]: r for r in eng._rules_for(a.kind)}
    out = []
    for i in order:
        c = eng.attrs_plain[i]
        if not kind_match(a.kind, c.heads + eng.extra_nouns[i]):
            continue
        ok = True
        for key, r in rules.items():
            qv, cv = getattr(a, key), getattr(c, key)
            if not qv:
                continue
            if r.get("mode") == "strict" and not (qv & cv):
                ok = False
            if r.get("mode") == "steps":
                if not cv or min(eng._step_distance(x, y, r["steps"]) for x in qv for y in cv) > r.get("max_steps", 0):
                    ok = False
        if ok:
            out.append(i)
    return out


def metrics(queries: list[dict], outputs: dict[str, list[int]], eng: Engine) -> str:
    n_a = sum(q["type"] == "answer" for q in queries)
    n_n = len(queries) - n_a
    h1 = h5 = w1 = nf = nf_false = 0
    for q in queries:
        top = [eng.rows[i]["row"] for i in outputs[q["id"]]][:5]
        gold = set(q.get("same", [])) | set(q.get("substitute", []))
        if top and top[0] in set(q.get("wrong_key", [])):
            w1 += 1
        if q["type"] == "answer":
            h1 += bool(top and top[0] in gold)
            h5 += bool(gold & set(top))
            nf_false += not top
        else:
            nf += not top
    return f"{h1} из {n_a} | {h5} из {n_a} | {w1} из {len(queries)} | {nf} из {n_n} | {nf_false} из {n_a}"


def main() -> None:
    from sentence_transformers import CrossEncoder
    eng = Engine(with_ksr=True)
    ce = CrossEncoder(RERANKER, max_length=256)
    sets = {name: json.loads((config.ROOT / f"eval/{name}.json").read_text(encoding="utf-8"))["queries"]
            for name in ("dev_set", "control_set")}
    out: dict[str, dict[str, dict[str, list[int]]]] = {s: collections.defaultdict(dict) for s in sets}
    top_scores: dict[str, dict[str, float]] = {s: {} for s in sets}
    rr_orders: dict[str, dict[str, list[tuple[int, float]]]] = {s: {} for s in sets}
    for s, queries in sets.items():
        for q in queries:
            L = lists_for(eng, q["query"])
            pool_kv = rrf([L["ksr"], L["dense_card"], L["dense_name"]])
            out[s]["КСР + векторы карточки + векторы наименования"][q["id"]] = pool_kv[:5]
            found, hits = checked(eng, q["query"], pool_kv, L["cos_all"])
            out[s]["то же + наши проверки"][q["id"]] = hits
            pool = rrf([L["bm_names"], L["bm_cards"], L["dense_all"]])[:30]
            scores = ce.predict([(q["query"], eng.rows[i]["clean"]) for i in pool], show_progress_bar=False)
            order = [i for _, i in sorted(zip(scores, pool), key=lambda x: -x[0])]
            rr_orders[s][q["id"]] = sorted(zip(pool, map(float, scores)), key=lambda x: -x[1])
            top_scores[s][q["id"]] = float(max(scores))
            out[s]["Реранкер вместо проверок"][q["id"]] = order[:5]
            out[s]["Реранкер + только вид изделия и правила из файла"][q["id"]] = hard_filter(eng, q["query"], order)[:5]
    # порог «не найдено» для реранкера только по набору настройки
    dev = sets["dev_set"]
    cands = sorted(set(top_scores["dev_set"].values()))
    best_t, best_acc = None, -1
    for t in cands + [cands[-1] + 1]:
        acc = sum((top_scores["dev_set"][q["id"]] >= t) == (q["type"] == "answer") for q in dev)
        if acc > best_acc:
            best_t, best_acc = t, acc
    print(f"порог реранкера по набору настройки: {best_t:.3f}, верных решений «есть/нет ответа» {best_acc} из {len(dev)}")
    for s in sets:
        for q in sets[s]:
            sc = dict(rr_orders[s][q["id"]])
            for v in ("Реранкер вместо проверок", "Реранкер + только вид изделия и правила из файла"):
                lst = out[s][v][q["id"]]
                out[s][v + ", с порогом «не найдено»"][q["id"]] = lst if lst and sc[lst[0]] >= best_t else []
    for s, queries in sets.items():
        print(f"\n{s}: вариант | верный первым | в пятёрке | первой другая характеристика | «не найдено» без ответа | ложное «не найдено»")
        for v, res in out[s].items():
            print(f"  {v} | {metrics(queries, res, eng)}")
    json.dump({s: {v: {k: [eng.rows[i]['row'] for i in x] for k, x in r.items()} for v, r in out[s].items()} for s in out},
              open(config.ROOT / "eval/control/alt_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
