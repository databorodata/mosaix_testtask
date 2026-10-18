"""Обучаемое решение «подходит или нет» вместо ручных фильтров: признаки кандидата → логистическая регрессия.

Запуск: .venv/bin/python -m scripts.eval_learned
Наборы: dev_set, control_set, big_set (136 запросов). Оценка перекрёстной проверкой по запросам:
модель учится на 4/5 запросов и проверяется на оставшейся 1/5, которую не видела. Итоговый набор ChatGPT не используется.
Жёсткими остаются только правила специалиста из rules/rules.yaml.
"""

from __future__ import annotations

import collections
import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from app import config
from app.attrs import code_relation, kind_match, parse
from app.ksr import tokens as norm_tokens
from app.normalize import clean
from app.searcher import FIELDS, RRF_K, Engine

POOL = 30
SETS = ["dev_set", "control_set", "big_set"]
FEATS = ["rrf", "log_rank", "cos_all", "cos_name", "bm_name", "bm_card", "kind", "n_match", "n_diff", "n_missing",
         "code_match", "code_near", "code_none", "rerank"]


def candidates(eng: Engine, query: str):
    q = parse(query)
    bn = dict(eng.bm_norm.search(norm_tokens(clean(query)), 100))
    bc = dict(eng.bm_card.search(norm_tokens(query), 100))
    qv = eng.qvec(query)
    cos_all, cos_name = eng.cos(qv, FIELDS), eng.cos(qv, ["name"])
    lists = [sorted(bn, key=lambda i: -bn[i]), sorted(bc, key=lambda i: -bc[i]), [int(i) for i in np.argsort(-cos_all)[:100]]]
    fused = collections.defaultdict(float)
    for lst in lists:
        for r, i in enumerate(lst):
            fused[i] += 1.0 / (RRF_K + r + 1)
    pool = [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[:POOL]]
    top = fused[pool[0]]
    rr = eng.rerank_scores(query, pool)
    mbn, mbc = max(bn.values(), default=1), max(bc.values(), default=1)
    rows = []
    for rank, i in enumerate(pool, 1):
        c = eng.attrs_plain[i]
        _, status, h = eng.judge(q, c, c.heads + eng.extra_nouns[i], soft_code=True)
        rule_fail = any("совпадают строго" in d or "ступен" in d or "у кандидата не указано" in d for d in h.differs)
        kind_ok = kind_match(q.kind, c.heads + eng.extra_nouns[i])
        rel, _ = code_relation(q, c)
        n_diff = sum(1 for d in h.differs if not d.startswith("другой вид") and "строго" not in d)
        rows.append({"i": i, "rule_fail": rule_fail, "f": [fused[i] / top, math.log(rank), float(cos_all[i]), float(cos_name[i]),
                     bn.get(i, 0) / mbn, bc.get(i, 0) / mbc, float(kind_ok), len(h.matched), n_diff, len(h.missing),
                     float(rel == "match"), float(rel == "near"), float(rel == "none"), rr[i]]})
    return rows


def evaluate(queries, preds, eng, t):
    n_a = sum(q["type"] == "answer" for q in queries)
    h1 = h5 = w1 = nf = nff = 0
    for q in queries:
        cands = [(p, i) for p, i, fail in preds[q["id"]] if not fail]
        cands.sort(key=lambda x: -x[0])
        top = [eng.rows[i]["row"] for p, i in cands[:5]] if cands and cands[0][0] >= t else []
        gold = set(q.get("same", [])) | set(q.get("substitute", []))
        if top and top[0] in set(q.get("wrong_key", [])):
            w1 += 1
        if q["type"] == "answer":
            h1 += bool(top and top[0] in gold); h5 += bool(gold & set(top)); nff += not top
        else:
            nf += not top
    return h1, h5, w1, nf, nff, n_a, len(queries) - n_a


def main() -> None:
    eng = Engine()
    queries = []
    for s in SETS:
        for q in json.loads((config.ROOT / f"eval/{s}.json").read_text(encoding="utf-8"))["queries"]:
            q = dict(q); q["id"] = f"{s}:{q['id']}"; queries.append(q)
    data = {}
    for k, q in enumerate(queries, 1):
        data[q["id"]] = candidates(eng, q["query"])
        if k % 20 == 0:
            print(f"признаки: {k}/{len(queries)}", flush=True)
    byrow = {x["row"]: x["i"] for x in eng.rows}
    for use in (FEATS[:-1], FEATS):  # без реранкера и с ним
        cols = [FEATS.index(f) for f in use]
        preds: dict[str, list] = {}
        gkf = GroupKFold(n_splits=5)
        ids = [q["id"] for q in queries]
        for tr, te in gkf.split(ids, groups=ids):
            X, y = [], []
            for j in tr:
                q = queries[j]
                gold = {byrow[r] for r in set(q.get("same", [])) | set(q.get("substitute", []))}
                for c in data[q["id"]]:
                    X.append([c["f"][k] for k in cols]); y.append(int(c["i"] in gold))
            sc = StandardScaler().fit(X)
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
            for j in te:
                q = queries[j]
                cs = data[q["id"]]
                p = m.predict_proba(sc.transform([[c["f"][k] for k in cols] for c in cs]))[:, 1]
                preds[q["id"]] = [(float(pp), c["i"], c["rule_fail"]) for pp, c in zip(p, cs)]
        # порог «не найдено»: подбирается по всем 136 запросам сразу (без вложенной проверки), цифры слегка оптимистичны;
        # честный подбор порога — в scripts/train_decision.py
        best = max((evaluate(queries, preds, eng, t)[0] + evaluate(queries, preds, eng, t)[3] * 2, t)
                   for t in np.linspace(0.05, 0.95, 37))
        t = best[1]
        title = "с оценкой реранкера" if "rerank" in use else "без реранкера"
        for s in SETS + ["все"]:
            qs = [q for q in queries if s == "все" or q["id"].startswith(s + ":")]
            h1, h5, w1, nf, nff, n_a, n_n = evaluate(qs, preds, eng, t)
            print(f"{title} · порог {t:.2f} · {s}: верный первым {h1} из {n_a} | в пятёрке {h5} из {n_a} | "
                  f"первой другая характеристика {w1} из {len(qs)} | «не найдено» {nf} из {n_n} | ложное {nff} из {n_a}")
        if "rerank" not in use:
            m_all = LogisticRegression(max_iter=2000, class_weight="balanced")
        print()


if __name__ == "__main__":
    main()
