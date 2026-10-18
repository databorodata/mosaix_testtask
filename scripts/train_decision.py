"""Обучение решения «подходит или нет» и честный подбор порога «не найдено».

Запуск: .venv/bin/python -m scripts.train_decision
Данные: наборы настройки dev_set, control_set, big_set (136 запросов). Итоговые наборы не используются.

Честность цифр: качество и порог считаются по перекрёстной проверке GroupKFold по запросам —
предсказание для запроса всегда делает модель, которая его не видела. Порог выбирается по этим же
внефолдовым предсказаниям. Итоговая модель обучается на всех 136 запросах и сохраняется
в data/model/decision.json вместе с порогом. Цифры на самих наборах настройки поэтому слегка
оптимистичны — честная оценка это перекрёстная проверка здесь и итоговые слепые наборы.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from app import config
from app.decision import FEATS, LIGHT_FEATS, MODEL_FILE, build
from app.searcher import Engine

SETS = ["dev_set", "control_set", "big_set"]
GRID = np.linspace(0.05, 0.95, 37)


def evaluate(queries, preds, rows, t):
    """Метрики ТЗ при пороге t: верный первым, в пятёрке, честное и ложное «не найдено»."""
    h1 = h5 = nf = nff = 0
    n_a = sum(q["type"] == "answer" for q in queries)
    for q in queries:
        cands = [(p, i) for p, i, fail in preds[q["id"]] if not fail]
        cands.sort(key=lambda x: -x[0])
        top = [rows[i]["row"] for p, i in cands[:5]] if cands and cands[0][0] >= t else []
        gold = set(q.get("same", [])) | set(q.get("substitute", []))
        if q["type"] == "answer":
            h1 += bool(top and top[0] in gold); h5 += bool(gold & set(top)); nff += not top
        else:
            nf += not top
    return h1, h5, nf, nff, n_a, len(queries) - n_a


def main() -> None:
    eng = Engine()
    queries = []
    for s in SETS:
        for q in json.loads((config.ROOT / f"eval/{s}.json").read_text(encoding="utf-8"))["queries"]:
            q = dict(q); q["id"] = f"{s}:{q['id']}"; q["set"] = s; queries.append(q)
    print(f"запросов настройки: {len(queries)}")
    feats = {}
    for k, q in enumerate(queries, 1):
        _, cands = build(eng, q["query"], use_rerank=True)
        feats[q["id"]] = cands
        if k % 20 == 0:
            print(f"признаки: {k}/{len(queries)}", flush=True)
    byrow = {x["row"]: x["i"] for x in eng.rows}
    report = {}
    out = {"version": 1, "feats_all": FEATS, "models": {}}
    for name, use in (("rerank", FEATS), ("light", LIGHT_FEATS)):
        cols = [FEATS.index(f) for f in use]
        # Внефолдовые предсказания: модель не видела запрос, для которого предсказывает.
        preds: dict[str, list] = {}
        ids = [q["id"] for q in queries]
        for tr, te in GroupKFold(n_splits=5).split(ids, groups=ids):
            X, y = [], []
            for j in tr:
                q = queries[j]
                gold = {byrow[r] for r in set(q.get("same", [])) | set(q.get("substitute", []))}
                for c in feats[q["id"]]:
                    X.append([c.feats[k] for k in cols]); y.append(int(c.i in gold))
            sc = StandardScaler().fit(X)
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
            for j in te:
                q = queries[j]
                cs = feats[q["id"]]
                p = m.predict_proba(sc.transform([[c.feats[k] for k in cols] for c in cs]))[:, 1]
                preds[q["id"]] = [(float(pp), c.i, c.rule_fail) for pp, c in zip(p, cs)]
        # Порог по тем же внефолдовым предсказаниям: цель — верный первым + 2 × честное «не найдено».
        best_t, best_v = 0.5, -1
        for t in GRID:
            h1, _, nf, _, _, _ = evaluate(queries, preds, eng.rows, t)
            if h1 + 2 * nf > best_v:
                best_v, best_t = h1 + 2 * nf, float(t)
        lines = []
        for s in SETS + ["все"]:
            qs = [q for q in queries if s == "все" or q["set"] == s]
            h1, h5, nf, nff, n_a, n_n = evaluate(qs, preds, eng.rows, best_t)
            lines.append(f"  {s}: верный первым {h1} из {n_a} | в пятёрке {h5} из {n_a} | "
                         f"«не найдено» {nf} из {n_n} | ложное {nff} из {n_a}")
        report[name] = {"threshold": best_t, "cv": lines}
        print(f"\n{name} · перекрёстная проверка · порог {best_t:.2f}")
        print("\n".join(lines))
        # Итоговая модель: обучение на всех 136 запросах, порог — из перекрёстной проверки.
        X, y = [], []
        for q in queries:
            gold = {byrow[r] for r in set(q.get("same", [])) | set(q.get("substitute", []))}
            for c in feats[q["id"]]:
                X.append([c.feats[k] for k in cols]); y.append(int(c.i in gold))
        sc = StandardScaler().fit(X)
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
        out["models"][name] = {"feats": use, "mean": [float(v) for v in sc.mean_],
                               "scale": [float(v) for v in sc.scale_], "coef": [float(v) for v in m.coef_[0]],
                               "intercept": float(m.intercept_[0]), "threshold": best_t,
                               "cv": [l.strip() for l in lines]}
        top = sorted(zip(use, m.coef_[0]), key=lambda t: -abs(t[1]))
        print("  веса: " + ", ".join(f"{n} {w:+.2f}" for n, w in top))
    out["trained_on"] = {"sets": SETS, "n_queries": len(queries)}
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nмодель сохранена: {MODEL_FILE}")


if __name__ == "__main__":
    main()
