"""Гибридный поиск без нормализации и проверок: слова как есть плюс векторы, для сравнения с полным поиском.

Запуск: .venv/bin/python -m scripts.eval_hybrid --set eval/control_set.json
Код замороженного поиска не меняется: здесь только другой способ собрать выдачу из тех же индексов.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from app import config
from app.searcher import FIELDS, RRF_K, Engine, raw_tokens

VARIANTS = {
    "dense_name": "Только векторы, по наименованию",
    "dense_all": "Только векторы, по наименованию и карточке",
    "hybrid_name": "Гибрид: слова как есть + векторы по наименованию",
    "hybrid_all": "Гибрид: слова как есть + векторы по наименованию и карточке",
}


def ranked(eng: Engine, query: str, variant: str, k: int = 5) -> list[int]:
    qv = eng.qvec(query)
    fields = ["name"] if variant.endswith("name") else FIELDS
    cos = eng.cos(qv, fields)
    dense = [int(i) for i in np.argsort(-cos)[:100]]
    if variant.startswith("dense"):
        return dense[:k]
    words = [i for i, _ in eng.bm_raw.search(raw_tokens(query), 100)]
    fused: dict[int, float] = collections.defaultdict(float)
    for lst in (words, dense):
        for rank, i in enumerate(lst):
            fused[i] += 1.0 / (RRF_K + rank + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    args = ap.parse_args()
    queries = json.loads((config.ROOT / args.set).read_text(encoding="utf-8"))["queries"]
    eng = Engine(with_ksr=False)
    n_a = sum(q["type"] == "answer" for q in queries)
    n_n = len(queries) - n_a
    print(f"| Вариант | Верный ответ первым | Верный ответ в пятёрке | Первой стоит позиция с другой характеристикой | «Не найдено» на запросах без ответа |")
    print("|---|---|---|---|---|")
    for v, title in VARIANTS.items():
        hit1 = hit5 = wrong1 = nf = 0
        for q in queries:
            top = [eng.rows[i]["row"] for i in ranked(eng, q["query"], v)]
            gold = set(q.get("same", [])) | set(q.get("substitute", []))
            if top and top[0] in set(q.get("wrong_key", [])):
                wrong1 += 1
            if q["type"] == "answer":
                hit1 += bool(top and top[0] in gold)
                hit5 += bool(gold & set(top))
            else:
                nf += not top
        print(f"| {title} | {hit1} из {n_a} | {hit5} из {n_a} | {wrong1} из {len(queries)} | {nf} из {n_n} |")


if __name__ == "__main__":
    main()
