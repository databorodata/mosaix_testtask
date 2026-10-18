"""Сверка классификатора типа наименования с ручной разметкой 120 строк.

Запуск: python3 -m scripts.eval_nametype
"""

from __future__ import annotations

import collections
import json

from app import config
from app.nametype import LABEL_GROUP, classify


def main() -> None:
    labeled = json.loads((config.DATA / "labels" / "nametype_sample120.json").read_text(encoding="utf-8"))
    pairs = [(x["label"], LABEL_GROUP[classify(x["name"])], x["name"]) for x in labeled]
    hits = sum(1 for t, p, _ in pairs if t == p)
    print(f"совпало с ручной разметкой: {hits} из {len(pairs)}")
    confusion = collections.Counter((t, p) for t, p, _ in pairs if t != p)
    for (t, p), c in confusion.most_common():
        print(f"  размечено {t}, классификатор {p}: {c}")
    print("примеры расхождений:")
    for t, p, n in [x for x in pairs if x[0] != x[1]][:15]:
        print(f"  [{t} → {p}] {n[:100]}")


if __name__ == "__main__":
    main()
