"""Выборка для дневной отладки.

7 строк-якорей из примеров задания плюс 43 строки по типам наименования из случайной выборки
120 строк, размеченной вручную 17.10 (seed 2026). Разметка сохраняется отдельным файлом:
на ней же меряется классификатор типа наименования.

Запуск: python3 -m scripts.select_debug
"""

from __future__ import annotations

import json
import random

import openpyxl

from app import config
from app.normalize import clean

# Номера позиций в выборке 120 (1..120) по ручной разметке типа наименования.
LABELS = {
    "article": [2, 3, 7, 10, 14, 15, 22, 23, 24, 27, 29, 30, 31, 32, 34, 36, 37, 42, 43, 47, 50, 51, 54, 55, 58, 59,
                62, 66, 67, 70, 72, 81, 90, 91, 93, 94, 96, 98, 104, 106, 107, 109, 111, 112, 115, 116],
    "standard": [1, 6, 11, 12, 13, 16, 19, 21, 26, 33, 35, 38, 39, 45, 52, 57, 60, 63, 65, 69, 71, 74, 75, 78, 79, 80,
                 82, 83, 84, 87, 88, 92, 95, 97, 100, 101, 103, 105, 108, 118, 120],
    "descriptive": [4, 5, 8, 17, 18, 20, 28, 44, 48, 49, 53, 56, 64, 68, 73, 76, 85, 86, 99, 102, 110, 113, 114, 117, 119],
    "service": [25, 41, 61, 89],
    "kit": [9, 40, 46, 77],
}
ANCHORS = [
    "Кабель контрольный КВВГЭнг(A)-LS 7х1,5",
    "Видеокамера уличная корпусная 2Мп, моторизированный объектив 2,8",
    "Уличный источник бесперебойного питания SKAT-V.12DC-18",
    "Ворота откатные 2250х6000",
    "Гофрированная труба из ПВХ (серия 9) с протяжкой d=25",
    "Трансформатор ЗНОЛ-СВЭЛ-35",
    "Шкаф ДЗТ Т-1(2), тип ШЭТ 211.02-1",
]
QUOTA = {"article": 15, "standard": 14, "descriptive": 8, "service": 3, "kit": 3}
TOTAL = 50


def load_base() -> list[tuple[str, object]]:
    wb = openpyxl.load_workbook(config.BASE_XLSX, read_only=True)
    ws = wb["База позиций"]
    return [(str(r[0]) if r[0] is not None else "", r[1]) for r in ws.iter_rows(min_row=2, values_only=True)]


def main() -> None:
    base = load_base()
    random.seed(2026)
    sample = random.sample(range(len(base)), 120)
    assert clean(base[sample[0]][0]).startswith("Проволока арматурная"), "выборка 120 не воспроизвелась"
    assert clean(base[sample[1]][0]).startswith("Быстрозажимная гайка"), "выборка 120 не воспроизвелась"
    label_of = {pos: lab for lab, positions in LABELS.items() for pos in positions}
    labeled = [{"row": idx + 2, "idx": idx, "name": base[idx][0], "price": base[idx][1], "label": label_of[k]}
               for k, idx in enumerate(sample, 1)]
    out_labels = config.DATA / "labels" / "nametype_sample120.json"
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out_labels.write_text(json.dumps(labeled, ensure_ascii=False, indent=1), encoding="utf-8")

    by_idx = {x["idx"]: x for x in labeled}
    chosen: list[dict] = []
    seen: set[int] = set()
    for prefix in ANCHORS:
        idx = next(i for i, (n, _) in enumerate(base) if clean(n).startswith(clean(prefix)))
        item = dict(by_idx.get(idx) or {"row": idx + 2, "idx": idx, "name": base[idx][0], "price": base[idx][1], "label": None})
        item["anchor"] = True
        chosen.append(item)
        seen.add(idx)
    for lab, quota in QUOTA.items():
        for x in [x for x in labeled if x["label"] == lab and x["idx"] not in seen][:quota]:
            chosen.append({**x, "anchor": False})
            seen.add(x["idx"])
    for x in labeled:
        if len(chosen) >= TOTAL:
            break
        if x["idx"] not in seen:
            chosen.append({**x, "anchor": False})
            seen.add(x["idx"])
    out = config.DATA / "debug" / "rows_50.json"
    out.write_text(json.dumps(chosen[:TOTAL], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"разметка 120: {out_labels.relative_to(config.ROOT)}; выборка {len(chosen[:TOTAL])}: {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
