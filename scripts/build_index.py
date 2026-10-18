"""Векторный индекс базы: четыре поля на строку (наименование, каноническое название, синонимы, описание).

Запуск: .venv/bin/python -m scripts.build_index --model e5-base
Пишет data/index/<model>/<поле>.npy и meta.json. Строки без карточки получают нулевые векторы карточных полей.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from app import config
from app.normalize import clean
from app.searcher import FIELDS, INDEX_DIR, MODELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-base", choices=list(MODELS))
    ap.add_argument("--prep", default="data/full/prep_all.jsonl")
    args = ap.parse_args()
    cfg = MODELS[args.model]
    recs = [json.loads(x) for x in (config.ROOT / args.prep).read_text(encoding="utf-8").splitlines() if x.strip()]
    mkey = list(recs[0]["cards"].keys())[0]
    texts: dict[str, list[str]] = {f: [] for f in FIELDS}
    for r in recs:
        card = r["cards"][mkey].get("card") or {}
        texts["name"].append(clean(r["name"]))
        texts["canonical"].append(card.get("canonical_name", ""))
        texts["synonyms"].append("; ".join(card.get("synonyms", [])))
        texts["description"].append(card.get("description", ""))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["hf"])
    out = INDEX_DIR / args.model
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for f in FIELDS:
        vec = np.zeros((len(recs), model.get_sentence_embedding_dimension()), dtype=np.float32)
        idx = [i for i, t in enumerate(texts[f]) if t.strip()]
        enc = model.encode([cfg["doc"] + texts[f][i] for i in idx], batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)
        vec[idx] = enc
        np.save(out / f"{f}.npy", vec.astype(np.float16))
        print(f"{f}: {len(idx)} текстов, {time.time() - t0:.0f} с")
    meta = {"model": cfg["hf"], "rows": [r["row"] for r in recs], "prep": args.prep, "fields": FIELDS}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"индекс: {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
