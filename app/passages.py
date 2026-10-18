"""Фильтр пассажей выдачи: оставляем то, что говорит именно об этой позиции.

Документ про соседний артикул выбрасывается целиком, торговые фразы вырезаются,
повторы склеиваются. Пассажи с кодом строки и её числами идут первыми."""

from __future__ import annotations

import re

from .normalize import code_key, codes, fold, levenshtein, numbers

TRADE = re.compile(
    r"купить|оптом|в розницу|доставк|по низкой|выгодн|скидк|акци[яи]|в наличии|интернет-магазин|заказать|"
    r"отзыв|бонус|кэшбэк|рассрочк|\bцен[аыуе]\b|\bруб\b|₽|\bндс\b|самовывоз|оплат",
    re.I,
)
_SENT = re.compile(r"(?<=[.!?])\s+")
MAX_PASSAGES = 12


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text or "") if s.strip()]


def filter_docs(name: str, docs: list[dict]) -> dict:
    row_codes = {code_key(c) for c in codes(name)}
    long_codes = {c for c in row_codes if len(c) >= 6}  # короткие коды дают ложных «соседей»
    row_nums = numbers(name)
    kept: list[dict] = []
    seen: set[str] = set()
    neighbor_docs: list[str] = []
    trade_cut = 0
    for i, doc in enumerate(docs, 1):
        texts = [doc["title"], doc.get("headline", "")] + list(doc["passages"])
        joined = " ".join(t for t in texts if t)
        joined_key = code_key(joined)
        exact = any(c in joined_key for c in row_codes)
        if long_codes and not exact:
            doc_codes = {code_key(c) for c in codes(joined)}
            near = [dc for dc in doc_codes for rc in long_codes if 0 < levenshtein(rc, dc, 2) <= 2]
            if near:
                neighbor_docs.append(f"{doc['domain']}: {near[0]}")
                continue
        for text in [doc.get("headline", "")] + list(doc["passages"]):
            if not text:
                continue
            sents = _sentences(text)
            clean_sents = [s for s in sents if not TRADE.search(s)]
            if len(clean_sents) < len(sents):
                trade_cut += 1
            body = " ".join(clean_sents).strip()
            if len(body) < 25:
                continue
            key = fold(body)[:160]
            if key in seen:
                continue
            seen.add(key)
            body_key = code_key(body + " " + doc["title"])
            kept.append({
                "doc": i, "domain": doc["domain"], "title": doc["title"], "text": body,
                "hit_code": any(c in body_key for c in row_codes),
                "hit_nums": len(row_nums & numbers(body)),
            })
    kept.sort(key=lambda p: (not p["hit_code"], -p["hit_nums"], p["doc"]))
    need = max(1, (len(row_nums) + 1) // 2)
    specific = [p for p in kept if p["hit_code"] or (row_nums and p["hit_nums"] >= need)]
    kind = "specific" if specific else ("general" if kept else "none")
    chosen = (specific or kept)[:MAX_PASSAGES]
    return {"passages": chosen, "kind": kind, "neighbor_docs": neighbor_docs,
            "trade_cut": trade_cut, "n_kept_total": len(kept)}
