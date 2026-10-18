"""Компактные выжимки прогона для ручного разбора по частям.

Запуск: python3 -m scripts.make_digests --prep data/full/prep_all.jsonl --out data/full/digest --chunks 10
"""

from __future__ import annotations

import argparse
import json

from app import config
from app.nametype import NAMES

KSR_TITLE = {"code": "тот же ресурс", "group": "та же группа", "none": "нет в КСР", "service": "услуга", "error": "ошибка"}


def one(n: int, r: dict, model: str, flags: list[str]) -> str:
    f = r["filter"]
    res = r["cards"][model]
    card = res.get("card") or {}
    v = res["verify"]
    k = r.get("ksr") or {}
    doms = ", ".join(dict.fromkeys(p["domain"] for p in f["passages"]))[:120]
    lines = [
        f"#{n} · Excel {r['row']} · {NAMES[r['ntype']]} · контекст {f['kind']}, пассажей {len(f['passages'])} · {doms}"
        + (f" · отброшены соседи: {', '.join(f['neighbor_docs'][:3])}" if f["neighbor_docs"] else ""),
        f"СТРОКА: {r['name']}",
    ]
    for p in f["passages"][:2]:
        lines.append(f"П: {p['domain']}: {p['text'][:170]}")
    if not card:
        lines.append(f"КАРТОЧКИ НЕТ: {res.get('error')}")
    else:
        attrs = "; ".join(f"{a['name']}: {a['value']} {a['unit']}".strip() for a in card.get("attributes", []))[:300]
        lines += [
            f"НАЗВАНИЕ: {card.get('canonical_name', '')}",
            f"ОПИСАНИЕ: {card.get('description', '')}",
            f"СИНОНИМЫ: {'; '.join(card.get('synonyms', []))}",
            f"ХАРАКТЕРИСТИКИ: {attrs}",
            f"ПРОИЗВОДИТЕЛЬ: {card.get('manufacturer', '') or '—'} · МОДЕЛЬ: {card.get('model_code', '') or '—'} · "
            f"уверенность {card.get('confidence', '')} · проверка кодом {v['status']}"
            + (f" ({'; '.join(v['issues'])})" if v["issues"] else ""),
        ]
    ks = f"КСР: {KSR_TITLE.get(k.get('status', ''), k.get('status', ''))}"
    if k.get("name"):
        ks += f" · {k.get('code') or 'без кода'} «{k['name'][:150]}»" + (f" · {k['unit']}" if k.get("unit") else "")
    lines.append(ks + (f" · почему: {k.get('reason', '')[:200]}" if k.get("reason") else ""))
    if flags:
        lines.append(f"ФЛАГИ: {', '.join(flags)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--flags", default="")
    args = ap.parse_args()
    recs = [json.loads(x) for x in (config.ROOT / args.prep).read_text(encoding="utf-8").splitlines() if x.strip()]
    model = list(recs[0]["cards"].keys())[0]
    flags = {}
    if args.flags:
        for x in (config.ROOT / args.flags).read_text(encoding="utf-8").splitlines():
            if x.strip():
                d = json.loads(x)
                flags[d["n"]] = d["flags"]
    out = config.ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    size = -(-len(recs) // args.chunks)
    for c in range(args.chunks):
        part = list(enumerate(recs, 1))[c * size:(c + 1) * size]
        if not part:
            continue
        text = "\n\n".join(one(n, r, model, flags.get(n, [])) for n, r in part)
        (out / f"chunk_{c + 1:02d}.txt").write_text(text, encoding="utf-8")
        print(f"chunk_{c + 1:02d}: строки {part[0][0]}–{part[-1][0]}, слов {len(text.split())}")


if __name__ == "__main__":
    main()
