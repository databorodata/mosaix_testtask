"""Автоматический разбор прогона: распределения, флаги на каждую строку, примеры.

Запуск: python3 -m scripts.analyze_full --prep data/full/prep_all.jsonl --out data/full
Пишет stats.md (сводка) и flags.jsonl (флаги по строкам).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics

from app import config
from app.nametype import NAMES
from app.normalize import code_key, fold

MARKETPLACES = ("market.yandex", "ozon.ru", "wildberries", "avito.ru", "satu.kz", "prom.uz", "aliexpress", "megamarket",
                "lemanapro", "petrovich", "vseinstrumenti")
FILLER = re.compile(r"обеспечива\w* (?:надежн|надёжн|безопасн|эффективн|долговечн|высок)|широко примен|различн\w+ (?:отрасл|сфер|област)"
                    r"|высок\w* качеств|надежн\w* работ|надёжн\w* работ|идеальн", re.I)
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def pct(a: int, b: int) -> str:
    return f"{a} ({a / b * 100:.0f}%)" if b else "0"


def flags_for(r: dict, model: str) -> list[str]:
    f = r["filter"]
    res = r["cards"][model]
    card = res.get("card") or {}
    v = res["verify"]
    out = []
    if r.get("search_error"):
        out.append("SEARCH_ERR")
    if f["kind"] == "none":
        out.append("CTX_NONE")
    elif f["kind"] == "general":
        out.append("CTX_GENERAL")
    if f["neighbor_docs"]:
        out.append("NEIGHBOR_DROP")
    doms = [p["domain"] for p in f["passages"]]
    if doms and sum(any(m in d for m in MARKETPLACES) for d in doms) / len(doms) >= 0.5:
        out.append("MARKET_DOMINANT")
    if not card:
        out.append("CARD_ERR")
        return out
    if v["status"] == "fail":
        out.append("VERIFY_FAIL")
    elif v["status"] == "warn":
        out.append("VERIFY_WARN")
    name_f = fold(r["name"])
    manu = card.get("manufacturer", "").strip()
    if manu and not any(fold(w) in name_f for w in re.findall(r"[\wА-Яа-яЁё\-]{3,}", manu)):
        out.append("MANUF_LEAK")
    syns = card.get("synonyms", [])
    keys = [code_key(s) for s in syns]
    if len(set(keys)) < len(keys):
        out.append("SYN_DUP")
    if any(code_key(s) == code_key(r["name"]) for s in syns) and len(r["name"]) > 60:
        out.append("SYN_COPY")
    row_words = {fold(w) for w in LATIN_WORD.findall(r["name"] + " " + card.get("model_code", ""))}
    if any(fold(w) not in row_words for s in syns for w in LATIN_WORD.findall(s) if len(w) >= 4):
        out.append("SYN_LATIN_NEW")
    if FILLER.search(card.get("description", "")):
        out.append("FILLER")
    if card.get("confidence") == "низкая":
        out.append("LOW_CONF")
    k = r.get("ksr") or {}
    if k.get("status") == "error":
        out.append("KSR_ERR")
    if "понижен" in (k.get("reason") or ""):
        out.append("KSR_DOWNGRADE")
    if k.get("status") == "none":
        out.append("KSR_NONE")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [json.loads(x) for x in (config.ROOT / args.prep).read_text(encoding="utf-8").splitlines() if x.strip()]
    model = list(recs[0]["cards"].keys())[0]
    n = len(recs)
    outdir = config.ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    flags = []
    for i, r in enumerate(recs, 1):
        flags.append({"n": i, "row": r["row"], "ntype": r["ntype"], "flags": flags_for(r, model)})
    with (outdir / "flags.jsonl").open("w", encoding="utf-8") as fh:
        for x in flags:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    L: list[str] = [f"# Автоматическая сводка прогона: {n} строк, модель {model}", ""]
    types = collections.Counter(r["ntype"] for r in recs)
    L += ["## Тип наименования", "", "| Тип | Строк |", "|---|---|"]
    L += [f"| {NAMES[t]} | {pct(c, n)} |" for t, c in types.most_common()] + [""]

    L += ["## Поиск и контекст", ""]
    se = sum(1 for r in recs if r.get("search_error"))
    L.append(f"- Ошибок поиска: {se}")
    kinds = collections.Counter(r["filter"]["kind"] for r in recs)
    L.append("- Контекст: " + ", ".join(f"{k} {pct(c, n)}" for k, c in kinds.most_common()))
    L.append(f"- Документов в выдаче, медиана: {statistics.median(r['n_docs'] for r in recs)}")
    L.append(f"- Пассажей после фильтра, медиана: {statistics.median(len(r['filter']['passages']) for r in recs)}")
    nb = [r for r in recs if r["filter"]["neighbor_docs"]]
    L.append(f"- Строк с отброшенными соседями: {pct(len(nb), n)}, документов отброшено {sum(len(r['filter']['neighbor_docs']) for r in nb)}")
    L += ["", "| Тип | specific | general | none |", "|---|---|---|---|"]
    for t in types:
        c = collections.Counter(r["filter"]["kind"] for r in recs if r["ntype"] == t)
        L.append(f"| {NAMES[t]} | {c['specific']} | {c['general']} | {c['none']} |")
    dom = collections.Counter(p["domain"] for r in recs for p in r["filter"]["passages"])
    total_p = sum(dom.values())
    mk = sum(c for d, c in dom.items() if any(m in d for m in MARKETPLACES))
    L += ["", f"- Пассажей всего {total_p}, из них с маркетплейсов {pct(mk, total_p)}", "", "Топ-25 доменов по пассажам:", ""]
    L += [f"- {d}: {c}" for d, c in dom.most_common(25)] + [""]

    L += ["## Карточки", ""]
    res = [r["cards"][model] for r in recs]
    errs = [r for r in recs if not r["cards"][model].get("card")]
    L.append(f"- Ошибок модели (нет карточки): {len(errs)}")
    vs = collections.Counter(x["verify"]["status"] for x in res)
    L.append("- Проверка кодом: " + ", ".join(f"{k} {pct(c, n)}" for k, c in vs.most_common()))
    iss = collections.Counter()
    for x in res:
        for i in x["verify"]["issues"]:
            iss[i.split(":")[0]] += 1
    L.append("- Типы замечаний: " + ", ".join(f"{k} {c}" for k, c in iss.most_common()))
    conf = collections.Counter((x.get("card") or {}).get("confidence", "") for x in res)
    L.append("- Уверенность модели: " + ", ".join(f"{k} {c}" for k, c in conf.most_common()))
    words = [x["verify"]["desc_words"] for x in res if x.get("card")]
    L.append(f"- Слов в описании: медиана {statistics.median(words)}, минимум {min(words)}, максимум {max(words)}")
    pt = [x["usage"]["prompt_tokens"] for x in res if x["usage"]["prompt_tokens"]]
    ct = [x["usage"]["completion_tokens"] for x in res if x["usage"]["completion_tokens"]]
    cost = sum(x["cost_rub"] for x in res if not x.get("cached"))
    L.append(f"- Токены: вход медиана {statistics.median(pt) if pt else 0}, выход медиана {statistics.median(ct) if ct else 0}; новые вызовы на {cost:.2f} ₽")
    L += ["", "| Тип | ok | warn | fail | error |", "|---|---|---|---|---|"]
    for t in types:
        c = collections.Counter(r["cards"][model]["verify"]["status"] for r in recs if r["ntype"] == t)
        L.append(f"| {NAMES[t]} | {c['ok']} | {c['warn']} | {c['fail']} | {c['error']} |")

    L += ["", "## Флаги", "", "| Флаг | Строк |", "|---|---|"]
    fc = collections.Counter(f for x in flags for f in x["flags"])
    L += [f"| {f} | {pct(c, n)} |" for f, c in fc.most_common()] + [""]

    L += ["## КСР", ""]
    ks = collections.Counter((r.get("ksr") or {}).get("status", "") for r in recs)
    L.append("- Статус: " + ", ".join(f"{k} {pct(c, n)}" for k, c in ks.most_common()))
    kcost = sum((r.get("ksr") or {}).get("cost_rub", 0) for r in recs)
    L.append(f"- Выбор КСР моделью: {kcost:.2f} ₽")
    L += ["", "| Тип | code | group | none | service | error |", "|---|---|---|---|---|---|"]
    for t in types:
        c = collections.Counter((r.get("ksr") or {}).get("status", "") for r in recs if r["ntype"] == t)
        L.append(f"| {NAMES[t]} | {c['code']} | {c['group']} | {c['none']} | {c['service']} | {c['error']} |")
    books = collections.Counter(((r.get("ksr") or {}).get("path") or "").split(" / ")[0] for r in recs
                                if (r.get("ksr") or {}).get("status") == "code")
    L += ["", "Коды КСР по книгам:", ""] + [f"- {b or 'без книги'}: {c}" for b, c in books.most_common(15)] + [""]

    L += ["## Примеры по флагам", ""]
    for flag in ("CARD_ERR", "SEARCH_ERR", "VERIFY_FAIL", "MANUF_LEAK", "SYN_DUP", "MARKET_DOMINANT", "CTX_NONE", "KSR_DOWNGRADE"):
        ex = [recs[x["n"] - 1] for x in flags if flag in x["flags"]][:6]
        if not ex:
            continue
        L.append(f"### {flag}")
        for r in ex:
            card = r["cards"][model].get("card") or {}
            extra = "; ".join(r["cards"][model]["verify"]["issues"])[:140]
            if flag == "MANUF_LEAK":
                extra = f"производитель в карточке: {card.get('manufacturer', '')}"
            if flag == "KSR_DOWNGRADE":
                extra = (r.get("ksr") or {}).get("name", "")[:120]
            L.append(f"- {r['name'][:110]} · {extra}")
        L.append("")
    (outdir / "stats.md").write_text("\n".join(L), encoding="utf-8")
    print(f"сводка: {(outdir / 'stats.md').relative_to(config.ROOT)}; флаги: {(outdir / 'flags.jsonl').relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
