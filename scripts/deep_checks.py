"""Точечные проверки полного прогона к отчёту: закономерности из ручного чтения, посчитанные кодом по всей базе.

Запуск: python3 -m scripts.deep_checks --prep data/full/prep_all.jsonl \
    --agents data/full/analysis/agent_merged.json --out data/full/analysis/deep_checks.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import re

from app import config
from app.nametype import NAMES
from app.normalize import code_key, fold, numbers
from app.verify import card_text

V_ORDER = ("ок", "частично", "ошибка", "мало данных")
COMMERCE_SYN = re.compile(r"купить|цена|стоимост|недорог|оптом|в наличии|доставк|интернет-магазин|прайс", re.I)
COMMERCE_DESC = re.compile(r"Москв|Санкт-Петербург|Екатеринбург|Краснодар|Новосибирск|оптов|розничн|в наличии|доставк|купить|\bцен[аыу]\b", re.I)
META = re.compile(r"не указан|не уточн|в наименовании|во входн|фрагмент|исходн\w* данн|предоставленн\w* (?:данн|информац)"
                  r"|отсутству\w* (?:сведен|информац|данн)|информаци\w* (?:не|отсутств)", re.I)
DOUBLE_NUM = re.compile(r"\(\s*\d+\.\d+\s*(?:мм2|мм|м2|м3|м|кВ|В|кг|т)?\s*\)")
CABLE = re.compile(r"кабел|провод", re.I)
KV1 = re.compile(r"-1000\b|-1\s*кВ|\b1\s*кВ")
KV066 = re.compile(r"0[,.]66\s*кВ|660\s*В")
ANALOG = re.compile(r"по типу|или аналог|или эквивалент|аналог", re.I)
NOTE = re.compile(r"\(\s*прим", re.I)
DOTTED = re.compile(r"\d+\.\d+\.\d+")
RANGE = re.compile(r"\d+(?:[.,]\d+)?\s*[-–…]\s*\d+(?:[.,]\d+)?")
CONCRETE = re.compile(r"\b[ВBМM]\d+(?:[.,]\d+)?\b")


def vdist(rows: list[int], ag: dict) -> str:
    c = collections.Counter(ag[str(n)]["verdict"] for n in rows)
    return " / ".join(f"{v} {c[v]}" for v in V_ORDER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--agents", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [json.loads(x) for x in (config.ROOT / args.prep).read_text(encoding="utf-8").splitlines() if x.strip()]
    ag = json.loads((config.ROOT / args.agents).read_text(encoding="utf-8"))
    model = list(recs[0]["cards"].keys())[0]
    N = len(recs)
    L: list[str] = []

    def card(r: dict) -> dict:
        return r["cards"][model].get("card") or {}

    def has_tag(n: int, t: str) -> bool:
        return t in ag[str(n)]["ntags"]

    rows = list(enumerate(recs, 1))
    with_card = [(n, r) for n, r in rows if card(r)]

    # 1. Напряжение кабелей
    kv1 = [n for n, r in with_card if CABLE.search(r["name"]) and KV1.search(r["name"])]
    kv_bad = [n for n in kv1 if KV066.search(card_text(card(recs[n - 1])))]
    L += ["## 1. Кабели «-1000» (1 кВ), у которых в карточке 0,66 кВ или 660 В",
          f"кабельных строк с «-1000» или «1 кВ»: {len(kv1)}; с 0,66 кВ / 660 В в карточке: {len(kv_bad)}",
          f"примеры: {kv_bad[:12]}", ""]

    # 2. Двойная запись чисел и удвоенные единицы
    dbl = [n for n, r in with_card if DOUBLE_NUM.search(card(r).get("description", ""))]
    udup = [n for n, r in with_card
            if any(a["unit"].strip() and a["value"].strip().endswith(a["unit"].strip()) for a in card(r).get("attributes", []))]
    L += ["## 2. Двойная запись чисел в описании и удвоенные единицы в характеристиках",
          f"описаний со скобкой вида «(1.5 мм2)»: {len(dbl)}; строк с единицей и в значении, и в поле единицы: {len(udup)}",
          f"примеры двойной записи: {dbl[:8]}; удвоенных единиц: {udup[:8]}", ""]

    # 3. Торговые слова и города
    syn_c = [n for n, r in with_card if any(COMMERCE_SYN.search(s) for s in card(r).get("synonyms", []))]
    desc_c = [n for n, r in with_card if COMMERCE_DESC.search(card(r).get("description", ""))]
    L += ["## 3. Торговые слова в синонимах и описаниях",
          f"синонимы с «купить», «цена», «оптом» и т. п.: {len(syn_c)} строк; примеры {syn_c[:10]}",
          f"описания с городами, «оптом», «в наличии», «доставка», «цена»: {len(desc_c)} строк; примеры {desc_c[:10]}", ""]

    # 4. Мета-текст и длина описания
    meta = [n for n, r in with_card if META.search(card(r).get("description", ""))]
    words = {n: r["cards"][model]["verify"]["desc_words"] for n, r in with_card}
    L += ["## 4. Мета-текст и длина описания",
          f"описаний с фразами «не указан», «в наименовании», «фрагмент», «информация отсутствует»: {len(meta)}; примеры {meta[:10]}",
          f"слов в описании: меньше 60: {sum(w < 60 for w in words.values())}, 60–100: {sum(60 <= w <= 100 for w in words.values())}, "
          f"101–130: {sum(100 < w <= 130 for w in words.values())}, больше 130: {sum(w > 130 for w in words.values())}", ""]

    # 5. Дубли строк
    groups = collections.defaultdict(list)
    for n, r in rows:
        groups[" ".join(fold(r["name"]).split())].append(n)
    dups = [g for g in groups.values() if len(g) > 1]
    incons = [g for g in dups if len({code_key(card(recs[n - 1]).get("canonical_name", "")) for n in g}) > 1]
    k_incons = [g for g in dups if len({((recs[n - 1].get("ksr") or {}).get("status"), (recs[n - 1].get("ksr") or {}).get("code")) for n in g}) > 1]
    L += ["## 5. Дословные дубли строк",
          f"групп дублей: {len(dups)}, строк в них: {sum(len(g) for g in dups)}",
          f"групп с разным каноническим названием: {len(incons)}; примеры {incons[:6]}",
          f"групп с разной привязкой КСР: {len(k_incons)}; примеры {k_incons[:6]}", ""]

    # 6. Производитель по типам
    L += ["## 6. Поле «производитель» и тег vendor_leak по типам наименования",
          "| Тип | строк | производитель заполнен | производителя нет в строке | тег vendor_leak у агентов |", "|---|---|---|---|---|"]
    for t in NAMES:
        tn = [n for n, r in rows if r["ntype"] == t]
        if not tn:
            continue
        filled = [n for n in tn if card(recs[n - 1]).get("manufacturer", "").strip()]
        leak = [n for n in filled if not any(fold(w) in fold(recs[n - 1]["name"])
                                             for w in re.findall(r"[\wА-Яа-яЁё\-]{3,}", card(recs[n - 1])["manufacturer"]))]
        vl = [n for n in tn if has_tag(n, "vendor_leak")]
        L.append(f"| {NAMES[t]} | {len(tn)} | {len(filled)} | {len(leak)} | {len(vl)} |")
    L.append("")

    # 7. Скобки «прим.» и «по типу / или аналог»
    note = [n for n, r in rows if NOTE.search(r["name"])]
    analog = [n for n, r in rows if ANALOG.search(r["name"])]
    an_manuf = [n for n in analog if card(recs[n - 1]).get("manufacturer", "").strip()]
    L += ["## 7. Скобки «(прим. …)» и «по типу / или аналог»",
          f"строк со скобкой «прим.»: {len(note)}; вердикты агентов: {vdist(note, ag)}; примеры {note[:8]}",
          f"строк с «по типу», «или аналог», «аналог»: {len(analog)}; вердикты: {vdist(analog, ag)}; "
          f"с заполненным производителем: {len(an_manuf)}; примеры {analog[:8]}", ""]

    # 8. КСР: ресурсы атомной отрасли
    P = json.loads((config.CACHE / "ksr_v43_positions.json").read_text(encoding="utf-8"))
    safety = sum("класс безопасности" in p["name"] for p in P)
    bound = [n for n, r in rows if "класс безопасности" in ((r.get("ksr") or {}).get("name") or "")]
    L += ["## 8. КСР: ресурсы с классом безопасности (атомная отрасль)",
          f"позиций КСР с «класс безопасности»: {safety} из {len(P)}",
          f"строк базы, привязанных к таким ресурсам: {len(bound)}; статусы: "
          f"{dict(collections.Counter(recs[n - 1]['ksr']['status'] for n in bound))}; примеры {bound[:10]}", ""]

    # 9. Причины понижения кода до группы
    down = [(n, r) for n, r in rows if "понижен" in ((r.get("ksr") or {}).get("reason") or "")]
    causes = collections.Counter()
    by_cause = collections.defaultdict(list)
    for n, r in down:
        kname = r["ksr"]["name"]
        have = numbers(r["name"]) | numbers(card_text(card(r)))
        miss = numbers(kname) - have
        cs = set()
        if DOTTED.search(kname) and any(m in numbers(" ".join(DOTTED.findall(kname))) for m in miss):
            cs.add("обозначение с точками (12.4.3) читается как дробь")
        if "класс безопасности" in kname:
            cs.add("класс безопасности")
        if any(m in numbers(" ".join(RANGE.findall(kname))) for m in miss):
            cs.add("диапазон в ресурсе")
        if any(m in numbers(" ".join(CONCRETE.findall(kname))) for m in miss):
            cs.add("класс или марка бетона, стали")
        if not cs:
            cs.add("прочее: в ресурсе есть параметр, которого нет в строке")
        for c in cs:
            causes[c] += 1
            by_cause[c].append(n)
    L += ["## 9. Почему код КСР понижен до группы",
          f"понижений: {len(down)}; вердикты агентов по КСР: "
          f"{dict(collections.Counter(ag[str(n)]['ksr'] for n, _ in down))}"]
    for c, k in causes.most_common():
        ns = by_cause[c]
        miss_share = sum(ag[str(n)]["ksr"] == "пропуск" for n in ns)
        L.append(f"- {c}: {k}, из них агенты считают пропуском {miss_share}; примеры {ns[:8]}")
    L.append("")

    # 10. Выбор моделью против правила
    cross = collections.Counter(((r.get("ksr") or {}).get("status"), (r.get("ksr_rule") or {}).get("status")) for _, r in rows)
    miss_rule = [n for n, r in rows if ag[str(n)]["ksr"] == "пропуск" and (r.get("ksr_rule") or {}).get("status") == "code"]
    L += ["## 10. КСР: выбор моделью против правила",
          "модель × правило: " + ", ".join(f"{a}/{b} {c}" for (a, b), c in cross.most_common()),
          f"из {sum(ag[str(n)]['ksr'] == 'пропуск' for n, _ in rows)} пропусков у правила был код в {len(miss_rule)} строках", ""]

    # 11. Качество контекста, маркетплейсы, уверенность × вердикт
    L += ["## 11. Что предсказывает плохую карточку"]
    for title, key in (("контекст", lambda r: r["filter"]["kind"]),
                       ("уверенность модели", lambda r: card(r).get("confidence", "нет карточки")),
                       ("проверка кодом", lambda r: r["cards"][model]["verify"]["status"])):
        L.append(f"{title}:")
        for val in sorted({key(r) for _, r in rows}):
            ns = [n for n, r in rows if key(r) == val]
            bad = sum(ag[str(n)]["verdict"] == "ошибка" or has_tag(n, "other_product") for n in ns)
            L.append(f"  {val}: {len(ns)} строк; {vdist(ns, ag)}; ошибка или чужой товар {bad} ({bad / len(ns) * 100:.0f}%)")
    other = [n for n, _ in rows if has_tag(n, "other_product") or ag[str(n)]["verdict"] == "ошибка"]
    dom = collections.Counter(d for n in other for d in dict.fromkeys(p["domain"] for p in recs[n - 1]["filter"]["passages"]))
    L.append(f"домены в пассажах у строк «ошибка» или «чужой товар» ({len(other)}): " + ", ".join(f"{d} {c}" for d, c in dom.most_common(12)))
    L.append("")

    # 12. Сбои модели
    errs = [(n, r["cards"][model].get("error") or "") for n, r in rows if not card(r)]
    kinds = collections.Counter("429 квота одновременных сессий" if "429" in e else e[:60] for _, e in errs)
    L += ["## 12. Сбои генерации", f"строк без карточки: {len(errs)}: " + ", ".join(f"{k} {c}" for k, c in kinds.items()),
          f"номера: {[n for n, _ in errs]}", ""]

    # 13. Провалы проверки кодом: что именно
    iss = collections.Counter()
    for _, r in rows:
        for i in r["cards"][model]["verify"]["issues"]:
            iss[i.split(":")[0]] += 1
    fail_ok = [n for n, r in rows if r["cards"][model]["verify"]["status"] == "fail" and ag[str(n)]["verdict"] == "ок"]
    L += ["## 13. Замечания проверки кодом", ", ".join(f"{k} {c}" for k, c in iss.most_common()),
          f"провал проверки при вердикте агента «ок»: {len(fail_ok)}; примеры {fail_ok[:10]}", ""]

    out = config.ROOT / args.out
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
