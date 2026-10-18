"""Файл ручной проверки карточек в Markdown: сводная таблица и раздел на каждую строку базы.

Запуск: python3 -m scripts.make_review_md --prep data/debug/prep_50.jsonl --out data/debug/review_50.md
"""

from __future__ import annotations

import argparse
import json

from app import config
from app.nametype import NAMES

MODEL_TITLE = {"qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "yandexgpt-5.1": "YandexGPT 5.1"}
KSR_TITLE = {"code": "тот же ресурс", "group": "та же группа", "none": "нет в КСР", "service": "услуга", "error": "ошибка"}


def esc(s: object) -> str:
    return str(s or "").replace("|", "/").replace("\n", " ").strip()


def card_block(label: str, res: dict) -> list[str]:
    v = res["verify"]
    issues = "; ".join(v["issues"])
    out = [f"**{label} · {MODEL_TITLE.get(res['model'], res['model'])}** · проверка кодом: {v['status']}" + (f" ({issues})" if issues else "")]
    card = res.get("card")
    if not card:
        return out + [f"- карточки нет: {res.get('error') or ''}", ""]
    attrs = "; ".join(f"{a['name']}: {a['value']} {a['unit']}".strip() for a in card.get("attributes", []))
    out += [
        f"- Название: {card.get('canonical_name', '')}",
        f"- Описание: {card.get('description', '')}",
        f"- Синонимы: {'; '.join(card.get('synonyms', []))}",
        f"- Характеристики: {attrs}",
        f"- Производитель и модель: {card.get('manufacturer', '') or 'нет'} / {card.get('model_code', '') or 'нет'} · уверенность: {card.get('confidence', '')} · фрагменты: {card.get('sources_used', [])}",
        "",
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdicts", default="")
    args = ap.parse_args()
    verdicts = json.loads((config.ROOT / args.verdicts).read_text(encoding="utf-8")) if args.verdicts else {}
    recs = [json.loads(line) for line in (config.ROOT / args.prep).read_text(encoding="utf-8").splitlines() if line.strip()]
    models = list(recs[0]["cards"].keys())

    lines = [
        "# Проверка карточек: 50 строк дневного прогона",
        "",
        "Как оценивать: «ок» если карточка верна, числа и коды на месте, лишних фактов нет; «частично» если в целом верно, но потеряно важное или синонимы слабые; «ошибка» если неверное число, чужая модель или размер, выдуманный факт, неверный тип изделия. Привязку к КСР оценивайте отдельно: верен ли ресурс или группа.",
        "",
        "Проверка кодом: ok без замечаний; warn если не всё перенесено или описание короткое; fail если число не из входа или потерян код из наименования.",
        "",
        "Строка «Предварительная оценка Claude» это моё прочтение карточки A (Qwen); её достаточно подтвердить или поправить.",
        "",
        "## Сводка",
        "",
        "| № | Строка базы | Тип наименования | Контекст | A | B | КСР | Оценка Claude |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, r in enumerate(recs, 1):
        ks = r["ksr"]
        lines.append(f"| {k} | {esc(r['name'])[:90]} | {NAMES[r['ntype']]} | {r['filter']['kind']} | "
                     f"{r['cards'][models[0]]['verify']['status']} | "
                     f"{r['cards'][models[1]]['verify']['status'] if len(models) > 1 else ''} | {KSR_TITLE.get(ks['status'], ks['status'])} | {verdicts.get(str(k), {}).get('A', '')} |")
    lines.append("")

    for k, r in enumerate(recs, 1):
        f = r["filter"]
        domains = ", ".join(sorted({p["domain"] for p in f["passages"]}))
        lines += [
            f"## {k}. {esc(r['name'])}",
            "",
            f"- Строка Excel {r['row']} · цена {r.get('price')} · {'пример из задания' if r.get('anchor') else 'выборка'}",
            f"- Тип наименования: {NAMES[r['ntype']]}",
            f"- Контекст выдачи: {f['kind']}, пассажей {len(f['passages'])}; домены: {domains or 'нет'}"
            + (f"; отброшены как соседние артикулы: {', '.join(f['neighbor_docs'])}" if f["neighbor_docs"] else ""),
            "- Первые фрагменты:",
        ]
        for p in f["passages"][:3]:
            lines.append(f"  - {p['domain']}: {esc(p['text'])[:240]}")
        lines.append("")
        lines += card_block("A", r["cards"][models[0]])
        if len(models) > 1:
            lines += card_block("B", r["cards"][models[1]])
        ks = r["ksr"]
        ksr_line = f"**КСР** · {KSR_TITLE.get(ks['status'], ks['status'])}"
        if ks.get("name"):
            ksr_line += f" · {ks.get('code') or 'без кода'} «{ks['name']}»" + (f" · {ks['unit']}" if ks.get("unit") else "")
        lines += [ksr_line, f"- Почему: {ks.get('reason', '')}", ""]
        v = verdicts.get(str(k))
        if v:
            lines.append(f"Предварительная оценка Claude · A: {v['A']}" + (f" ({v['why']})" if v["why"] else "") + (f" · КСР: {v['ksr']}" if v["ksr"] else ""))
            lines.append("")
        lines += ["Оценка A: … · Оценка B: … · КСР: … · Комментарий: …", "", "---", ""]

    out = config.ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"файл проверки: {out.relative_to(config.ROOT)}, строк {len(lines)}")


if __name__ == "__main__":
    main()
