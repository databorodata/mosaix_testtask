"""Обучаемое решение «подходит или нет»: признаки кандидата → логистическая регрессия.

Модель обучается скриптом scripts/train_decision.py на наборах настройки (dev, control, big)
и сохраняется в data/model/decision.json: имена признаков, нормировка, коэффициенты и порог
«не найдено». В поиске вероятность считается по сохранённым числам, scikit-learn в онлайне
не нужен. Жёсткими остаются только правила специалиста из rules/rules.yaml: их нарушение
отбрасывает кандидата до модели. Разбор характеристик здесь — признаки и объяснение, не фильтр.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import config
from .attrs import Attrs, code_relation, kind_match, parse
from .ksr import tokens as norm_tokens
from .normalize import clean

MODEL_FILE = config.DATA / "model" / "decision.json"
POOL = 30

FEATS = ["rrf", "log_rank", "cos_all", "cos_name", "bm_name", "bm_card", "kind", "n_match", "n_diff",
         "n_missing", "code_match", "code_near", "code_none", "rerank"]
LIGHT_FEATS = FEATS[:-1]  # без оценки реранкера

# Человеческие названия признаков для строки «из чего сложилась оценка».
PHRASES = {
    "rrf": "высоко сразу в нескольких каналах поиска",
    "log_rank": "место в объединённом списке",
    "cos_all": "близость по смыслу",
    "cos_name": "близость по смыслу к наименованию",
    "bm_name": "совпадение слов наименования",
    "bm_card": "совпадение слов карточки",
    "kind": "вид изделия",
    "n_match": "совпавшие характеристики",
    "n_diff": "отличия в характеристиках",
    "n_missing": "не указанные у кандидата характеристики",
    "code_match": "марка совпала",
    "code_near": "марка похожа",
    "code_none": "марки из запроса нет",
    "rerank": "оценка сверяющей модели",
}

# Запрос об услуге ищет только среди услуг; обычный запрос услуг и надбавок не видит.
SERVICE_KINDS = {"перевозка", "погрузка", "разгрузка", "транспортировка", "доставка"}
ADJ_QUERY = re.compile(r"^\s*(надбавк|доплат|наценк)", re.I)
KIT_EMPTY = re.compile(r"в\s+составе\s*:?\s*$", re.I)


@dataclass
class Cand:
    i: int
    feats: list[float]
    status: str          # статус из judge
    hit: object          # Hit из judge: matched, differs, missing, notes
    rel: str             # отношение кодов: match / near / none / ""
    rule_fail: bool      # нарушено жёсткое правило специалиста
    channels: dict[str, int] = field(default_factory=dict)
    cos: float = 0.0
    prob: float = 0.0


def query_type(q: Attrs, query: str) -> str:
    """Тип запроса: service, adjustment или обычный (пустая строка)."""
    if ADJ_QUERY.search(query):
        return "adjustment"
    if q.kind in SERVICE_KINDS:
        return "service"
    return ""


def build(eng, query: str, use_rerank: bool = True, allowed=None) -> tuple[Attrs, list[Cand]]:
    """Признаки кандидатов: так же, как при обучении. allowed — фильтр строк по типу."""
    q = parse(query)
    bn = dict(eng.bm_norm.search(norm_tokens(clean(query)), 100))
    bc = dict(eng.bm_card.search(norm_tokens(query), 100))
    qv = eng.qvec(query)
    fields = list(eng.emb)
    cos_all, cos_name = eng.cos(qv, fields), eng.cos(qv, ["name"])
    lists = {"names": sorted(bn, key=lambda i: -bn[i]), "cards": sorted(bc, key=lambda i: -bc[i]),
             "dense": [int(i) for i in np.argsort(-cos_all)[:100]]}
    fused: dict[int, float] = {}
    channels: dict[int, dict[str, int]] = {}
    for ch, lst in lists.items():
        for r, i in enumerate(lst):
            fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r + 1)
            channels.setdefault(i, {})[ch] = r + 1
    items = [(i, s) for i, s in sorted(fused.items(), key=lambda x: -x[1]) if allowed is None or allowed(i)]
    pool = [i for i, _ in items[:POOL]]
    if not pool:
        return q, []
    top = fused[pool[0]]
    rr = eng.rerank_scores(query, pool) if use_rerank else {i: 0.0 for i in pool}
    mbn, mbc = max(bn.values(), default=1), max(bc.values(), default=1)
    out: list[Cand] = []
    for rank, i in enumerate(pool, 1):
        c = eng.attrs_plain[i]
        cand_nouns = c.heads + eng.extra_nouns[i]
        _, status, h = eng.judge(q, c, cand_nouns, soft_code=True)
        # Жёсткое правило специалиста: judge вернул «не подходит» не из-за вида изделия.
        rule_fail = status == "не подходит" and not (h.differs and h.differs[0].startswith("другой вид"))
        rel, _ = code_relation(q, c)
        n_diff = sum(1 for d in h.differs if not d.startswith("другой вид") and "строго" not in d)
        feats = [fused[i] / top, math.log(rank), float(cos_all[i]), float(cos_name[i]),
                 bn.get(i, 0) / mbn, bc.get(i, 0) / mbc, float(kind_match(q.kind, cand_nouns)),
                 float(len(h.matched)), float(n_diff), float(len(h.missing)),
                 float(rel == "match"), float(rel == "near"), float(rel == "none"), rr[i]]
        out.append(Cand(i, feats, status, h, rel, rule_fail, channels.get(i, {}), float(cos_all[i])))
    return q, out


def load_model() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"нет модели решения {MODEL_FILE}; запустите python -m scripts.train_decision")
    return json.loads(MODEL_FILE.read_text(encoding="utf-8"))


def predict(sub: dict, feats: list[list[float]]) -> list[float]:
    """Вероятность «подходит» по сохранённым коэффициентам (та же формула, что у scikit-learn)."""
    cols = [FEATS.index(f) for f in sub["feats"]]
    out = []
    for x in feats:
        z = sub["intercept"]
        for j, cidx in enumerate(cols):
            z += sub["coef"][j] * (x[cidx] - sub["mean"][j]) / sub["scale"][j]
        out.append(1.0 / (1.0 + math.exp(-z)))
    return out


def contributions(sub: dict, x: list[float], top: int = 3) -> list[str]:
    """Главные слагаемые оценки словами: «+ близость по смыслу», «− отличия в характеристиках»."""
    cols = [FEATS.index(f) for f in sub["feats"]]
    parts = []
    for j, cidx in enumerate(cols):
        c = sub["coef"][j] * (x[cidx] - sub["mean"][j]) / sub["scale"][j]
        parts.append((abs(c), c, sub["feats"][j]))
    parts.sort(key=lambda t: -t[0])
    out = []
    for _, c, name in parts[:top]:
        if abs(c) < 1e-6:
            continue
        out.append(("+ " if c > 0 else "− ") + PHRASES.get(name, name))
    return out
