"""Поиск аналогов по базе.

Шаги на каждый запрос: разбор запроса кодом; кандидаты по словам (названия и карточки), по смыслу
(векторы) и по ключу КСР; слияние по местам (RRF); сравнение характеристик и правила специалиста
из rules/rules.yaml; порог «не найдено». Модель в онлайне не вызывается, кроме кодировщика векторов.

Варианты (лестница для замера):
  words_raw    обычный поиск по словам: BM25 по сырым наименованиям, всегда до пяти строк
  words_norm   + нормализация записи: чисел, «х» и «*», латиницы и кириллицы, окончаний
  cards        + карточки: BM25 по названию, синонимам и характеристикам карточки
  vectors      + векторы по четырём полям строки
  rules        + характеристики, правила, вид изделия и порог «не найдено»
  rules_soft   как rules, но несовпадение кода опускает кандидата ниже, а не отбрасывает
  rerank_plain реранкер вместо проверок: порядок по оценке реранкера, без «не найдено»
  rerank       реранкер вместо ручных весов, проверки rules_soft остаются для «не найдено» и пояснений
  rules_nocard то же, что rules, но без карточек: слова и векторы только по наименованию
  main         итог: обучаемое решение (data/model/decision.json) с реранкером; статусы «та же позиция»,
               «допустимая замена», «похожая»; услуги и надбавки отдельно; жёсткие правила специалиста
  main_light   то же без реранкера (облегчённый режим)

КСР в поиске кандидатов не участвует (замер 17.10: не дал ни одного нового попадания). Он остаётся слоем
подготовки и подсказкой при «не найдено»: «в закупках не встречалось, в КСР есть …».
"""

from __future__ import annotations

import collections
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import config
from .attrs import Attrs, code_relation, fmt, heads, kind_match, parse
from . import decision
from .ksr import Index as KsrIndex
from .ksr import tokens as norm_tokens


def norm_and_query(query: str) -> str:
    return clean(query)
from .normalize import clean

VARIANTS = ["words_raw", "words_norm", "cards", "vectors", "rules", "rules_soft", "rerank_plain", "rerank", "rules_nocard", "main", "main_light"]
RERANKER = "BAAI/bge-reranker-v2-m3"
RERANK_TOP = 30
FIELDS = ["name", "canonical", "synonyms", "description"]
MODELS = {
    "e5-base": {"hf": "intfloat/multilingual-e5-base", "query": "query: ", "doc": "passage: "},
    "user2-base": {"hf": "deepvk/USER2-base", "query": "search_query: ", "doc": "search_document: "},
}
INDEX_DIR = config.DATA / "index"
RULES_FILE = config.ROOT / "rules" / "rules.yaml"
RRF_K = 60
POOL = 60
TAU = 0.80  # порог сходства по смыслу для «найдено»; подбирается на наборе настройки, не на контрольном

_RAW = re.compile(r"\w+")


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.dl = [len(d) or 1 for d in docs]
        self.avgdl = sum(self.dl) / max(len(docs), 1)
        self.inv: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for i, d in enumerate(docs):
            for w, c in collections.Counter(d).items():
                self.inv[w].append((i, c))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - len(lst) + 0.5) / (len(lst) + 0.5)) for w, lst in self.inv.items()}

    def search(self, q: list[str], k: int = 100) -> list[tuple[int, float]]:
        scores: dict[int, float] = collections.defaultdict(float)
        for w in set(q):
            for i, c in self.inv.get(w, ()):
                scores[i] += self.idf[w] * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


def raw_tokens(text: str) -> list[str]:
    return _RAW.findall((text or "").lower())


def card_text(card: dict) -> str:
    if not card:
        return ""
    attrs = " ".join(f"{a.get('name', '')} {a.get('value', '')}" for a in card.get("attributes", []))
    return " ".join([card.get("canonical_name", ""), " ".join(card.get("synonyms", [])), attrs])


@dataclass
class Hit:
    i: int
    row: int
    name: str
    price: object
    score: float
    cos: float
    status: str = ""
    matched: list[str] = field(default_factory=list)
    differs: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    channels: dict[str, int] = field(default_factory=dict)
    why: list[str] = field(default_factory=list)  # из чего сложилась оценка (вариант main)


@dataclass
class Result:
    query: str
    variant: str
    found: bool
    hits: list[Hit]
    rejected: list[Hit]
    q: Attrs
    reason: str = ""
    hint: list[dict] = field(default_factory=list)


class Engine:
    def __init__(self, prep_path: Path | None = None, model: str = "e5-base", rules_path: Path | None = None,
                 with_ksr: bool = True, tau: float = TAU):
        if prep_path is None:
            # prep_runtime.jsonl — то же самое без текстов веб-выдачи (их не публикуем); поля поиска совпадают
            runtime = config.DATA / "full" / "prep_runtime.jsonl"
            prep_path = runtime if runtime.exists() else config.DATA / "full" / "prep_all.jsonl"
        recs = [json.loads(x) for x in prep_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        mkey = list(recs[0]["cards"].keys())[0]
        self.rows = []
        for i, r in enumerate(recs):
            card = r["cards"][mkey].get("card") or {}
            special = "service" if r.get("ntype") == "service" else (
                "adjustment" if decision.ADJ_QUERY.search(r["name"]) else "")
            self.rows.append({"i": i, "row": r["row"], "name": r["name"], "clean": clean(r["name"]), "price": r.get("price"),
                              "card": card, "ksr": r.get("ksr") or {}, "ntype": r.get("ntype"), "special": special})
        self.tau = tau
        self.model_key = model
        self.bm_raw = BM25([raw_tokens(x["name"]) for x in self.rows])
        self.bm_norm = BM25([norm_tokens(x["clean"]) for x in self.rows])
        self.bm_card = BM25([norm_tokens(card_text(x["card"])) for x in self.rows])
        self.attrs_plain = [parse(x["clean"]) for x in self.rows]
        self.extra_nouns = [heads(x["card"].get("canonical_name", ""))[:3] if x["card"] else [] for x in self.rows]
        self.rules = yaml.safe_load((rules_path or RULES_FILE).read_text(encoding="utf-8")).get("rules", [])
        self.emb = self._load_emb(model)
        self._encoder = None
        self._reranker = None
        self.ksr_index = KsrIndex.load() if with_ksr else None
        self._decision = None  # модель решения, data/model/decision.json; загружается при первом вызове main
        self.by_ksr: dict[str, list[int]] = collections.defaultdict(list)
        for x in self.rows:
            if x["ksr"].get("code"):
                self.by_ksr[x["ksr"]["code"]].append(x["i"])

    # ---------- векторы ----------
    def _load_emb(self, model: str) -> dict[str, np.ndarray]:
        d = INDEX_DIR / model
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"нет векторного индекса {d}; запустите python -m scripts.build_index --model {model}")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["rows"] != [x["row"] for x in self.rows]:
            raise ValueError("векторный индекс собран для другой базы; пересоберите build_index")
        return {f: np.load(d / f"{f}.npy").astype(np.float32) for f in FIELDS}

    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(MODELS[self.model_key]["hf"])
        return self._encoder

    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER, max_length=256)
        return self._reranker

    def rerank_scores(self, query: str, idx: list[int]) -> dict[int, float]:
        scores = self.reranker().predict([(query, self.rows[i]["clean"]) for i in idx], show_progress_bar=False)
        return {i: float(sc) for i, sc in zip(idx, scores)}

    def qvec(self, query: str) -> np.ndarray:
        cfg = MODELS[self.model_key]
        return self.encoder().encode([cfg["query"] + query], normalize_embeddings=True)[0].astype(np.float32)

    def cos(self, qv: np.ndarray, fields: list[str]) -> np.ndarray:
        return np.max(np.stack([self.emb[f] @ qv for f in fields]), axis=0)

    # ---------- КСР: подсказка при «не найдено» ----------
    def ksr_hint(self, query: str, k: int = 3) -> list[dict]:
        if not self.ksr_index:
            return []
        out = []
        for pi, _ in self.ksr_index.search(norm_and_query(query), k=k):
            p = self.ksr_index.pos[pi]
            out.append({"code": p["code"], "name": p["name"], "unit": p.get("unit")})
        return out

    def ksr_channel(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Строки базы с тем же кодом КСР. В поиске не используется, оставлено для замеров."""
        if not self.ksr_index:
            return []
        out, seen = [], set()
        for rank, (pi, _) in enumerate(self.ksr_index.search(query, k=k)):
            for i in self.by_ksr.get(self.ksr_index.pos[pi]["code"], []):
                if i not in seen:
                    seen.add(i)
                    out.append((i, 1.0 / (rank + 1)))
        return out

    # ---------- сравнение характеристик ----------
    def _rules_for(self, kind: str) -> list[dict]:
        out = []
        for r in self.rules:
            if any(kind == k or (len(kind) >= 4 and (k in kind or kind in k)) for k in r.get("kinds", [])):
                out.append(r)
        return out

    @staticmethod
    def _step_distance(a: float, b: float, steps: list[float]) -> int:
        def idx(v: float) -> int:
            return min(range(len(steps)), key=lambda j: abs(steps[j] - v))
        return abs(idx(a) - idx(b))

    def judge(self, q: Attrs, c: Attrs, cand_nouns: list[str], soft_code: bool = False) -> tuple[float, str, Hit]:
        h = Hit(0, 0, "", None, 0.0, 0.0)
        bonus = 0.0
        hard = 0
        if not kind_match(q.kind, cand_nouns):
            h.differs.append(f"другой вид изделия: в запросе «{q.kind}»")
            return -9.0, "не подходит", h
        rules = {r["attribute"]: r for r in self._rules_for(q.kind)}
        fails: list[str] = []

        def num_attr(key: str, title: str, w_match: float, w_diff: float) -> None:
            nonlocal bonus, hard
            qv, cv = getattr(q, key), getattr(c, key)
            if not qv:
                return
            rule = rules.get(key)
            if not cv:
                h.missing.append(title)
                if rule:
                    fails.append(f"{rule['title']}: у кандидата не указано")
                return
            if rule and rule.get("mode") == "steps":
                dist = min(self._step_distance(a, b, rule["steps"]) for a in qv for b in cv)
                if dist == 0:
                    h.matched.append(f"{title}: {fmt(key, c)}")
                    bonus += w_match
                elif dist <= rule.get("max_steps", 0):
                    h.notes.append(f"{title}: {fmt(key, c)} вместо {fmt(key, q)}, допустимо по правилу «{rule['title']}»")
                else:
                    fails.append(f"{rule['title']}: {fmt(key, c)} вместо {fmt(key, q)}")
                return
            if qv & cv:
                h.matched.append(f"{title}: {fmt(key, c)}")
                bonus += w_match
            else:
                h.differs.append(f"{title}: {fmt(key, c)} вместо {fmt(key, q)}")
                bonus -= w_diff
                hard += 1
                if rule and rule.get("mode") == "strict":
                    fails.append(f"{rule['title']}: {fmt(key, c)} вместо {fmt(key, q)}")

        num_attr("section", "сечение", 0.3, 0.4)
        if q.sizes:
            if not c.sizes:
                h.missing.append("размеры")
                bonus -= 0.05
            else:
                miss = [x for x in q.sizes if not any(abs(x - y) <= 0.011 * max(1.0, x) for y in c.sizes)]
                if not miss:
                    h.matched.append(f"размеры: {fmt('sizes', c)}")
                    bonus += 0.3
                else:
                    h.differs.append(f"размеры: у кандидата {fmt('sizes', c)}, в запросе {fmt('sizes', q)}")
                    bonus -= 0.3
                    hard += 1
        for key, title in (("power_kva", "мощность"), ("power_kw", "мощность"), ("voltage", "напряжение"),
                           ("current", "ток"), ("resolution", "разрешение"), ("ip", "класс защиты"), ("mass", "масса")):
            num_attr(key, title, 0.15, 0.25)
        rel, raw = code_relation(q, c)
        if rel == "match":
            h.matched.append(f"код: {raw}")
            bonus += 0.4
        elif rel == "near":
            h.notes.append(f"код отличается: {raw} при запросе {fmt('codes', q)}")
            bonus -= 0.05
        elif rel == "none":
            bonus -= 0.2
            if soft_code:
                h.notes.append(f"кода из запроса {fmt('codes', q)} у кандидата нет")
            else:
                h.differs.append(f"код из запроса {fmt('codes', q)} не найден")
                hard += 1
        if fails:
            h.differs = fails + h.differs
            return bonus - 1.0, "не подходит", h
        if hard:
            return bonus - 0.5 * hard, "с отличием", h
        return bonus, ("характеристики совпали" if h.matched and not h.missing and not h.notes else "похожая"), h

    # ---------- поиск ----------
    def search(self, query: str, variant: str = "rules", k: int = 5) -> Result:
        q = parse(query)
        if variant in ("main", "main_light"):
            return self._main(query, q, k, use_rerank=variant == "main", variant=variant)
        if variant == "words_raw":
            return self._plain(query, variant, q, self.bm_raw.search(raw_tokens(query), k))
        if variant == "words_norm":
            return self._plain(query, variant, q, self.bm_norm.search(norm_tokens(clean(query)), k))
        use_cards = variant != "rules_nocard"
        lists: dict[str, list[tuple[int, float]]] = {"names": self.bm_norm.search(norm_tokens(clean(query)), 100)}
        if use_cards:
            lists["cards"] = self.bm_card.search(norm_tokens(query), 100)
        qv = self.qvec(query)
        cos = self.cos(qv, FIELDS if use_cards else ["name"])
        if variant != "cards":
            order = np.argsort(-cos)[:100]
            lists["dense"] = [(int(i), float(cos[i])) for i in order]
        fused: dict[int, float] = collections.defaultdict(float)
        ranks: dict[int, dict[str, int]] = collections.defaultdict(dict)
        for ch, lst in lists.items():
            for rank, (i, _) in enumerate(lst):
                fused[i] += 1.0 / (RRF_K + rank + 1)
                ranks[i][ch] = rank + 1
        pool = sorted(fused.items(), key=lambda x: -x[1])[:POOL]
        if not pool:
            return Result(query, variant, False, [], [], q, "нет кандидатов")
        top = pool[0][1]
        if variant in ("cards", "vectors"):
            hits = [self._hit(i, s / top, float(cos[i]), ranks[i]) for i, s in pool[:k]]
            return Result(query, variant, bool(hits), hits, [], q)
        rr: dict[int, float] = {}
        if variant in ("rerank_plain", "rerank"):
            pool = pool[:RERANK_TOP]
            rr = self.rerank_scores(query, [i for i, _ in pool])
        if variant == "rerank_plain":
            order = sorted(pool, key=lambda x: -rr[x[0]])[:k]
            hits = [self._hit(i, rr[i], float(cos[i]), ranks[i]) for i, _ in order]
            return Result(query, variant, bool(hits), hits, [], q)
        soft = variant in ("rules_soft", "rerank")
        judged: list[Hit] = []
        rejected: list[Hit] = []
        for i, s in pool:
            cand_nouns = self.attrs_plain[i].heads + (self.extra_nouns[i] if use_cards else [])
            bonus, status, h = self.judge(q, self.attrs_plain[i], cand_nouns, soft_code=soft)
            score = rr[i] if rr else s / top + bonus
            hit = self._hit(i, score, float(cos[i]), ranks[i])
            hit.status, hit.matched, hit.differs, hit.missing, hit.notes = status, h.matched, h.differs, h.missing, h.notes
            if status == "не подходит":
                rejected.append(hit)
            elif hit.cos < self.tau:
                hit.notes.append(f"сходство по смыслу {hit.cos:.2f} ниже порога {self.tau:.2f}")
                rejected.append(hit)
            else:
                judged.append(hit)
        ok_status = ("характеристики совпали", "похожая")
        if rr:  # реранкер вместо ручных весов: сначала кандидаты без отличий, внутри по оценке реранкера
            judged.sort(key=lambda h: (h.status not in ok_status, -h.score))
        else:
            judged.sort(key=lambda h: -h.score)
        rejected.sort(key=lambda h: -h.score)
        clean_hits = [h for h in judged if h.status in ok_status]
        if not clean_hits:
            reason = "нет кандидатов без отличий по названным характеристикам"
            return Result(query, variant, False, [], judged[:k] + rejected[: max(0, k - len(judged))], q, reason,
                          hint=self.ksr_hint(query))
        return Result(query, variant, True, judged[:k], rejected[:k], q)


    # ---------- итоговый вариант: обучаемое решение ----------
    DISPLAY_ORDER = {"та же позиция": 0, "допустимая замена": 1, "похожая": 2, "с отличием": 3}

    def _display_status(self, q: Attrs, h: Hit, rel: str) -> str:
        """Статус для показа. «Та же» — всё названное в запросе подтверждено; «замена» — отличие
        в пределах правила специалиста или другая марка при совпавших характеристиках."""
        if h.differs:
            return "с отличием"
        by_rule = any("допустимо по правилу" in n for n in h.notes)
        if not h.missing and not by_rule and (rel == "match" or not q.codes):
            return "та же позиция"
        if not h.missing and (by_rule or (q.codes and rel != "match")):
            return "допустимая замена"
        return "похожая"

    def _main(self, query: str, q: Attrs, k: int, use_rerank: bool, variant: str) -> Result:
        if self._decision is None:
            self._decision = decision.load_model()
        sub = self._decision["models"]["rerank" if use_rerank else "light"]
        if not q.kind and not q.codes:
            return Result(query, variant, False, [], [], q,
                          "в запросе нет ни вида изделия, ни марки — недостаточно данных")
        qtype = decision.query_type(q, query)  # услуга, надбавка или обычный запрос
        allowed = (lambda i: self.rows[i]["special"] == qtype) if qtype else (lambda i: not self.rows[i]["special"])
        q2, cands = decision.build(self, query, use_rerank=use_rerank, allowed=allowed)
        if not cands:
            return Result(query, variant, False, [], [], q, "нет кандидатов", hint=self.ksr_hint(query))
        probs = decision.predict(sub, [c.feats for c in cands])
        t = sub["threshold"]
        hits: list[Hit] = []
        rejected: list[Hit] = []
        for c, p in zip(cands, probs):
            hit = self._hit(c.i, round(p, 4), c.cos, c.channels)
            h = c.hit
            hit.matched, hit.differs, hit.missing, hit.notes = h.matched, h.differs, h.missing, list(h.notes)
            if decision.KIT_EMPTY.search(self.rows[c.i]["name"]):
                hit.notes.append("состав в строке не расписан")
            if c.rule_fail:
                hit.status = "не подходит"
                rejected.append(hit)
                continue
            if p < t:
                hit.status = "ниже порога"
                hit.notes.append(f"оценка {p:.2f} ниже порога {t:.2f}")
                rejected.append(hit)
                continue
            hit.status = self._display_status(q, h, c.rel)
            hit.why = decision.contributions(sub, c.feats)
            hits.append(hit)
        # Порядок и «найдено/не найдено» ровно те же, что при обучении и проверке модели:
        # сортировка по вероятности, найдено = хоть один кандидат выше порога. Статус — подпись, не фильтр.
        hits.sort(key=lambda h: -h.score)
        rejected.sort(key=lambda h: -h.score)
        if not hits:
            return Result(query, variant, False, [], rejected[:k], q,
                          "ни один кандидат не прошёл порог решения", hint=self.ksr_hint(query))
        return Result(query, variant, True, hits[:k], rejected[:k], q)

    def _plain(self, query: str, variant: str, q: Attrs, lst: list[tuple[int, float]]) -> Result:
        hits = [self._hit(i, s, 0.0, {}) for i, s in lst]
        return Result(query, variant, bool(hits), hits, [], q, "" if hits else "нет совпадений по словам")

    def _hit(self, i: int, score: float, cos: float, ranks: dict[str, int]) -> Hit:
        x = self.rows[i]
        return Hit(i, x["row"], x["name"], x["price"], round(score, 4), round(cos, 4), channels=dict(ranks))
