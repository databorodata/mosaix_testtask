"""Прототип: страница поиска аналогов и проверка ведомости файлом.

Запуск: uvicorn app.web:app --host 0.0.0.0 --port 8000 (или docker compose up).
Поиск — вариант main из app/searcher.py (обучаемое решение с реранкером); переключатель
«быстрый режим» использует main_light без реранкера. Подтверждения специалиста пишутся
в data/feedback.jsonl и при повторном таком же запросе поднимают подтверждённое наверх,
отклонённое прячут — модель не переобучается, это память поверх поиска.
"""

from __future__ import annotations

import io
import json
import re
import time
import uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import config
from .attrs import fmt
from .ksr import tokens as norm_tokens
from .normalize import clean
from .searcher import Engine

FEEDBACK = config.DATA / "feedback" / "feedback.jsonl"
STATIC = config.ROOT / "static"
EXAMPLES = ["кабель КВВГнг-LS 7*1.5", "камера IP уличная 2 Мп вариофокальная", "ворота откатные 6000х2250",
            "труба гофра ПВХ 25 мм", "ИБП 12В для улицы", "трансформатор силовой сухой 1000 кВА",
            "огнетушитель порошковый ОП-5"]

app = FastAPI(title="Поиск аналогов позиций")
eng: Engine | None = None
_sheets: dict[str, bytes] = {}


def engine() -> Engine:
    global eng
    if eng is None:
        eng = Engine()
    return eng


def qkey(query: str) -> str:
    return " ".join(norm_tokens(clean(query)))


def feedback_for(key: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if FEEDBACK.exists():
        for line in FEEDBACK.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d["key"] == key:
                out[d["row"]] = d  # последний вердикт по строке побеждает
    return out


def hit_json(h, verdict: str | None = None) -> dict:
    return {"row": h.row, "name": h.name, "price": h.price, "score": h.score, "status": h.status,
            "why": h.why, "matched": h.matched, "differs": h.differs, "missing": h.missing,
            "notes": h.notes, "verdict": verdict}


def do_search(query: str, fast: bool = False, k: int = 5) -> dict:
    e = engine()
    res = e.search(query, "main_light" if fast else "main", k=k)
    fb = feedback_for(qkey(query))
    hits, hidden = [], []
    for h in res.hits:
        v = fb.get(h.row, {}).get("verdict")
        (hidden if v == "reject" else hits).append(hit_json(h, v))
    # подтверждённые ранее строки — наверх
    hits.sort(key=lambda x: (x["verdict"] not in ("same", "substitute"), -x["score"]))
    q = res.q
    parsed = {"kind": q.kind or "—", "codes": list(q.codes.values()),
              "attrs": {t: fmt(k2, q) for k2, t in (("section", "сечение"), ("sizes", "размеры"),
                        ("power_kva", "мощность, кВА"), ("power_kw", "мощность, кВт"), ("voltage", "напряжение"),
                        ("current", "ток"), ("resolution", "разрешение"), ("ip", "IP"), ("mass", "масса"))
                        if getattr(q, k2)}}
    return {"query": query, "found": res.found and bool(hits), "reason": res.reason, "parsed": parsed,
            "hits": hits, "hidden": hidden,
            "rejected": [hit_json(h) for h in res.rejected[:3]],
            "hint": res.hint, "mode": "быстрый (без сверяющей модели)" if fast else "основной"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/examples")
def examples() -> list[str]:
    return EXAMPLES


@app.get("/api/search")
def search(q: str, fast: bool = False):
    if not q.strip():
        return JSONResponse({"error": "пустой запрос"}, status_code=400)
    return do_search(q.strip(), fast=fast)


@app.post("/api/feedback")
def feedback(payload: dict):
    row, verdict, query = int(payload["row"]), payload["verdict"], payload["query"]
    assert verdict in ("same", "substitute", "reject")
    rec = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "query": query, "key": qkey(query), "row": row,
           "verdict": verdict}
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}


def _sheet_rows(data: bytes, filename: str) -> list[str]:
    """Читает xlsx или csv и возвращает колонку наименований: по заголовку, иначе самая текстовая."""
    if filename.lower().endswith(".csv"):
        import csv
        text = data.decode("utf-8-sig", errors="replace")
        rows = [r for r in csv.reader(io.StringIO(text), delimiter=";" if ";" in text.splitlines()[0] else ",")]
    else:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
        rows = [[c if c is not None else "" for c in r] for r in wb.active.iter_rows(values_only=True)]
    if not rows:
        return []
    head = [str(c).lower() for c in rows[0]]
    col = next((j for j, c in enumerate(head) if re.search(r"наименован|позици|название", c)), None)
    start = 1
    if col is None:
        start = 0
        widths = [sum(len(str(r[j])) for r in rows[:50] if j < len(r) and not str(r[j]).replace(".", "").isdigit())
                  for j in range(max(len(r) for r in rows))]
        col = max(range(len(widths)), key=lambda j: widths[j])
    out = []
    for r in rows[start:]:
        val = str(r[col]).strip() if col < len(r) else ""
        out.append(val)
    return out[:2000]


@app.post("/api/sheet")
async def sheet(file: UploadFile = File(...), fast: bool = True):
    names = _sheet_rows(await file.read(), file.filename or "sheet.xlsx")
    results = []
    for n, name in enumerate(names, 1):
        if not name or len(name) < 3:
            results.append({"n": n, "name": name, "status": "пустая строка", "top": None})
            continue
        r = do_search(name, fast=fast, k=1)
        top = r["hits"][0] if r["hits"] else None
        results.append({"n": n, "name": name,
                        "status": (top["status"] if r["found"] else "не найдено"),
                        "reason": r["reason"], "top": top,
                        "hint": r["hint"][:1] if not r["found"] else []})
    token = uuid.uuid4().hex[:12]
    _sheets[token] = _sheet_xlsx(results)
    found = sum(1 for x in results if x["top"] and x["status"] != "не найдено")
    return {"rows": results, "token": token, "total": len(results), "found": found}


def _sheet_xlsx(results: list[dict]) -> bytes:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Проверка ведомости"
    ws.append(["№", "Строка ведомости", "Статус", "Найденная позиция", "Цена", "Оценка", "Чем отличается"])
    for x in results:
        t = x.get("top") or {}
        ws.append([x["n"], x["name"], x["status"], t.get("name", ""), t.get("price", ""), t.get("score", ""),
                   "; ".join(t.get("differs", []) + t.get("missing", []))[:300]])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@app.get("/api/sheet/download/{token}")
def sheet_download(token: str):
    data = _sheets.get(token)
    if not data:
        return JSONResponse({"error": "результат устарел, загрузите файл заново"}, status_code=404)
    path = config.DATA / "cache" / f"sheet_{token}.xlsx"
    path.write_bytes(data)
    return FileResponse(path, filename="проверка_ведомости.xlsx",
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
