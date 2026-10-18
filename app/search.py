"""Yandex Search API: синхронный запрос с кэшем ответа на диске. Ответ хранится как XML с датой."""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from . import config

CACHE_DIR = config.CACHE / "search"
_LOCK = threading.Lock()
_NEXT_SLOT = [0.0]
MIN_INTERVAL_S = 1.1  # квота синхронного режима: 1 запрос в секунду


def cache_key(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse(raw: bytes) -> list[dict]:
    root = ET.fromstring(raw)
    err = root.find(".//error")
    if err is not None:
        raise RuntimeError(f"Yandex Search API вернул ошибку: {_text(err)}")
    docs = []
    for d in root.findall(".//doc"):
        docs.append({
            "url": _text(d.find("url")),
            "domain": _text(d.find("domain")),
            "title": _text(d.find("title")),
            "headline": _text(d.find("headline")),
            "passages": [_text(p) for p in d.findall(".//passage")],
        })
    return docs


def _post(body: dict) -> bytes:
    data = json.dumps(body).encode("utf-8")
    headers = {"Authorization": f"Api-Key {config.YC_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(4):
        req = urllib.request.Request(config.SEARCH_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return base64.b64decode(json.loads(resp.read())["rawData"])
        except urllib.error.HTTPError as e:
            text = e.read()[:300].decode("utf-8", "ignore")
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"поиск: HTTP {e.code}: {text}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("поиск: исчерпаны попытки")


def search(query: str, *, groups: int = 10, passages: int = 4) -> tuple[list[dict], bool]:
    """Возвращает документы выдачи и признак «взято из кэша»."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(query)
    path = CACHE_DIR / f"{key}.xml"
    if path.exists():
        return parse(path.read_bytes()), True
    body = {
        "query": {"searchType": "SEARCH_TYPE_RU", "queryText": query, "familyMode": "FAMILY_MODE_NONE",
                  "page": "0", "fixTypoMode": "FIX_TYPO_MODE_ON"},
        "groupSpec": {"groupMode": "GROUP_MODE_FLAT", "groupsOnPage": str(groups), "docsInGroup": "1"},
        "maxPassages": str(passages), "region": "225", "l10N": "LOCALIZATION_RU",
        "folderId": config.YC_FOLDER_ID, "responseFormat": "FORMAT_XML",
    }
    with _LOCK:  # запросы стартуют не чаще раза в MIN_INTERVAL_S, но идут параллельно
        now = time.monotonic()
        slot = max(now, _NEXT_SLOT[0])
        _NEXT_SLOT[0] = slot + MIN_INTERVAL_S
    if slot > now:
        time.sleep(slot - now)
    raw = _post(body)
    path.write_bytes(raw)
    meta = {"query": query, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": "sync"}
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return parse(raw), False
