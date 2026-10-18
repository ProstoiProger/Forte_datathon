"""Client of the regulation search service (`rag serve` in ../rag): hybrid retrieval over Kazakhstan banking acts.

Used for unusual situations that no registry type covers. Norms found this way carry source="rag": they were
found by search and need a human check. The service is optional: every failure is logged and yields no norms.
"""

import http.client
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from cashwatch.analyzers.facts.schema import FactWindow, Norm

log = logging.getLogger("cashwatch.regsearch")

MAX_QUERY_CHARS = 2000
MAX_K = 10
QUOTE_CHARS = 800

# Regulation wording (Rules No. 231 and related acts) so that BM25 matches the norm's words, not the VLM's.
FLAG_PHRASES_RU: dict[str, str] = {
    "cashier_at_workplace": "кассовый работник находится на рабочем месте",
    "cashier_leaving": "кассовый работник покидает рабочее место",
    "cash_on_desk": "наличные деньги лежат на столе",
    "cash_in_hands": "кассовый работник держит наличные деньги в руках",
    "safe_open": "сейф открыт",
    "door_open": "дверь кассы открыта",
    "phone_in_hand": "кассовый работник пользуется мобильным телефоном (средством связи)",
    "drinking": "кассовый работник употребляет напитки на рабочем месте",
    "personal_items_on_desk": "личные вещи (сумка, верхняя одежда) на рабочем месте кассового работника",
    "collection_bag": "в помещении инкассаторская сумка",
    "visitor_present": "в помещении присутствуют посторонние лица",
}
CASHIER_ABSENT_RU = "кассовый работник отсутствует на рабочем месте"
CASH_UNATTENDED_RU = "наличные деньги оставлены на столе без присмотра"


class RegSearchError(Exception):
    pass


class RegSearchClient:
    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, k: int = 3) -> list[Norm]:
        query = query.strip()[:MAX_QUERY_CHARS]
        if not query:
            return []
        try:
            data = self._call("/search", {"query": query, "k": max(1, min(k, MAX_K))})
        except RegSearchError as exc:
            log.warning("regulation search failed at %s: %s", self.url, exc)
            return []
        hits = data.get("hits") if isinstance(data, dict) else None
        if not isinstance(hits, list):
            log.warning("regulation search at %s: unexpected response %s", self.url, str(data)[:200])
            return []
        return [norm for norm in map(hit_to_norm, hits) if norm is not None]

    def check(self) -> tuple[bool, str]:
        try:
            data = self._call("/health")
        except RegSearchError as exc:
            return False, f"regulation search unreachable at {self.url}: {exc}"
        if not isinstance(data, dict) or not data.get("ready"):
            error = data.get("error") if isinstance(data, dict) else None
            return False, f"regulation search at {self.url} not ready: {error or 'loading'}"
        return True, f"regulation search: {data.get('chunks')} chunks, embedder on {data.get('device')}"

    def _call(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = urllib.request.Request(f"{self.url}{path}", data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RegSearchError(f"{path}: HTTP {exc.code} {_error_body(exc)}") from exc
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError) as exc:
            raise RegSearchError(f"{path}: {exc}") from exc


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:200]
    except (OSError, http.client.HTTPException):
        return ""


def hit_to_norm(hit: Any) -> Norm | None:
    if not isinstance(hit, dict):
        return None
    citation = str(hit.get("citation") or "").strip()
    text = str(hit.get("text") or "").strip()
    if not citation or not text:
        return None
    requisites = str(hit.get("act_requisites") or "").strip()
    title = str(hit.get("act_title") or "").strip() or requisites
    if requisites and requisites not in citation:
        citation = f"{citation} ({requisites})"
    return Norm(title=title, citation=citation, quote=truncate(text), url=str(hit.get("url") or ""), source="rag")


def truncate(text: str, limit: int = QUOTE_CHARS) -> str:
    """Cuts at the last whitespace before limit and marks the cut with an ellipsis."""
    if len(text) <= limit:
        return text
    cut = max(text.rfind(" ", 0, limit + 1), text.rfind("\n", 0, limit + 1))
    if cut < limit // 2:
        cut = limit
    return text[:cut].rstrip(" ,;:\n") + "…"


def _persons(n: int) -> str:
    few = n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14)
    return f"{n} человека" if few else f"{n} человек"


def describe_facts_ru(window: FactWindow) -> str:
    """Short Russian query for the regulation search, e.g.
    "В помещении кассового узла 3 человека. Кассовый работник покидает рабочее место; сейф открыт."
    """
    flags = {key for key, value in window.flags.items() if value and key in FLAG_PHRASES_RU}
    leaving = "cashier_leaving" in flags
    present = "cashier_at_workplace" in flags and not leaving
    phrases = []
    for key in FLAG_PHRASES_RU:
        if key not in flags or (key == "cashier_at_workplace" and leaving):
            continue
        phrases.append(CASH_UNATTENDED_RU if key == "cash_on_desk" and not present else FLAG_PHRASES_RU[key])
    if not present and not leaving and flags & {"cash_on_desk", "safe_open"}:
        phrases.insert(0, CASHIER_ABSENT_RU)
    room = "В помещении кассового узла"
    people = f"{room} {_persons(window.people)}." if window.people > 0 else f"{room} нет людей."
    if not phrases:
        return people
    body = "; ".join(phrases)
    return f"{people} {body[0].upper()}{body[1:]}."
