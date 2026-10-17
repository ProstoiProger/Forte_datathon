import argparse
import json
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ACTS = {
    "nbrk-231": ("V1900019680", "Постановление Правления Национального Банка Республики Казахстан от 29 ноября 2019 года № 231"),
    "nbrk-14": ("V2000020075", "Постановление Правления Национального Банка Республики Казахстан от 24 февраля 2020 года № 14"),
    "nbrk-49": ("V1900018545", "Постановление Правления Национального Банка Республики Казахстан от 4 апреля 2019 года № 49"),
    "nbrk-230": ("V1700016120", "Постановление Правления Национального Банка Республики Казахстан от 29 ноября 2017 года № 230"),
    "gov-305": ("P2100000305", "Постановление Правительства Республики Казахстан от 6 мая 2021 года № 305"),
}
MIRROR = "https://zakon.uchet.kz/rus/docs/{code}"
OFFICIAL = "https://adilet.zan.kz/rus/docs/{code}"
BLOCKS = ("p", "h1", "h2", "h3", "h4", "li", "td")
CLAUSE = re.compile(r"^(\d{1,3}(?:-\d)?)\.\s+(.*)")
CHAPTER = re.compile(r"^(Глава|Параграф|Раздел)\s+\d+")
NOTE = re.compile(r"^(Сноска|Примечание ИЗПИ|Примечание РЦПИ)")


class TextBlocks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[str] = []
        self._buf: list[str] | None = None
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in BLOCKS:
            self._buf = []
        elif tag == "br" and self._buf is not None:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1
        elif tag in BLOCKS and self._buf is not None:
            text = re.sub(r"[ \t ]+", " ", "".join(self._buf)).strip()
            if text:
                self.blocks.append(text)
            self._buf = None

    def handle_data(self, data):
        if self._skip == 0 and self._buf is not None:
            self._buf.append(data)


def fetch(code: str, dest: Path) -> str:
    if not dest.exists():
        request = urllib.request.Request(MIRROR.format(code=code), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            dest.write_bytes(response.read())
    return dest.read_text(encoding="utf-8", errors="replace")


def rule_body(blocks: list[str]) -> list[str]:
    start = next((i for i, b in enumerate(blocks) if b.startswith("Утверждены")), None)
    if start is None:
        start = next(i for i, b in enumerate(blocks) if b.startswith("Приложение 1") and "к постановлению" in b)
    end = next((i for i in range(start + 1, len(blocks)) if blocks[i].startswith(("Приложение", "Режим открытия документов"))), len(blocks))
    return blocks[start:end]


def split_clauses(blocks: list[str]) -> list[dict]:
    clauses, chapter, current = [], "", None
    for block in blocks:
        for line in block.splitlines():
            s = line.strip()
            if not s or NOTE.match(s):
                continue
            if CHAPTER.match(s):
                chapter, current = s, None
                continue
            m = CLAUSE.match(s)
            if m and not re.search(r"[.:;]$", s) and len(s) < 140:
                chapter, current = s, None
                continue
            if m:
                current = {"clause": m.group(1), "chapter": chapter, "text": s}
                clauses.append(current)
            elif current is not None:
                current["text"] += "\n" + s
    return clauses


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the legal acts and split them into clauses")
    parser.add_argument("--out", type=Path, default=Path("data/legal"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {}
    for act_id, (code, approved_by) in ACTS.items():
        page = TextBlocks()
        page.feed(fetch(code, args.out / f"{code}.html"))
        title = page.blocks[0] if page.blocks else ""
        clauses = split_clauses(rule_body(page.blocks))
        result[act_id] = {
            "code": code,
            "title": title,
            "approved_by": approved_by,
            "url": OFFICIAL.format(code=code),
            "source": MIRROR.format(code=code),
            "clauses": clauses,
        }
        print(f"{act_id:9s} {code} clauses={len(clauses)}")
    (args.out / "clauses.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
