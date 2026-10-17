"""npa_bank_kz texts -> citeable chunks.

A chunk is one legal unit: a point ("п. 75") of a resolution, rules or appendix, or an article ("ст. 252-1") of a
code or law, with its heading path (part, chapter, paragraph). Units longer than MAX_CHARS are split at line
boundaries and keep the same label. Editorial notes ("Сноска.", "Примечание ИЗПИ!") are kept as metadata and are
not indexed: they describe amendments, not the norm.

Acts are classified by the registry: "repealed" (status "Утративший силу"), "amendment" (amending acts, section
91_amendments; their content is already merged into the consolidated texts) and "base" (everything else).
"""

import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_CHARS = 1800

RE_APPROVED = re.compile(r"^(Утвержден[аоы]?|УТВЕРЖДЕН[АОЫ]?)\s*$")
RE_APPENDIX = re.compile(r"^(Приложение(?:\s+[\d-]+)?)\s*$|^(Приложение\s+[\d-]+)\s+(к\s.*)$")
RE_PART = re.compile(r"^(Раздел|РАЗДЕЛ)\s+([\dIVXLC]+(?:-\d+)?)\.?\s*(.*)$")
RE_CHAPTER = re.compile(r"^(Глава|ГЛАВА)\s+(\d+(?:-\d+)?)\.?\s*(.*)$")
RE_PARAGRAPH = re.compile(r"^Параграф\s+(\d+(?:-\d+)?)\.?\s*(.*)$")
RE_ARTICLE = re.compile(r"^Статья\s+(\d+(?:-\d+)*)\.?\s*(.*)$")
RE_POINT = re.compile(r"^(\d+(?:-\d+)?)\.\s+\S")
RE_FOOTNOTE = re.compile(r"^Сноска\.")
RE_EDITOR = re.compile(r"^Примечание\s+(ИЗПИ|РЦПИ)")
HEADER_END = re.compile(r"^Дата выгрузки")


@dataclass
class Chunk:
    id: str
    act_id: str
    kind: str                # base | amendment | repealed
    status: str
    section: str
    act_title: str
    act_requisites: str
    segment: str             # resolution, approved document or appendix the unit belongs to
    path: list[str]          # part / chapter / paragraph headings
    unit: str                # point | article | text
    label: str               # "п. 75", "ст. 252-1", "" for unnumbered text
    part: str                # "2/3" when a long unit is split, else ""
    text: str
    notes: list[str] = field(default_factory=list)
    line: int = 0            # 1-based line of the unit start in the text file
    source: str = ""

    @property
    def citation(self) -> str:
        bits = [self.segment] + self.path[-1:] + ([self.label] if self.label else [])
        return ", ".join(b for b in bits if b)

    def index_text(self) -> str:
        """Text given to BM25 and the embedder: short act title, heading path, label, body."""
        head = " / ".join([self.segment[:200], *self.path])
        return f"{head}. {self.label}. {self.text}" if self.label else f"{head}. {self.text}"


def act_kind(act: dict) -> str:
    if act["status"] == "Утративший силу":
        return "repealed"
    if act["section"] == "91_amendments":
        return "amendment"
    return "base"


def short_requisites(requisites: str) -> str:
    """'Постановление ... от 29 ноября 2019 года № 231. Зарегистрировано ...' -> first sentence."""
    m = re.match(r"(.+?№\s*[\w-]+)", requisites)
    return (m.group(1) if m else requisites.split(". Зарегистрировано")[0]).strip()


def _split_long(lines: list[str], limit: int) -> list[list[str]]:
    parts, cur, size = [], [], 0
    for ln in lines:
        while len(ln) > limit:  # a single huge line (tables): cut at the last space before the limit
            cut = ln.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if cur:
                parts.append(cur)
                cur, size = [], 0
            parts.append([ln[:cut]])
            ln = ln[cut:].lstrip()
        if size + len(ln) > limit and cur:
            parts.append(cur)
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        parts.append(cur)
    return parts


def parse_act(act: dict, text: str) -> Iterator[Chunk]:
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines[:12]):
        if HEADER_END.match(ln):
            start = i + 1
            break
    kind = act_kind(act)
    requisites = short_requisites(act.get("requisites") or "")
    segment = act["title"]
    path: list[str] = []
    unit = {"unit": "text", "label": "", "lines": [], "notes": [], "line": start + 1}
    seq = 0
    editor_note = False
    pending_title = False   # after an "Утверждены" block the next long line is the document title
    in_approval = False
    in_appendix_header = False
    appendix_to = False      # the appendix header already has its "к ..." line
    in_article = False

    def flush():
        nonlocal seq
        body = [ln for ln in unit["lines"] if ln.strip()]
        if not body:
            return
        pieces = _split_long(body, MAX_CHARS)
        for k, piece in enumerate(pieces, 1):
            seq += 1
            yield Chunk(
                id=f"{act['id']}:{seq:05d}", act_id=act["id"], kind=kind, status=act["status"],
                section=act["section"], act_title=act["title"], act_requisites=requisites, segment=segment,
                path=list(path), unit=unit["unit"], label=unit["label"],
                part=f"{k}/{len(pieces)}" if len(pieces) > 1 else "", text="\n".join(piece),
                notes=list(unit["notes"]) if k == 1 else [], line=unit["line"], source=act.get("source", ""))

    def new_unit(kind_: str, label: str, line_no: int, first: str | None):
        unit.update(unit=kind_, label=label, lines=[first] if first else [], notes=[], line=line_no)

    for i in range(start, len(lines)):
        raw = lines[i]
        ln = raw.strip()
        if not ln:
            if in_approval:          # the approval block ends at the first blank line
                in_approval, pending_title = False, True
            if in_appendix_header:
                in_appendix_header = False
            continue
        if RE_FOOTNOTE.match(ln):
            unit["notes"].append(ln)
            continue
        if RE_EDITOR.match(ln):
            unit["notes"].append(ln)
            editor_note = True
            continue
        if editor_note:              # the paragraph right after "Примечание ИЗПИ!" is editorial too
            unit["notes"].append(ln)
            editor_note = False
            continue
        if in_approval:
            continue
        if in_appendix_header:       # "Приложение 1" / "к Правилам ..." / continuation lines up to a blank line
            if appendix_to or ln[:2].lower() == "к ":
                segment += " " + ln
                appendix_to = True
                continue
            in_appendix_header = False
        if RE_APPROVED.match(ln) and not segment.startswith("Приложение"):
            yield from flush()
            new_unit("text", "", i + 1, None)
            in_approval, path, in_article = True, [], False
            continue
        if pending_title:
            pending_title = False
            if not (RE_CHAPTER.match(ln) or RE_PART.match(ln) or RE_POINT.match(ln) or RE_ARTICLE.match(ln)):
                segment = ln
                continue
        m = RE_APPENDIX.match(ln)
        if m and len(ln) < 160:
            yield from flush()
            new_unit("text", "", i + 1, None)
            segment = ln
            path, in_article, in_appendix_header = [], False, True
            appendix_to = bool(m.group(3))
            continue
        for rx, level in ((RE_PART, 0), (RE_CHAPTER, 1), (RE_PARAGRAPH, 2)):
            if rx.match(ln):
                yield from flush()
                new_unit("text", "", i + 1, None)
                depth = {0: 0, 1: 1 if path and path[0].lower().startswith("раздел") else 0, 2: None}[level]
                if level == 2:
                    path = [p for p in path if not p.startswith("Параграф")][:2] + [ln]
                else:
                    path = path[:depth] + [ln]
                in_article = False
                break
        else:
            m = RE_ARTICLE.match(ln)
            if m:
                yield from flush()
                new_unit("article", f"ст. {m.group(1)}", i + 1, ln)
                in_article = True
                continue
            m = RE_POINT.match(ln)
            if m and not in_article:
                yield from flush()
                new_unit("point", f"п. {m.group(1)}", i + 1, ln)
                continue
            unit["lines"].append(ln)
            continue
        continue
    yield from flush()


def load_registry(npa_dir: Path) -> list[dict]:
    return json.loads((npa_dir / "registry.json").read_text(encoding="utf-8"))["acts"]


def build_chunks(npa_dir: Path, kinds: set[str]) -> list[Chunk]:
    chunks = []
    for act in load_registry(npa_dir):
        if act_kind(act) not in kinds:
            continue
        text = (npa_dir / act["textFile"]).read_text(encoding="utf-8")
        chunks.extend(parse_act(act, text))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
