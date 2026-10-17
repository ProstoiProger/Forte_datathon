import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class Act:
    act_id: str
    short: str
    title: str
    approved_by: str
    url: str


@dataclass(frozen=True)
class Citation:
    act: Act
    clause: str
    quote: str
    verified: bool

    def text(self) -> str:
        return f"{self.clause}, {self.act.short}"


@dataclass(frozen=True)
class ViolationType:
    type_id: str
    title: str
    rationale: str
    citations: tuple[Citation, ...]

    @property
    def verified_citations(self) -> tuple[Citation, ...]:
        return tuple(c for c in self.citations if c.verified)


class Registry:
    def __init__(self, types: dict[str, ViolationType]) -> None:
        self._types = types

    def get(self, type_id: str) -> ViolationType:
        if type_id not in self._types:
            raise KeyError(f"unknown violation type {type_id!r}")
        return self._types[type_id]

    def __contains__(self, type_id: str) -> bool:
        return type_id in self._types

    def all(self) -> list[ViolationType]:
        return list(self._types.values())


def load_registry(path: Path | None = None) -> Registry:
    if path is None:
        raw = resources.files(__package__).joinpath("registry.json").read_text(encoding="utf-8")
    else:
        raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    acts = {k: Act(act_id=k, **v) for k, v in data["acts"].items()}
    types = {}
    for type_id, spec in data["types"].items():
        citations = tuple(
            Citation(act=acts[c["act"]], clause=c["clause"], quote=c.get("quote", ""), verified=bool(c.get("verified")))
            for c in spec.get("citations", [])
        )
        for c in citations:
            if c.verified and not c.quote.strip():
                raise ValueError(f"{type_id}: verified citation {c.clause} needs a verbatim quote")
        types[type_id] = ViolationType(type_id=type_id, title=spec["title"], rationale=spec["rationale"], citations=citations)
    return Registry(types)
