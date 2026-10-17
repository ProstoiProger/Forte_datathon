import json
from importlib import resources

from cashwatch.analyzers.base import Event
from cashwatch.results import describe
from cashwatch.rules import load_registry


def _catalog() -> dict:
    return json.loads(resources.files("cashwatch.rules").joinpath("catalog.json").read_text(encoding="utf-8"))


def test_every_violation_type_has_verified_citation() -> None:
    registry = load_registry()
    for vtype in registry.all():
        if vtype.type_id == "stub_motion":
            assert vtype.citations == ()
            continue
        assert vtype.verified_citations, vtype.type_id
        for citation in vtype.verified_citations:
            assert citation.quote.strip()
            assert citation.act.url.startswith("https://adilet.zan.kz/")


def test_catalog_covers_every_clause_once() -> None:
    catalog = _catalog()
    for act, stats in catalog["stats"].items():
        assert stats["A"] + stats["B"] + stats["DEF"] + stats["C"] == stats["total"] == catalog["acts"][act]["clauses_total"]
    excluded = {(e["act"], e["clause"]) for e in catalog["excluded"]}
    annotated = {(e["act"], e["clause"]) for e in catalog["entries"]}
    assert not excluded & annotated
    assert all(e["reason"] for e in catalog["excluded"])


def test_catalog_entries_verified_against_adilet() -> None:
    catalog = _catalog()
    assert catalog["entries"]
    assert all(e["verified_adilet"] for e in catalog["entries"])
    assert all(e["tier"] in {"A", "B", "DEF"} for e in catalog["entries"])


def test_corpus_review_covers_every_base_act() -> None:
    catalog = _catalog()
    acts = catalog["corpus_review"]["acts"]
    assert len(acts) == 31 == len({a["id"] for a in acts})
    levels = {"да", "частично", "нет"}
    assert all(a["applicable"] in levels and a["visual"] in levels and a["reason"] for a in acts)
    parsed = {a["catalog"] for a in acts if a["catalog"]}
    assert parsed <= set(catalog["acts"])
    assert all(a["applicable"] != "нет" for a in acts if a["visual"] != "нет")


def test_liability_is_verified_and_linked() -> None:
    catalog = _catalog()
    types = {t["type_id"] for t in catalog["types"]}
    entries = {e["id"] for e in catalog["entries"]}
    assert {i["id"] for i in catalog["liability"]} == {"koap/252-1/5", "koap/252-1/6"}
    for item in catalog["liability"]:
        assert item["verified_adilet"]
        assert set(item["types"]) <= types and set(item["related"]) <= entries
        assert item["url"].startswith("https://adilet.zan.kz/")


def test_registry_types_match_catalog_types() -> None:
    registry_ids = {t.type_id for t in load_registry().all()} - {"stub_motion"}
    assert registry_ids == {t["type_id"] for t in _catalog()["types"]}


def test_description_cites_primary_clause_only() -> None:
    registry = load_registry()
    vtype = registry.get("unauthorized_entry")
    text = describe(Event(type_id="unauthorized_entry", timecode_sec=1.0, confidence=0.9, observation="В помещение входит мужчина."), vtype)
    assert text.count("Основание:") == 1
    assert "п. 75" in text
    stub = describe(Event(type_id="stub_motion", timecode_sec=1.0, confidence=0.5, observation="x"), registry.get("stub_motion"))
    assert "Основание" not in stub
