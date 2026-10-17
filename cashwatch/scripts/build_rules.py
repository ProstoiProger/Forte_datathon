import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

TIER_RANK = {"A": 3, "B": 2, "DEF": 1}
LEVELS = {"да", "частично", "нет"}
PACKAGE_RULES = Path("src/cashwatch/rules")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quotes_of(entry: dict) -> list[str]:
    q = entry["quote"]
    return q if isinstance(q, list) else [q]


def c_reason(act: str, clause: dict, rules: list[dict], default: str) -> str:
    for rule in rules:
        if rule["act"] != act:
            continue
        if "clauses" in rule and clause["clause"] not in rule["clauses"]:
            continue
        if "chapter" in rule and not re.search(rule["chapter"], clause["chapter"], re.I):
            continue
        return rule["reason"]
    return default


def build(clauses_path: Path, annotations_path: Path) -> tuple[dict, dict, list[str]]:
    source = json.loads(clauses_path.read_text(encoding="utf-8"))
    ann = json.loads(annotations_path.read_text(encoding="utf-8"))
    index = {(act, c["clause"]): c for act, v in source.items() for c in v["clauses"]}
    errors: list[str] = []
    entries = []
    seen_ids = set()
    verified_ids = set(ann.get("adilet_verified", {}).get("ids", []))
    for entry in ann["entries"]:
        if entry["id"] in seen_ids:
            errors.append(f"duplicate id {entry['id']}")
        seen_ids.add(entry["id"])
        clause = index.get((entry["act"], entry["clause"]))
        if clause is None:
            errors.append(f"{entry['id']}: clause {entry['act']} {entry['clause']} not found")
            continue
        for q in quotes_of(entry):
            if norm(q) not in norm(clause["text"]):
                errors.append(f"{entry['id']}: quote not found verbatim: {q[:80]}")
        for t in entry.get("types", []):
            if t not in ann["types"]:
                errors.append(f"{entry['id']}: unknown type {t}")
        entries.append({**entry, "quote": quotes_of(entry), "chapter": clause["chapter"], "clause_text": clause["text"], "verified_adilet": entry["id"] in verified_ids})
    by_id = {e["id"]: e for e in entries}
    for type_id, spec in ann["types"].items():
        for cid in spec["citations"]:
            if cid not in by_id:
                errors.append(f"type {type_id}: citation {cid} has no entry")
            elif by_id[cid]["tier"] not in ("A", "B"):
                errors.append(f"type {type_id}: citation {cid} is not tier A or B")

    best: dict[tuple[str, str], str] = {}
    for e in entries:
        key = (e["act"], e["clause"])
        if TIER_RANK[e["tier"]] > TIER_RANK.get(best.get(key, ""), 0):
            best[key] = e["tier"]

    acts, excluded, stats = {}, [], {}
    for act, v in source.items():
        meta = ann["acts"][act]
        acts[act] = {
            "short": meta["short"],
            "document": meta["document"],
            "approved_by": v["approved_by"],
            "url": v["url"],
            "text_source": v["source"],
            "clauses_total": len(v["clauses"]),
        }
        counts = Counter()
        for c in v["clauses"]:
            tier = best.get((act, c["clause"]))
            if tier:
                counts[tier] += 1
                continue
            counts["C"] += 1
            head = norm(c["text"])
            excluded.append({
                "act": act,
                "clause": c["clause"],
                "chapter": c["chapter"],
                "tier": "C",
                "reason": c_reason(act, c, ann["c_reasons"], ann["default_c_reason"]),
                "head": head[:200] + ("..." if len(head) > 200 else ""),
            })
        stats[act] = {"total": len(v["clauses"]), **{k: counts.get(k, 0) for k in ("A", "B", "DEF", "C")}}

    type_list = []
    for type_id, spec in ann["types"].items():
        cited = [by_id[c] for c in spec["citations"] if c in by_id]
        type_list.append({
            "type_id": type_id,
            "title": spec["title"],
            "rationale": spec["rationale"],
            "citations": spec["citations"],
            "best_tier": max((e["tier"] for e in cited), key=lambda t: TIER_RANK[t], default=None),
            "dataset": spec.get("dataset", []),
        })

    codes = {v["code"]: act for act, v in source.items()}
    review = ann.get("corpus_review", {"acts": []})
    review_ids = [a["id"] for a in review["acts"]]
    if len(review_ids) != len(set(review_ids)):
        errors.append("corpus_review: duplicate act ids")
    for a in review["acts"]:
        if a["applicable"] not in LEVELS or a["visual"] not in LEVELS:
            errors.append(f"corpus_review {a['id']}: applicable and visual must be one of {sorted(LEVELS)}")
        if not a.get("reason"):
            errors.append(f"corpus_review {a['id']}: reason is empty")
        if a.get("catalog") and a["catalog"] not in acts:
            errors.append(f"corpus_review {a['id']}: catalog act {a['catalog']} is not in the catalog")
        if a["id"] in codes and a.get("catalog") != codes[a["id"]]:
            errors.append(f"corpus_review {a['id']}: parsed act must reference catalog id {codes[a['id']]}")

    liability = []
    for item in ann.get("liability", []):
        for t in item["types"]:
            if t not in ann["types"]:
                errors.append(f"{item['id']}: unknown type {t}")
        for r in item.get("related", []):
            if r not in by_id:
                errors.append(f"{item['id']}: related entry {r} not found")
        liability.append({**item, "verified_adilet": item["id"] in verified_ids})

    catalog = {
        "verification": ann.get("adilet_verified", {}),
        "acts": acts,
        "stats": stats,
        "types": type_list,
        "entries": entries,
        "excluded": excluded,
        "liability": liability,
        "corpus_review": review,
    }

    registry = {"acts": {}, "types": {}}
    for act, a in acts.items():
        registry["acts"][act] = {"short": a["short"], "title": a["document"], "approved_by": a["approved_by"], "url": a["url"]}
    registry["types"]["stub_motion"] = {
        "title": "Тестовое событие (заглушка)",
        "rationale": "Служебное событие тестового анализатора: отмечает момент наибольшей активности в кадре и не является выводом о нарушении.",
        "citations": [],
    }
    for type_id, spec in ann["types"].items():
        registry["types"][type_id] = {
            "title": spec["title"],
            "rationale": spec["rationale"],
            "citations": [
                {
                    "act": by_id[c]["act"],
                    "clause": by_id[c]["ref"],
                    "quote": " ".join(by_id[c]["quote"]),
                    "verified": by_id[c]["verified_adilet"],
                }
                for c in spec["citations"]
                if c in by_id
            ],
        }
    return catalog, registry, errors


TIER_TEXT = {
    "A": "видно напрямую",
    "B": "видно частично или зависит от ракурса",
    "DEF": "определение, используется для толкования",
    "C": "по видео не проверяется",
}


def short_quote(entry: dict, limit: int = 220) -> str:
    text = norm(" ".join(entry["quote"]))
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def render_liability(catalog: dict) -> list[str]:
    items = catalog.get("liability", [])
    if not items:
        return []
    titles = {t["type_id"]: t["title"] for t in catalog["types"]}
    lines = [
        "",
        "## Ответственность",
        "",
        "Справочный слой: сервис не делает вывод о наказании, но показывает, какая ответственность установлена за видимое нарушение.",
        "",
    ]
    for item in items:
        mark = "сверено с adilet" if item["verified_adilet"] else "не сверено с adilet"
        lines += [
            f"### {item['ref']}, {item['document']} ({mark})",
            "",
            *[f"> {norm(q)}" for q in item["quote"]],
            "",
            f"- Санкция: {item['sanction']}",
            f"- Типы: {', '.join(titles[t] for t in item['types'])}",
            f"- Связанные нормы: {', '.join(item['related'])}",
            f"- {item['note']}",
            "",
        ]
    return lines


def render_corpus_review(catalog: dict) -> list[str]:
    review = catalog.get("corpus_review") or {}
    acts = review.get("acts", [])
    if not acts:
        return []
    parsed = sum(1 for a in acts if a.get("catalog"))
    lines = [
        "",
        "## Обзор корпуса НПА",
        "",
        f"Источник: {review['source']}.",
        "",
        f"Метод: {review['method']}.",
        "",
        "Состав архива: " + "; ".join(f"{k} - {v}" for k, v in review["sections"].items()) + ".",
        "",
        f"Итог по {len(acts)} основным актам: разобраны по пунктам - {parsed}; "
        f"применимы к кассе банка - {sum(a['applicable'] == 'да' for a in acts)}, частично - {sum(a['applicable'] == 'частично' for a in acts)}, "
        f"не применимы - {sum(a['applicable'] == 'нет' for a in acts)}; "
        f"содержат визуально проверяемые нормы - {sum(a['visual'] == 'да' for a in acts)}, частично - {sum(a['visual'] == 'частично' for a in acts)}.",
        "",
        "| Акт | Реквизиты | Статус | Применим к кассе банка | Видимые нормы | Разбор по пунктам | Причина |",
        "|---|---|---|---|---|---|---|",
    ]
    order = {"да": 0, "частично": 1, "нет": 2}
    for a in sorted(acts, key=lambda a: (order[a["applicable"]], order[a["visual"]], a["section"])):
        title = a["title"] if len(a["title"]) <= 110 else a["title"][:110].rsplit(" ", 1)[0] + " ..."
        link = f"[{title}](https://adilet.zan.kz/rus/docs/{a['id']})"
        lines.append(f"| {link} | {a['requisites']} | {a['status']} | {a['applicable']} | {a['visual']} | {a['catalog'] or '-'} | {a['reason']} |")
    lines += ["", review["not_reviewed"] + "."]
    return lines


def render_markdown(catalog: dict) -> str:
    acts, stats, by_id = catalog["acts"], catalog["stats"], {e["id"]: e for e in catalog["entries"]}
    ver = catalog.get("verification", {})
    lines = [
        "# Нормативная база и сопоставление с видео",
        "",
        "Документ сгенерирован `scripts/build_rules.py` из `legal/annotations.json` и полного текста актов.",
        "",
        "## Источники",
        "",
        "| Акт | Утвержден | Пунктов | A | B | Опр. | C |",
        "|---|---|---|---|---|---|---|",
    ]
    total = Counter()
    for act_id, a in acts.items():
        s = stats[act_id]
        total.update(s)
        lines.append(f"| [{a['short']}]({a['url']}) | {a['approved_by']} | {s['total']} | {s['A']} | {s['B']} | {s['DEF']} | {s['C']} |")
    lines.append(f"| Всего | | {total['total']} | {total['A']} | {total['B']} | {total['DEF']} | {total['C']} |")
    lines += [
        "",
        "## Метод",
        "",
        "1. Полный текст каждого акта разбит на пункты; каждый пункт отнесен к одному уровню видимости.",
        "2. Уровни: " + "; ".join(f"**{k}** - {v}" for k, v in TIER_TEXT.items()) + ".",
        "3. Для пунктов уровней A, B и определений выписана дословная цитата, что именно видно на видео и что не видно.",
        "4. Каждый тип нарушения ссылается только на пункты уровней A и B. Сервис выводит ссылку на норму только из этого реестра; модель текст норм не генерирует.",
        f"5. Все цитаты сверены с текстом на adilet.zan.kz ({ver.get('checked_on', 'дата не указана')}): "
        f"{sum(e['verified_adilet'] for e in catalog['entries'])} из {len(catalog['entries'])} совпали дословно; все акты имеют статус действующих.",
        f"6. Для полноты охвата оценены все {len(catalog['corpus_review']['acts'])} основных акта архива НПА по кассовым операциям "
        "(раздел \"Обзор корпуса НПА\"); акты с визуально проверяемыми нормами разобраны по пунктам.",
        "",
        "## Типы нарушений",
        "",
        "| Тип | Основание (первое - основное) | Уровень | Примеры в обучающих данных |",
        "|---|---|---|---|",
    ]
    for t in catalog["types"]:
        refs = "; ".join(f"{by_id[c]['ref']} ({acts[by_id[c]['act']]['short']})" for c in t["citations"])
        data = ", ".join(t["dataset"]) or "-"
        lines.append(f"| {t['title']} | {refs} | {t['best_tier']} | {data} |")
    lines += ["", "Отметка `?` у примера означает гипотезу, требующую проверки просмотром.", "", "## Визуально проверяемые пункты", ""]
    for e in catalog["entries"]:
        if e["tier"] == "DEF":
            continue
        a = acts[e["act"]]
        lines += [
            f"### {e['ref']}, {a['short']} - уровень {e['tier']}",
            "",
            f"> {short_quote(e, 600)}",
            "",
            f"- Видно на видео: {e['observable']}",
        ]
        if e.get("not_observable"):
            lines.append(f"- Не видно: {e['not_observable']}")
        if e.get("note"):
            lines.append(f"- Толкование: {e['note']}")
        lines.append(f"- Типы: {', '.join(e['types'])}")
        lines.append("")
    lines += ["## Определения, используемые для толкования", ""]
    for e in catalog["entries"]:
        if e["tier"] == "DEF":
            lines.append(f"- {e['ref']}, {acts[e['act']]['short']}: {short_quote(e)}")
    lines += ["", "## Пункты, которые по видео не проверяются", ""]
    groups: dict[tuple[str, str], list[str]] = {}
    for x in catalog["excluded"]:
        groups.setdefault((x["act"], x["reason"]), []).append(x["clause"])
    for (act_id, reason), clauses in groups.items():
        lines.append(f"- **{acts[act_id]['short']}**: {reason}. Пункты: {', '.join(clauses)}.")
    lines += render_liability(catalog) + render_corpus_review(catalog)
    lines += [
        "",
        "## Ограничения",
        "",
        "- Видео не показывает документы, суммы, подписи, проводки и сроки, поэтому такие требования не проверяются (раздел 7 технического задания).",
        "- Допуск лица по распорядительному документу, наличие контролирующего работника вне кадра и наличие камеры у телефона по видео подтвердить нельзя; сервис формулирует вывод как признак возможного нарушения.",
        "- Пункты Правил № 49 применяются, если касса совмещает обменные операции; пункты требований № 305 - если объект отнесен к уязвимым в террористическом отношении.",
        "- Пункты 8 и 16 Правил № 14 - конструктивные требования; открытая дверь трактуется как неиспользование запирающегося входа.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate legal annotations and build the rule catalog and registry")
    parser.add_argument("--clauses", type=Path, default=Path("data/legal/clauses.json"))
    parser.add_argument("--annotations", type=Path, default=Path("legal/annotations.json"))
    parser.add_argument("--out", type=Path, default=PACKAGE_RULES)
    parser.add_argument("--doc", type=Path, default=Path("docs/RULES.md"))
    args = parser.parse_args()
    catalog, registry, errors = build(args.clauses, args.annotations)
    for error in errors:
        print(f"ERROR {error}")
    if errors:
        return 1
    (args.out / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(render_markdown(catalog), encoding="utf-8")
    total = Counter()
    for act, s in catalog["stats"].items():
        print(f"{act:9s} total={s['total']:3d} A={s['A']:2d} B={s['B']:2d} DEF={s['DEF']:2d} C={s['C']:3d}")
        total.update(s)
    print(f"{'all':9s} total={total['total']:3d} A={total['A']:2d} B={total['B']:2d} DEF={total['DEF']:2d} C={total['C']:3d}")
    verified = sum(e["verified_adilet"] for e in catalog["entries"])
    print(f"entries={len(catalog['entries'])} verified_on_adilet={verified} types={len(catalog['types'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
