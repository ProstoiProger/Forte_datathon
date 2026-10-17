import numpy as np
import pytest

from npa_rag import config
from npa_rag.corpus import act_kind, parse_act, short_requisites
from npa_rag.evaluate import norm
from npa_rag.hybrid import rrf_fuse, top_ranks
from npa_rag.sparse import Sparse

ACT = {"id": "V1", "title": "Об утверждении Правил кассовых операций", "status": "Обновленный",
       "section": "02_cash", "requisites": "Постановление Правления НБРК от 1 января 2020 года № 7. Зарегистрировано ...",
       "source": "https://adilet.zan.kz/rus/docs/V1"}

TEXT = """Об утверждении Правил кассовых операций
Постановление Правления НБРК от 1 января 2020 года № 7.
Статус источника: Обновленный
Дата изменения/принятия: 01.01.2026
Источник: https://adilet.zan.kz/rus/docs/V1
Дата выгрузки:17.10.2026; состояние базы: 14.10.2026

Примечание ИЗПИ!
Настоящее постановление вводится в действие с 1 января 2020 года.

1. Утвердить прилагаемые Правила.

2. Настоящее постановление вводится в действие.

Утверждены
постановлением Правления
от 1 января 2020 года № 7

Правила кассовых операций
Глава 1. Общие положения

1. Настоящие Правила определяют порядок.

Глава 2. Хранение
Параграф 1. Сейф

12. Хранение личных вещей осуществляется вне кассового узла.
Допускается внос лекарственных средств.

Сноска. Пункт 12 - в редакции постановления от 01.01.2026 № 1.
13. Сейф закрывается на ключ:
1) после окончания работы;
2) при выходе кассира.

Приложение 1
к Правилам кассовых операций

Форма
1. Наименование банка
"""


def chunks():
    return list(parse_act(ACT, TEXT))


def test_parse_segments_labels_and_paths():
    cs = chunks()

    def find(segment: str, label: str):
        return next(c for c in cs if c.segment.startswith(segment) and c.label == label)

    assert find("Об утверждении Правил", "п. 1")             # point of the resolution itself
    rules_12 = find("Правила кассовых операций", "п. 12")
    assert rules_12.path == ["Глава 2. Хранение", "Параграф 1. Сейф"]
    assert "Допускается внос" in rules_12.text
    assert rules_12.notes == ["Сноска. Пункт 12 - в редакции постановления от 01.01.2026 № 1."]
    assert "Сноска" not in rules_12.text
    rules_13 = find("Правила кассовых операций", "п. 13")
    assert rules_13.text.endswith("2) при выходе кассира.")
    appendix = [c for c in cs if c.segment.startswith("Приложение 1 к Правилам")]
    assert appendix and appendix[-1].label == "п. 1"
    assert rules_12.citation == "Правила кассовых операций, Параграф 1. Сейф, п. 12"


def test_editorial_note_not_indexed():
    first = chunks()[0]
    assert "вводится в действие с 1 января 2020" not in first.text


def test_articles_keep_their_parts():
    act = {**ACT, "id": "K1", "title": "Кодекс об административных правонарушениях"}
    text = "\n".join(["t", "r", "s", "d", "i", "Дата выгрузки: x", "", "Глава 15. Нарушения", "",
                      "Статья 252-1. Нарушение требований", "", "1. Первая часть", "", "влечет штраф.", "",
                      "2. Вторая часть", "", "Статья 253. Другое", "", "1. Текст"])
    cs = list(parse_act(act, text))
    assert [c.label for c in cs] == ["ст. 252-1", "ст. 253"]
    assert "2. Вторая часть" in cs[0].text and cs[0].unit == "article"


def test_long_unit_is_split_with_same_label():
    body = "\n".join(f"{i}) " + "слово " * 60 for i in range(1, 15))
    text = "h\nh\nh\nh\nh\nДата выгрузки: x\n\n5. Начало пункта:\n" + body
    cs = list(parse_act(ACT, text))
    assert len(cs) > 1 and {c.label for c in cs} == {"п. 5"}
    assert all(len(c.text) <= 1800 for c in cs) and cs[0].part == f"1/{len(cs)}"


def test_kinds_and_requisites():
    assert act_kind({"status": "Утративший силу", "section": "90_archive"}) == "repealed"
    assert act_kind({"status": "Новый", "section": "91_amendments"}) == "amendment"
    assert act_kind({"status": "Обновленный", "section": "02_cash"}) == "base"
    assert short_requisites(ACT["requisites"]) == "Постановление Правления НБРК от 1 января 2020 года № 7"


def test_rrf_prefers_agreement():
    fused = rrf_fuse({"bm25": [1, 2, 3], "dense": [3, 1, 4]})
    order = [i for i, _, _ in fused]
    assert order[:2] == [1, 3]                 # in both lists near the top
    assert fused[0][2] == {"bm25": 1, "dense": 2}
    assert fused[0][1] == pytest.approx(1 / (config.RRF_K + 1) + 1 / (config.RRF_K + 2))


def test_top_ranks_mask_and_zero():
    s = np.array([0.0, 3.0, 2.0, 5.0])
    m = np.array([True, True, True, False])
    assert top_ranks(s, m, 10) == [1, 2]       # masked row 3 and zero-score row 0 dropped


def test_bm25_russian_stemming():
    sp = Sparse.build(["Кассир покинул рабочее место", "Сейф закрывается на ключ"], ["a", "b"])
    assert int(np.argmax(sp.scores("кассиры покидают место"))) == 0
    assert int(np.argmax(sp.scores("сейфа ключи"))) == 1
    assert not sp.scores("инкассатор").any()


def test_norm_quotes():
    assert norm("«Сейф» — закрыт ё.") == norm('"сейф" - закрыт е')
