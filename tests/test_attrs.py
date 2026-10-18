"""Тесты разбора характеристик: что работает и что известно сломанным.

xfail(strict=True) — известные ошибки разбора: тест ПАДАЛ на момент сдачи. Если после правки
он прошёл, pytest потребует снять пометку — так починка фиксируется явно. После любой правки
разбора прогнать ещё: scripts/attrs_diff (строки базы) и scripts/whatchanged (выдача).
"""

import pytest

from app.attrs import code_relation, kind_match, parse
from app.decision import query_type
from app.normalize import clean


def P(text):
    return parse(clean(text))


# --- что должно работать ---

def test_cable_section():
    a = P("кабель КВВГнг-LS 7*1.5")
    assert a.kind == "кабель" and (7.0, 1.5) in a.section


def test_section_separators_equal():
    assert P("кабель 3х1,5").section == P("кабель 3*1.5").section == P("кабель 3×1,5 мм2").section


def test_sizes_order_free():
    assert P("ворота откатные 6000х2250").sizes == P("ворота 2250x6000").sizes == {6000.0, 2250.0}


def test_transformer_power():
    assert 1000.0 in P("трансформатор силовой сухой 1000 кВА").power_kva


def test_camera_resolution_and_kind():
    a = P("камера IP уличная 2 Мп вариофокальная")
    assert a.kind == "камера" and 2.0 in a.resolution


def test_kind_synonym_from_yaml():
    assert kind_match("камера", ["видеокамера"])


def test_abbr_ibp():
    assert "источник" in " ".join(P("ИБП 12В для улицы").nouns)


def test_du_capital_is_size():
    assert 80.0 in P("Фланец Ду80").sizes


def test_code_relation_latin_cyrillic():
    q, c = P("кабель КВВГнг(A)-LS 5х1,5"), P("Кабель контрольный КВВГнг(А)-LS 5х1,5")
    assert code_relation(q, c)[0] == "match"


def test_split_code_kept():
    # ключ кода нормализован: кириллические двойники сложены с латиницей (С→C, Т→T)
    assert "ФБC1243" in P("Блоки ФБС 12.4.3 т").codes


def test_service_query_type():
    q = P("перевозка грузов до 5 т на расстояние до 6 км")
    assert query_type(q, "перевозка грузов до 5 т") == "service"


def test_adjustment_query_type():
    q = P("надбавка по водонепроницаемости W6")
    assert query_type(q, "надбавка по водонепроницаемости W6") == "adjustment"


# --- известные ошибки разбора (чинить только со снимком базы и повторным замером) ---

@pytest.mark.xfail(strict=True, reason="«миллиметров» словом читается как метры: 159 000 мм")
def test_bug_millimeters_word():
    assert P("труба наружным диаметром 159 миллиметров").sizes == {159.0}


@pytest.mark.xfail(strict=True, reason="строчное «д125» принимается за код, размер теряется")
def test_bug_lowercase_d_diameter():
    assert 125.0 in P("воздуховод спиральный оцинк. д125").sizes


@pytest.mark.xfail(strict=True, reason="строчные «ду80», «ру10» принимаются за коды")
def test_bug_lowercase_du():
    assert 80.0 in P("фланцы привар. ду80 ру10 к-т").sizes


@pytest.mark.xfail(strict=True, reason="вид «датчик» не совпадает с «извещатель», нет в словаре синонимов")
def test_bug_kind_detector():
    assert kind_match("датчик", ["извещатель"])


@pytest.mark.xfail(strict=True, reason="«006В» из маркировки читается как напряжение 6 В")
def test_bug_leading_zero_voltage():
    assert not P("клапан BA10-006В").voltage


@pytest.mark.xfail(strict=True, reason="кириллическая «ф» перед числом не распознаётся как диаметр")
def test_bug_cyrillic_f_diameter():
    assert 159.0 in P("труба ф159х6").sizes
