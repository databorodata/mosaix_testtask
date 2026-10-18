"""Тип наименования: какой слой имени у строки. Графы спецификации по ГОСТ 21.110 склеивают
наименование, тип или марку и поставщика, поэтому в строке встречаются марки по ГОСТ,
типовые обозначения ФСК, артикулы производителей и описания без кода."""

from __future__ import annotations

import re

from .normalize import clean, codes

NAMES = {
    "gost": "марка или обозначение по ГОСТ или ТУ",
    "typical": "типовое обозначение (ФСК, типовые серии)",
    "article": "артикул или бренд производителя",
    "descriptive": "описание без кода",
    "service": "услуга или работа",
    "kit": "комплект «в составе»",
}
LABEL_GROUP = {"gost": "standard", "typical": "standard", "article": "article",
               "descriptive": "descriptive", "service": "service", "kit": "kit"}

SERVICE = re.compile(r"перевозк|погрузк|разгрузк|транспортировк", re.I)
KIT = re.compile(r"в\s+составе", re.I)
GOST_MARK = re.compile(
    r"(?<![А-Яа-яA-Za-z])(?:[АA]?(?:КВВГ|ВВГ|ВБШв|ПвП|ПвБ|ПвВ|ПВ[1-5]|КГ|КуГВВ|ПуГВ|СИП|КСПВ|КИПЭВ|МКЭШ|ОКГТ|ОКСН|ОКЛ|ОКБ)"
    r"|АС ?\d|РК ?\d|А-[IV]+|A-[IV]+|А(?:240|300|400|500|600)\b|Вр-I|С(?:235|245|255|345|355|390)\b|Ст3"
    r"|М\d{1,2}(?:[хx]\d+)?\b|ГОСТ|ТУ\s?\d|SN\d|SDR)"
)
TYPICAL = re.compile(
    r"(?<![А-Яа-я])(?:ШЭТ|ШЭЛ|ШЭЭ|ШЭ\d|ШНЭ|ПС-\d+[ЯШПТ]|ФПС ?\d|Ф\d-\d|П\d{2,3}-\d|У\d{2,3}-\d|ПРТ-|ЗПС-|ОА-\d"
    r"|У1К-|ПТМ-|КГП-|ПРВТ|ДС-\d|ЛК ?\d|ЦН\.\d|Р\d-[АA])"
)
BRANDS = re.compile(
    r"(?<![A-Za-zА-Яа-я])(?:DKC|ДКС|K-FLEX|KFLEX|Hanwha|Wisenet|Samsung|Omicron|OMICRON|Weidmuller|Logitech|HP|Asus"
    r"|Rexant|ЭРА|Ecoplast|Legrand|Schneider|ABB|Rittal|Siemens|IEK|EKF|TDM|Phoenix|Wago|Airbox|Rainpark|Metrol"
    r"|METROL|Fargo|FARGO|SKAT|СКАТ|Бастион|СВЭЛ|ЭКРА|Релематика|ССД|РЕТОМ|Икапласт|ИКАПЛАСТ|ИЗОКОРСИС|Техноэласт"
    r"|Ровен|Веза|TANKMEISTER|Точка-С|Стандарт|LPS|Hyperline|Optibox|ДЫМЪ|Цинол|Elsys|Freezstop)(?![A-Za-zА-Яа-я])"
)


def classify(name: str) -> str:
    s = clean(name)
    if KIT.search(s):
        return "kit"
    if SERVICE.search(s):
        return "service"
    found = codes(s)
    latin_code = any(re.search(r"[A-Za-z]", c) and re.search(r"\d", c) for c in found)
    if BRANDS.search(s) or latin_code:
        return "article"
    if TYPICAL.search(s):
        return "typical"
    if GOST_MARK.search(s):
        return "gost"
    if found:
        return "article"  # кириллический код без бренда: по умолчанию считаем фирменным
    return "descriptive"
