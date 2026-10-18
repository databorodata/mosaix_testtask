# Карта репозитория и команды (для человека и ИИ-агента)

## Какой файл за что отвечает

| Шаг | Файл · функция |
|---|---|
| Чистка строки, коды | `app/normalize.py` · `clean`, `codes` |
| Тип строки (услуга, комплект, артикул…) | `app/nametype.py` · `classify` |
| Веб-поиск на подготовке | `app/search.py` (кэш `data/cache/search/`) |
| Фильтр фрагментов | `app/passages.py` · `filter_docs` |
| Карточка позиции (модель) | `app/cards.py` · `make_card` |
| Проверка карточки | `app/verify.py` · `verify` |
| Справочник КСР | `app/ksr.py`, `app/ksr_select.py` |
| Разбор характеристик запроса и строк | `app/attrs.py` · `parse`, `kind_match`, `code_relation` |
| Словари специалиста | `rules/synonyms.yaml` (виды), `rules/rules.yaml` (правила замен) |
| Поиск: каналы, слияние, правила, статусы | `app/searcher.py` · `Engine.search`, `judge`, `_main` |
| Признаки и решение «подходит?» | `app/decision.py` (модель `data/model/decision.json`) |
| Обучение решения и порога | `scripts/train_decision.py` |
| Страница и API | `app/web.py`, `static/index.html` |
| Замер по наборам | `scripts/eval_search.py`, наборы и результаты в `eval/` |

## Команды

```bash
.venv/bin/python -m pytest -q                      # тесты разбора; известные ошибки — xfail
.venv/bin/python -m scripts.attrs_diff             # какие строки базы изменили разбор (снимок tests/snapshots/)
.venv/bin/python -m scripts.whatchanged            # какие из 176 запросов изменили выдачу (эталон eval/baseline_main.json)
.venv/bin/python -m scripts.eval_search --set eval/control_set.json --out /tmp/eval_check --variants words_raw,main
.venv/bin/python -m scripts.train_decision         # переобучить решение и порог (после правки признаков)
```

## Порядок починки ошибки

1. Падающий тест в `tests/test_attrs.py`, воспроизводящий симптом (или снять xfail со строгого).
2. Правка.
3. `attrs_diff` — разбор базы не пополз? `whatchanged` — какие запросы изменили выдачу?
4. `pytest` и быстрый замер на контрольном наборе.
5. Запись в `eval/JOURNAL.md`: что поменяли, какие цифры сдвинулись.

Правки словарей и правил — только в `rules/*.yaml`; содержимое промптов подготовки не менять без
повторного прогона затронутых строк. Итоговые наборы `eval/final_*.json` уже использованы в замере:
настройка по ним запрещена, любые новые цифры по ним помечать «после правок».
