# Заморозка поиска перед контрольным замером

Время: 2026-10-17T19:48:22.
Контрольный набор eval/control_set.json пишет отдельный агент вслепую; до этой записи я его не открывал.
Поиск настраивался только на eval/dev_set.json: семь примеров из задания и девять моих запросов.
После заморозки код поиска, правила, модель векторов и порог не меняются. Если найдётся ошибка в эталоне, правится эталон с записью в журнал, поиск остаётся прежним.

Модель векторов: intfloat/multilingual-e5-base. Порог сходства: 0.80.

| Файл | SHA-256, первые 12 знаков |
|---|---|
| app/__init__.py | e3b0c44298fc |
| app/attrs.py | f95d70406d51 |
| app/cards.py | f9b69b258188 |
| app/config.py | 032e7a3ca148 |
| app/ksr.py | 5ca7eddc9da4 |
| app/ksr_select.py | 1d1dcd4be3af |
| app/nametype.py | 1a2c4a5a87c8 |
| app/normalize.py | f25cbc4f8ad7 |
| app/passages.py | 70b4ceb4d2ed |
| app/prices.py | 2710180c781d |
| app/search.py | 426e65ae6409 |
| app/searcher.py | ddb3802fa19a |
| app/verify.py | 7638c14c9e47 |
| rules/rules.yaml | 0102fdfb05d3 |
| scripts/eval_search.py | 25df23ebbc6c |
| data/index/e5-base/meta.json | faedc1f2ba28 |

# Заморозка v3 перед итоговым замером

Время: 2026-10-18T16:21:13.
Итоговые наборы: ChatGPT (final_set) и Gemini (final_gemini_set), размечены агентами вслепую; поиск на них не запускался.
Основной вариант: main (обучаемое решение с реранкером, порог 0.92 из перекрёстной проверки), запасной main_light (порог 0.85).
Прогон один раз, вариантами words_raw, words_norm, cards, vectors, rules, main, main_light.
Заранее объявленные цифры, которые будут показаны по каждому набору и варианту: верный ответ первым, верный в пятёрке, MRR@5,
первым позиция с другой важной характеристикой, честное «не найдено», ложное «не найдено»; интервалы Уилсона;
счёт выигрышей и проигрышей main против поиска по словам по «верный первым». После прогона код не правится до разбора ошибок,
правки — с записью в журнал.

| Файл | SHA-256, первые 12 знаков |
|---|---|
| app/__init__.py | e3b0c44298fc |
| app/attrs.py | f95d70406d51 |
| app/cards.py | f9b69b258188 |
| app/config.py | 032e7a3ca148 |
| app/decision.py | 8054feeefe0f |
| app/ksr.py | 5ca7eddc9da4 |
| app/ksr_select.py | 1d1dcd4be3af |
| app/nametype.py | 1a2c4a5a87c8 |
| app/normalize.py | f25cbc4f8ad7 |
| app/passages.py | 70b4ceb4d2ed |
| app/prices.py | 2710180c781d |
| app/search.py | 426e65ae6409 |
| app/searcher.py | 48c19eef1444 |
| app/verify.py | 7638c14c9e47 |
| rules/rules.yaml | 0102fdfb05d3 |
| data/model/decision.json | a9a1d855ffe4 |
| scripts/eval_search.py | e6117072ded7 |
| scripts/train_decision.py | c4eafe1a7437 |
| eval/final_set.json | 9861907137a8 |
| eval/final_gemini_set.json | 5dbc1f9dda36 |
| data/full/prep_all.jsonl | dccf4b15577c |
