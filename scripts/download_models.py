"""Скачивание моделей в кэш при сборке образа: дальше контейнер работает без сети.

Загружаем через те же классы, что и поиск: в кэш попадают только нужные файлы
(у моделей на хабе лежат дубликаты в нескольких форматах, целиком они вдвое тяжелее).
"""
from sentence_transformers import CrossEncoder, SentenceTransformer

print("скачиваю intfloat/multilingual-e5-base", flush=True)
SentenceTransformer("intfloat/multilingual-e5-base")
print("скачиваю BAAI/bge-reranker-v2-m3", flush=True)
CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=256)
print("готово")
