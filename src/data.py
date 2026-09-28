"""Загрузка и подготовка данных.

Что делаем:
  * объединяем объявления из корпуса бенчмарка и из train в одну таблицу (тексты объявлений из train
    нужны для «расширения запроса» и для валидации);
  * лемматизируем заголовок / параметры / начало описания (pymorphy3);
  * разбираем параметры: вид услуги, типы услуги (для совпадения с фильтрами запроса).

Train обрабатывается ПОТОКОВО, по кускам: пиковая память не зависит от размера train.
Каждый обработанный кусок сразу сохраняется в cache/prep/ — если процесс прервётся,
повторный запуск продолжит с первого необработанного куска.
"""
import gc
import glob
import hashlib
import os
import re

import pandas as pd
import pyarrow.parquet as pq

from params import build_item_parsers, parse_query_params
from textproc import lemma, lemmas, normalize

DESC_TOKENS = 120  # из описания берём только начало: суть услуги обычно в первых фразах
CHUNK_ROWS = 20_000  # train читаем кусками по 20 тыс. строк, чтобы не держать все описания в памяти
ITEM_KEEP = ["item_id", "in_bench", "item_title_raw", "item_location_id", "item_microcat_id", "item_category_id",
             "item_latitude", "item_longitude", "item_price", "item_rating", "item_rating_reviews_count",
             "i_vid", "i_tips", "title_lem", "params_lem", "desc_lem", "title_norm"]
Q_COLS = ["search_query", "search_location_id", "search_infm_params_text", "search_category",
          "search_is_delivery_search"]


def _read(data_dir, name, columns=None):
    """Небольшие файлы (benchmark_*) читаем целиком."""
    return pd.read_parquet(os.path.join(data_dir, f"{name}.parquet"), columns=columns)


def iter_train(data_dir, columns=None, train_rows=None):
    """Отдаём train кусками по CHUNK_ROWS строк — в памяти всегда только один кусок."""
    path = os.path.join(data_dir, "train.parquet")
    batches = (b.to_pandas() for b in pq.ParquetFile(path).iter_batches(batch_size=CHUNK_ROWS, columns=columns))
    seen = 0
    for i, df in enumerate(batches):
        if train_rows:
            df = df.iloc[: max(0, train_rows - seen)]
        seen += len(df)
        if len(df) == 0:
            break
        yield i, df.reset_index(drop=True)


def _clean_params(s):
    # адрес («Место оказания услуг ...») для текстового поиска — шум
    return re.sub(r"Место оказания услуг .*?(?= Тип | Вид | Опыт | Как вы|$)", " ", s or "")


def _prep_queries(df):
    out = df[Q_COLS].copy().reset_index(drop=True)
    qp = df.search_infm_params_text.map(parse_query_params).apply(pd.Series).reset_index(drop=True)
    return pd.concat([out, qp], axis=1)


def _process_items(df, item_vid, item_tips):
    """Сырые объявления -> нормализованные тексты и разобранные параметры."""
    df = df.copy()
    for c in ["item_latitude", "item_longitude", "item_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["i_vid"] = df.item_infm_params_text.map(item_vid)
    df["i_tips"] = df.item_infm_params_text.map(item_tips)
    df["title_lem"] = df.item_title_raw.map(lemmas)
    df["params_lem"] = df.item_infm_params_text.map(_clean_params).map(lambda s: lemmas(s, 80))
    df["desc_lem"] = df.item_description_raw.str.slice(0, 1500).map(lambda s: lemmas(s, DESC_TOKENS))
    df["title_norm"] = df.item_title_raw.map(normalize)
    return df[ITEM_KEEP]


def source_signature(data_dir, train_rows=None):
    """Отпечаток исходных данных: имена и размеры файлов + train_rows. Изменились данные — кэш не годится."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    s = "|".join(f"{os.path.basename(f)}:{os.path.getsize(f)}" for f in files) + f"#rows={train_rows or 0}"
    return hashlib.md5(s.encode()).hexdigest()[:12]


def prepare(data_dir: str, cache_dir: str, train_rows=None, verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    f_items, f_pairs, f_bq = (os.path.join(cache_dir, n) for n in
                              ("items.parquet", "train_pairs.parquet", "bench_queries.parquet"))
    f_sig = os.path.join(cache_dir, "source_signature.txt")
    sig = source_signature(data_dir, train_rows)
    if all(os.path.exists(f) for f in (f_items, f_pairs, f_bq)):
        if not os.path.exists(f_sig):  # кэш от старой версии кода — считаем, что он от этих же данных
            open(f_sig, "w").write(sig)
        if open(f_sig).read().strip() == sig:
            return pd.read_parquet(f_items), pd.read_parquet(f_pairs), pd.read_parquet(f_bq)
    say = print if verbose else (lambda *a, **k: None)

    # --- проход 1 (дешёвый): только поля запроса и item_id -> пары train и словарь фильтров
    q = _read(data_dir, "benchmark_queries")
    bq = _prep_queries(q)
    bq.insert(0, "query_id", q.query_id.values)
    pairs = []
    for _, df in iter_train(data_dir, columns=Q_COLS + ["item_id"], train_rows=train_rows):
        p = _prep_queries(df)
        p["item_id"] = df.item_id.values
        pairs.append(p)
    pairs = pd.concat(pairs, ignore_index=True)
    say(f"пар train: {len(pairs)}")

    # словарь значений фильтров — из всех запросов (train + бенчмарк)
    vids = sorted((set(pairs.q_vid) | set(bq.q_vid)) - {""})
    tips = sorted({t for s in pd.concat([pairs.q_tips, bq.q_tips]) for t in s.split("|") if t})
    item_vid, item_tips = build_item_parsers(vids, tips)
    # «отпечаток» словаря фильтров: если данные изменятся, старые кэши кусков не подхватятся
    tag = hashlib.md5(("|".join(vids) + "#" + "|".join(tips)).encode()).hexdigest()[:8]
    prep_dir = os.path.join(cache_dir, "prep", tag)
    os.makedirs(prep_dir, exist_ok=True)

    # --- корпус бенчмарка
    f_bench = os.path.join(prep_dir, "bench_items.parquet")
    if not os.path.exists(f_bench):
        say("обработка корпуса бенчмарка...")
        it = _read(data_dir, "benchmark_items")
        it["in_bench"] = True
        _process_items(it, item_vid, item_tips).to_parquet(f_bench + ".tmp")
        os.replace(f_bench + ".tmp", f_bench)  # атомарная запись: не бывает «полузаписанных» файлов
        del it
        gc.collect()
    bench_ids = set(pd.read_parquet(f_bench, columns=["item_id"]).item_id)

    # --- проход 2: объявления из train, кусок за куском, с сохранением каждого куска
    item_cols = None
    for i, df in iter_train(data_dir, train_rows=train_rows):
        f_chunk = os.path.join(prep_dir, f"train_items_{i:03d}.parquet")
        if os.path.exists(f_chunk):
            continue  # уже обработан в прошлый запуск
        item_cols = item_cols or [c for c in df.columns if c.startswith("item_")]
        new = df[item_cols].drop_duplicates("item_id")
        new = new[~new.item_id.isin(bench_ids)].assign(in_bench=False)
        _process_items(new, item_vid, item_tips).to_parquet(f_chunk + ".tmp")
        os.replace(f_chunk + ".tmp", f_chunk)
        say(f"  кусок train {i}: {len(new)} новых объявлений")
        del df, new
        lemma.cache_clear()
        gc.collect()

    chunks = sorted(glob.glob(os.path.join(prep_dir, "train_items_*.parquet")))
    items = pd.concat([pd.read_parquet(f_bench)] + [pd.read_parquet(f) for f in chunks], ignore_index=True)
    items = items.drop_duplicates("item_id", keep="first").reset_index(drop=True)
    say(f"объявлений всего: {len(items)} (в корпусе бенчмарка: {int(items.in_bench.sum())})")

    items.to_parquet(f_items)
    pairs.to_parquet(f_pairs)
    bq.to_parquet(f_bq)
    open(f_sig, "w").write(sig)  # последним: подпись есть только у полностью готового кэша
    return items, pairs, bq
