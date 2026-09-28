"""Честная валидация: имитируем бенчмарк на train.

Бенчмарк похож на выборку РАЗНЫХ текстов запросов (средняя длина ~3 слова), и лишь часть его текстов
встречалась в train (seen_share, на полном train ~37%; считается по данным в pipeline.py).
Поэтому отложенные запросы сэмплируем равномерно по уникальным текстам, для доли (1 - seen_share)
из них убираем из обучения ВСЕ строки с этим текстом (как «новый» запрос), а для доли seen_share —
только саму отложенную группу (как «знакомый» запрос).
"""
import numpy as np
import pandas as pd

KEY = ["search_query", "search_location_id", "search_infm_params_text"]


def make_split(pairs: pd.DataFrame, n_holdout=10000, seen_share=0.37, seed=42):
    pairs = pairs.copy()
    pairs["g"] = pairs.groupby(KEY, sort=False).ngroup()
    rng = np.random.RandomState(seed)
    groups = pairs.drop_duplicates("g")
    texts = groups.search_query.unique()
    sel = rng.choice(texts, n_holdout, replace=False)
    hold_g = groups[groups.search_query.isin(sel)].groupby("search_query").sample(1, random_state=seed)
    seen = rng.rand(len(hold_g)) < seen_share
    unseen_texts = set(hold_g.search_query[~seen])
    drop = pairs.g.isin(hold_g.g) | pairs.search_query.isin(unseen_texts)
    fit_pairs = pairs[~drop]
    hold_pairs = pairs[pairs.g.isin(hold_g.g)]
    # запросы (по одному на группу) + множество релевантных объявлений
    hq = hold_pairs.groupby("g").agg(
        **{c: (c, "first") for c in hold_pairs.columns if c not in ("g", "item_id")},
        relevant=("item_id", lambda s: set(s)),
    ).reset_index()
    hq["part"] = np.where(rng.rand(len(hq)) < 2 / 3, "rank_train", "val")
    return fit_pairs, hq
