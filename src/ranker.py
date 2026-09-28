"""Второй шаг внутри кандидатогенерации: LightGBM (lambdarank) выбирает 50 лучших из пула ~400.

Почему ранкер, а не просто сумма скоров: сигналов много (BM25, похожесть символов, категория,
расстояние, фильтры) и их вклад нелинейный — например, «соседний город» полезен только когда
текст совпадает очень хорошо. Подбирать веса руками долго и хуже, чем выучить их на train.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

NON_FEATURES = {"qi", "item_idx", "item_id", "label", "part"}

PARAMS = dict(
    objective="lambdarank", metric="ndcg", eval_at=[50], lambdarank_truncation_level=60,
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42,
    deterministic=True, force_row_wise=True, num_threads=4,  # воспроизводимость результата
)


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def _dataset(df, feats):
    df = df.sort_values("qi", kind="stable")
    return lgb.Dataset(df[feats], df.label, group=df.groupby("qi", sort=True).size().values), df


def train(cand: pd.DataFrame, feats, es_share=0.15, max_rounds=1000, seed=42):
    """Обучение с ранней остановкой на части обучающих запросов (валидация не используется)."""
    # запросы без единого релевантного в пуле ранкеру ничего не дают
    cand = cand[cand.groupby("qi").label.transform("max") > 0]
    qs = np.sort(cand.qi.unique())
    rng = np.random.RandomState(seed)
    es_q = set(rng.choice(qs, int(len(qs) * es_share), replace=False))
    tr, es = cand[~cand.qi.isin(es_q)], cand[cand.qi.isin(es_q)]
    dtr, _ = _dataset(tr, feats)
    des, _ = _dataset(es, feats)
    m = lgb.train(PARAMS, dtr, max_rounds, valid_sets=[des],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    best = m.best_iteration
    # финальная модель — на всех обучающих запросах с найденным числом деревьев
    dall, _ = _dataset(cand, feats)
    return lgb.train(PARAMS, dall, best), best


def top_k(cand: pd.DataFrame, scores, k=50):
    """Для каждого запроса — k лучших item_id (при равенстве скоров порядок стабилен)."""
    df = cand[["qi", "item_id"]].assign(s=scores)
    df = df.sort_values(["qi", "s", "item_id"], ascending=[True, False, True], kind="stable")
    return df.groupby("qi").head(k).groupby("qi").item_id.apply(list)


def recall_at_k(pred: pd.Series, relevant: pd.Series):
    """pred: qi -> list(item_id); relevant: qi -> set(item_id). Recall@K как в условии."""
    r = [len(set(pred.get(i, [])) & rel) / len(rel) for i, rel in relevant.items()]
    return float(np.mean(r))
