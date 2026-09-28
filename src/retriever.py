"""Кандидатогенерация: несколько независимых источников сигналов + сбор признаков для ранкера.

Идея: для каждого запроса
  1) ограничиваем корпус «разрешёнными» локациями: своя локация + куда люди из этой локации
     реально уходят по train (>=1% выборов) + соседние локации в радиусе 60 км;
  2) считаем несколько текстовых скоров (BM25 по заголовку, BM25 по полному тексту,
     char n-gram TF-IDF, расширение запроса через похожие запросы из train) и
     «категорийный» скор (вероятность подкатегории объявления по тексту запроса);
  3) из каждого источника берём топ-N, объединяем, добавляем немного лучших по тексту из ВСЕГО
     корпуса (без фильтра локации) — это пул кандидатов (~400 шт., полнота ~0.96);
  4) для пула собираем признаки, дальше LightGBM выбирает финальные 50.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize as l2norm

from textproc import lemmas, normalize


def _haversine_rad(la1, lo1, la2, lo2):
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- BM25
class BM25:
    """BM25 через разреженные матрицы: считаем вес каждого (документ, слово) заранее,
    тогда скор запроса = X @ q_binary. Это ровно формула Okapi BM25."""

    def __init__(self, k1=1.2, b=0.75, min_df=1):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(token_pattern=r"\S+", min_df=min_df, dtype=np.float32)

    def fit_transform(self, docs):
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * self.idf[tf.indices]
        self.X = tf
        return self

    def query_matrix(self, qdocs):
        Q = self.vec.transform(qdocs)
        Q.data[:] = 1.0  # в коротких запросах повтор слова ничего не значит
        return Q.astype(np.float32)


def _topk(scores, k):
    """Индексы top-k (без сортировки) для 1D массива; нулевые/отрицательные скоры не берём."""
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[scores[idx] > 0]


class Retriever:
    def __init__(self, n_title=150, n_full=150, n_char=100, n_exp=150, n_cat=100, n_mem=30,
                 n_glob_full=60, n_glob_exp=40, geo_km=60):
        self.n = dict(title=n_title, full=n_full, char=n_char, exp=n_exp, cat=n_cat, mem=n_mem)
        self.n_glob = dict(full=n_glob_full, exp=n_glob_exp)
        self.geo_km = geo_km

    # ------------------------------------------------------------------ обучение
    def fit(self, items: pd.DataFrame, fit_pairs: pd.DataFrame, all_items: pd.DataFrame):
        """items — корпус, в котором ищем; fit_pairs — train-пары (запрос, выбранное объявление),
        из которых учим статистики; all_items — все известные объявления (для текстов из train)."""
        self.items = items.reset_index(drop=True)
        it = self.items
        N = len(it)
        self.item_index = pd.Series(np.arange(N), index=it.item_id)

        # --- текстовые индексы по корпусу
        self.bm_title = BM25(k1=1.2, b=0.5).fit_transform(it.title_lem)
        full = (it.title_lem + " ") * 2 + it.params_lem + " " + it.desc_lem  # заголовок важнее — дублируем
        self.bm_full = BM25(k1=1.5, b=0.75, min_df=2).fit_transform(full)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, sublinear_tf=True,
                                    dtype=np.float32)
        self.Xc = self.char.fit_transform(it.title_norm)

        # --- локации: куда реально уходят пользователи из search_location_id (по train)
        lt = fit_pairs.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        lt["p"] = lt.n / lt.groupby("search_location_id").n.transform("sum")
        self.loc_trans = lt
        self.loc_p = {(a, b): p for a, b, p in
                      lt[["search_location_id", "item_location_id", "p"]].itertuples(index=False)}
        self.loc_items = {loc_id: np.asarray(ix) for loc_id, ix in it.groupby("item_location_id").indices.items()}

        # --- классификатор «запрос -> подкатегория (microcat)»
        fp = fit_pairs.merge(all_items[["item_id", "item_microcat_id"]], on="item_id", how="left")
        fp = fp.dropna(subset=["item_microcat_id"])
        fp["q_lem"] = fp.search_query.map(lemmas)
        fp["q_norm"] = fp.search_query.map(normalize)
        # один пример на уникальный (текст, подкатегория) с весом = частота: быстрее и без перекоса к топ-запросам
        agg = fp.groupby(["q_lem", "q_norm", "item_microcat_id"]).size().rename("w").reset_index()
        # словари ограничены 150 тыс. признаков: веса SGD = (~750 классов x признаки) во float64 —
        # без ограничения на полном train это ~2.7 ГБ, а редкие n-граммы почти ничего не дают
        self.q_word = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                      max_features=150_000, dtype=np.float32)
        self.q_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                      max_features=150_000, dtype=np.float32)
        Xq = sp.hstack([self.q_word.fit_transform(agg.q_lem), self.q_char.fit_transform(agg.q_norm)]).tocsr()
        self.clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=15, tol=None, random_state=42)
        self.clf.fit(Xq, agg.item_microcat_id.astype(int), sample_weight=np.log1p(agg.w))
        self.mc_classes = pd.Series(np.arange(len(self.clf.classes_)), index=self.clf.classes_)
        it_mc = it.item_microcat_id.map(self.mc_classes)
        self.item_mc_col = it_mc.fillna(-1).astype(int).values

        # --- похожие запросы из train: для расширения запроса и «памяти» (кого выбирали раньше)
        tq = fp.groupby("q_norm").agg(q_lem=("q_lem", "first")).reset_index()
        self.tq = tq
        self.tq_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True,
                                       dtype=np.float32)
        self.Xtq = self.tq_char.fit_transform(tq.q_norm)
        # для каждого прошлого запроса — «мешок слов» заголовков выбранных объявлений
        fp2 = fp.merge(all_items[["item_id", "title_lem"]], on="item_id", how="left")
        titles = fp2.groupby("q_norm").title_lem.apply(lambda s: " ".join(s.fillna(""))).reindex(tq.q_norm)
        T = self.bm_title.vec.transform(titles.fillna("").values).astype(np.float32)
        self.T_exp = l2norm(T.log1p())  # словарь = словарь заголовков корпуса
        # память: какие объявления корпуса выбирали по прошлым запросам
        mem = fp[fp.item_id.isin(self.item_index.index)]
        rows = mem.q_norm.map(pd.Series(np.arange(len(tq)), index=tq.q_norm)).values
        cols = self.item_index[mem.item_id].values
        self.M_mem = sp.csr_matrix((np.ones(len(mem), np.float32), (rows, cols)), shape=(len(tq), N))

        # --- география. Координаты локации поиска: медиана координат объявлений этой локации;
        # если локация «региональная» (объявлений с ней нет) — медиана координат выбранных из неё объявлений.
        ll = all_items.groupby("item_location_id")[["item_latitude", "item_longitude"]].median().dropna()
        fpc = fit_pairs.merge(all_items[["item_id", "item_latitude", "item_longitude"]], on="item_id")
        reg = fpc.groupby("search_location_id")[["item_latitude", "item_longitude"]].median().dropna()
        self.loc_coord = pd.concat([ll, reg[~reg.index.isin(ll.index)]])
        self.loc_coord.columns = ["lat", "lon"]
        # «размытость» локации поиска: доля самой частой локации выбранных объявлений (у «Всей России» маленькая)
        self.loc_top_share = lt.groupby("search_location_id").p.max().to_dict()
        # соседние локации корпуса в радиусе GEO_KM
        cl = self.loc_coord[self.loc_coord.index.isin(self.loc_items.keys())]
        self._corpus_locs = cl.index.values
        self._corpus_loc_rad = np.radians(cl[["lat", "lon"]].values)
        self.item_lat = it.item_latitude.values.astype(np.float64)
        self.item_lon = it.item_longitude.values.astype(np.float64)

        # --- априорные признаки объявления
        pop = fit_pairs.item_id.value_counts()
        self.item_pop = it.item_id.map(pop).fillna(0).values.astype(np.float32)
        mc_pop = fp.item_microcat_id.value_counts()
        self.mc_pop = it.item_microcat_id.map(mc_pop).fillna(0).values.astype(np.float32)
        return self

    # ------------------------------------------------------------------ кандидаты + признаки
    def allowed_items(self, loc):
        """Индексы объявлений в разрешённых локациях: своя + те, куда ушло >=1% выборов из этой локации."""
        if not hasattr(self, "_allowed_cache"):
            self._allowed_cache = {}
            lt = self.loc_trans[self.loc_trans.p >= 0.01]
            self._loc_targets = lt.groupby("search_location_id").item_location_id.apply(set).to_dict()
        if loc in self._allowed_cache:
            return self._allowed_cache[loc]
        locs = {loc} | self._loc_targets.get(loc, set())
        if loc in self.loc_coord.index:  # + все локации корпуса в радиусе geo_km
            p = np.radians(self.loc_coord.loc[loc].values)
            d = _haversine_rad(p[0], p[1], self._corpus_loc_rad[:, 0], self._corpus_loc_rad[:, 1])
            locs |= set(self._corpus_locs[d <= self.geo_km])
        parts = [self.loc_items[loc_id] for loc_id in locs if loc_id in self.loc_items]
        if not parts:  # незнакомая локация и в корпусе её нет — ищем по всему корпусу
            res = (np.arange(len(self.items)), False)
        else:
            res = (np.unique(np.concatenate(parts)), True)
        self._allowed_cache[loc] = res
        return res

    def candidates(self, queries: pd.DataFrame, batch=64):
        """queries: search_query, search_location_id, q_vid, q_tips, q_rating4, search_infm_params_text.
        Возвращает DataFrame кандидатов (строка = пара запрос-объявление) с признаками."""
        q = queries.reset_index(drop=True)
        q_lem = q.search_query.map(lemmas).values
        q_norm = q.search_query.map(normalize).values
        Qt = self.bm_title.query_matrix(q_lem)
        Qf = self.bm_full.query_matrix(q_lem)
        Qc = self.char.transform(q_norm)
        Xq = sp.hstack([self.q_word.transform(q_lem), self.q_char.transform(q_norm)]).tocsr()
        P_mc = self.clf.predict_proba(Xq).astype(np.float32)
        S_tq = (self.tq_char.transform(q_norm) @ self.Xtq.T).tocsr()  # похожесть на прошлые запросы

        out = []
        for b0 in range(0, len(q), batch):
            chunk = []
            sl = slice(b0, b0 + batch)
            st = (Qt[sl] @ self.bm_title.X.T).toarray()
            sf = (Qf[sl] @ self.bm_full.X.T).toarray()
            sc = (Qc[sl] @ self.Xc.T).toarray()
            # расширение: берём топ-20 похожих прошлых запросов, взвешиваем их «мешки заголовков»
            W = S_tq[sl].toarray()
            W[W < 0.3] = 0
            for r in range(W.shape[0]):
                if (W[r] > 0).sum() > 20:
                    W[r][W[r] < np.sort(W[r])[-20]] = 0
            W = sp.csr_matrix(W)
            E = W @ self.T_exp                          # вектор расширения в словаре заголовков
            se = (E @ self.bm_title.X.T).toarray()
            sm = (W @ self.M_mem).toarray()             # память
            for r in range(st.shape[0]):
                i = b0 + r
                allowed, loc_ok = self.allowed_items(q.search_location_id[i])
                pmc = P_mc[i]
                mcs = np.where(self.item_mc_col[allowed] >= 0, pmc[np.maximum(self.item_mc_col[allowed], 0)], 0)
                src = {
                    "title": st[r, allowed], "full": sf[r, allowed], "char": sc[r, allowed],
                    "exp": se[r, allowed], "mem": sm[r, allowed],
                    # категорийный источник: вероятность подкатегории + чуть-чуть популярности (разбить ничьи)
                    "cat": mcs * (1 + 0.01 * np.log1p(self.item_pop[allowed] + 0.1 * self.mc_pop[allowed])),
                }
                pool = np.unique(np.concatenate([_topk(v, self.n[k]) for k, v in src.items()]))
                gi = allowed[pool]
                # глобальный источник: лучшие по тексту во ВСЁМ корпусе, без фильтра по локации
                # (ловит соседние города и запросы «по всей России»; решение, брать ли их, — за ранкером)
                glob = np.unique(np.concatenate([_topk(sf[r], self.n_glob["full"]), _topk(se[r], self.n_glob["exp"])]))
                glob = np.setdiff1d(glob, gi)
                gi = np.concatenate([gi, glob])
                f = {"title": st[r, gi], "full": sf[r, gi], "char": sc[r, gi], "exp": se[r, gi], "mem": sm[r, gi]}
                mc_all = np.where(self.item_mc_col[gi] >= 0, pmc[np.maximum(self.item_mc_col[gi], 0)], 0)
                f["cat"] = mc_all * (1 + 0.01 * np.log1p(self.item_pop[gi] + 0.1 * self.mc_pop[gi]))
                f["mc_prob"] = mc_all
                f["mc_rank"] = (-pmc).argsort().argsort()[np.maximum(self.item_mc_col[gi], 0)]
                f["mc_rank"] = np.where(self.item_mc_col[gi] >= 0, f["mc_rank"], 999)
                df = pd.DataFrame(f)
                df["qi"] = i
                df["item_idx"] = gi
                df["from_glob"] = np.r_[np.zeros(len(gi) - len(glob), np.int8), np.ones(len(glob), np.int8)]
                df["loc_ok"] = int(loc_ok)
                df["pool_size"] = len(allowed)
                # нормированные версии текстовых скоров внутри запроса (сравнимость между запросами)
                for k in ["title", "full", "char", "exp"]:
                    m = src[k].max()
                    df[k + "_rel"] = df[k] / m if m > 0 else 0.0
                chunk.append(df)
            # признаки считаем сразу для батча и храним во float32 — так пик памяти в разы меньше
            out.append(self._add_features(pd.concat(chunk, ignore_index=True), q))
            del st, sf, sc, se, sm, W, E, chunk
        return pd.concat(out, ignore_index=True)

    def bm25_baseline(self, queries: pd.DataFrame, k=50, use_location=True):
        """Бейзлайн для сравнения: только BM25 по полному тексту (± фильтр по локации)."""
        q = queries.reset_index(drop=True)
        Qf = self.bm_full.query_matrix(q.search_query.map(lemmas).values)
        ids = self.items.item_id.values
        res = {}
        for b0 in range(0, len(q), 256):
            sf = (Qf[b0:b0 + 256] @ self.bm_full.X.T).toarray()
            for r in range(sf.shape[0]):
                i = b0 + r
                allowed = self.allowed_items(q.search_location_id[i])[0] if use_location else np.arange(len(ids))
                top = allowed[_topk(sf[r, allowed], k)]
                res[i] = list(ids[top])
        return pd.Series(res)

    def _add_features(self, cand, q):
        it = self.items
        gi = cand.item_idx.values
        qi = cand.qi.values
        s_loc = q.search_location_id.values[qi]
        i_loc = it.item_location_id.values[gi]
        cand["same_loc"] = (s_loc == i_loc).astype(np.int8)
        cand["loc_p"] = [self.loc_p.get((a, b), 0.0) for a, b in zip(s_loc, i_loc, strict=True)]
        lat = pd.Series(s_loc).map(self.loc_coord.lat).values
        lon = pd.Series(s_loc).map(self.loc_coord.lon).values
        cand["dist_km"] = _haversine_rad(np.radians(lat), np.radians(lon),
                                         np.radians(self.item_lat[gi]), np.radians(self.item_lon[gi]))
        cand["loc_top_share"] = pd.Series(s_loc).map(self.loc_top_share).values
        qv, qt = q.q_vid.values[qi], q.q_tips.values[qi]
        iv, itp = it.i_vid.values[gi], it.i_tips.values[gi]
        # -1 = фильтра нет, 1 = объявление ему соответствует, 0 = не соответствует
        cand["vid_match"] = np.where(qv == "", -1, (qv == iv).astype(int))
        # типов в фильтре может быть несколько: совпадение, если у объявления есть хотя бы один из них
        cand["tip_match"] = np.where(qt == "", -1, [int(bool(set(a.split("|")) & set(b.split("|")) - {""}))
                                                    for a, b in zip(qt, itp, strict=True)])
        r = it.item_rating.values[gi]
        cand["rating"] = r
        cand["rating_ok"] = np.where(q.q_rating4.values[qi] == 1, (np.nan_to_num(r) >= 4).astype(int), -1)
        cand["log_reviews"] = np.log1p(np.nan_to_num(it.item_rating_reviews_count.values[gi]))
        cand["has_price"] = (~np.isnan(it.item_price.values[gi])).astype(np.int8)
        cand["item_pop"] = self.item_pop[gi]
        cand["mc_pop"] = np.log1p(self.mc_pop[gi])
        cand["q_len"] = q.search_query.str.split().str.len().values[qi]
        cand["q_has_params"] = (q.search_infm_params_text.values[qi] != "").astype(np.int8)
        cand["cat_ok"] = (it.item_category_id.values[gi] == 114).astype(np.int8)
        cand["item_id"] = it.item_id.values[gi]
        fl = cand.select_dtypes("float64").columns
        cand[fl] = cand[fl].astype(np.float32)  # экономим память
        ints = [c for c in cand.select_dtypes("int64").columns if c not in ("qi", "item_idx")]
        cand[ints] = cand[ints].astype(np.int32)
        return cand
