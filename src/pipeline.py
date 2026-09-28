"""Этапы решения как отдельные функции — их вызывают и run_pipeline.py (CLI), и solution.ipynb.

Каждый этап сохраняет результат в cache_dir и при повторном вызове читает готовое:
прервался процесс — запускаем ещё раз, он продолжит с первого незавершённого этапа.
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import ranker
from data import prepare
from retriever import Retriever
from split import make_split

STAGE_FILES = dict(
    cand_hold="cand_holdout.parquet", hq="holdout_queries.pkl", base="baselines.json",
    model_v="ranker_val.txt", val_scored="val_scored.parquet", report="validation_report.json",
    model="ranker.txt", cand_bench="cand_bench.parquet",
)


def _save_parquet(df, path):
    df.to_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)  # атомарно: либо файла нет, либо он целый


def _save_json(obj, path):
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)


def label(cand, relevant):
    """Разметка кандидатов: 1, если объявление — одно из выбранных пользователем по этому запросу."""
    rel = relevant.values
    cand["label"] = np.fromiter((x in rel[i] for i, x in zip(cand.qi.values, cand.item_id.values, strict=True)),
                                dtype=np.int8, count=len(cand))
    return cand


def check_answer(ans: pd.DataFrame, bq: pd.DataFrame, corpus_ids: set):
    """Все требования к answer.csv из условия."""
    assert list(ans.columns) == ["query_id", "answer"]
    assert len(ans) == len(bq) and set(ans.query_id) == set(bq.query_id) and ans.query_id.is_unique
    for a in ans.answer:
        ids = a.split(" ")
        assert 0 < len(ids) <= 50 and len(ids) == len(set(ids))
        assert all(len(x) == 16 and x in corpus_ids for x in ids)


class Pipeline:
    def __init__(self, data_dir, cache_dir="cache", n_holdout=10000, train_rows=0, verbose=True):
        self.data_dir, self.C = data_dir, cache_dir
        self.n_holdout, self.train_rows = n_holdout, train_rows
        self.f = {k: os.path.join(cache_dir, v) for k, v in STAGE_FILES.items()}
        self.t0 = time.time()
        self.verbose = verbose
        os.makedirs(cache_dir, exist_ok=True)

    def log(self, msg):
        if self.verbose:
            print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)

    def _done(self, *keys):
        return all(os.path.exists(self.f[k]) for k in keys)

    # ------------------------------------------------------------------ 1. подготовка данных
    def prep(self):
        self.items, self.pairs, self.bq = prepare(self.data_dir, self.C, train_rows=self.train_rows,
                                                  verbose=self.verbose)
        # данные поменялись с прошлого запуска -> результаты следующих этапов устарели, удаляем
        sig = open(os.path.join(self.C, "source_signature.txt")).read().strip()
        f_stage_sig = os.path.join(self.C, "stages_signature.txt")
        if os.path.exists(f_stage_sig) and open(f_stage_sig).read().strip() != sig:
            for path in self.f.values():
                if os.path.exists(path):
                    os.remove(path)
            self.log("[prep] данные изменились — этапы 2-6 будут пересчитаны")
        open(f_stage_sig, "w").write(sig)
        self.loc = self.items[["item_id", "item_location_id"]]
        self.log(f"[prep] объявлений {len(self.items)}, пар train {len(self.pairs)}, запросов бенчмарка {len(self.bq)}")
        return self

    # ------------------------------------------------------------------ 2. кандидаты для валидации
    def val_candidates(self):
        if not self._done("cand_hold", "hq", "base"):
            # доля «знакомых» запросов в валидации = доля текстов бенчмарка, встречавшихся в train
            self.seen_share = float(self.bq.search_query.isin(set(self.pairs.search_query)).mean())
            fit_pairs, hq = make_split(self.pairs, n_holdout=self.n_holdout, seen_share=self.seen_share)
            fit_pairs = fit_pairs.merge(self.loc, on="item_id", how="left")
            corpus_val = self.items[self.items.in_bench | self.items.item_id.isin(set().union(*hq.relevant))]
            self.log(f"[val_cands] отложено {len(hq)} запросов, корпус {len(corpus_val)}")
            ret = Retriever().fit(corpus_val, fit_pairs, self.items)
            cand = label(ret.candidates(hq), hq.relevant)
            cand["part"] = hq.part.values[cand.qi.values]
            rel_val = hq.relevant[hq.part == "val"]
            base = {
                "val_seen_share": self.seen_share,
                "bm25_whole_corpus": ranker.recall_at_k(ret.bm25_baseline(hq, use_location=False), rel_val),
                "bm25_location_filter": ranker.recall_at_k(ret.bm25_baseline(hq, use_location=True), rel_val),
            }
            _save_parquet(cand, self.f["cand_hold"])
            hq.to_pickle(self.f["hq"])
            _save_json(base, self.f["base"])
            del ret, fit_pairs, corpus_val, cand
        self.cand = pd.read_parquet(self.f["cand_hold"])
        self.hq = pd.read_pickle(self.f["hq"])
        self.base = json.load(open(self.f["base"]))
        self.feats = ranker.feature_cols(self.cand)
        self.log(f"[val_cands] кандидатов {len(self.cand)}, в среднем {len(self.cand) / len(self.hq):.0f} на запрос")
        return self

    # ------------------------------------------------------------------ 3. ранкер + честная оценка
    def val_ranker(self):
        if not self._done("model_v", "report"):
            cand, hq = self.cand, self.hq
            model_v, best = ranker.train(cand[cand.part == "rank_train"], self.feats)
            model_v.save_model(self.f["model_v"])
            cv = cand[cand.part == "val"]
            sc = model_v.predict(cv[self.feats])
            rel_val = hq.relevant[hq.part == "val"]
            report = {
                "train_pairs_used": int(len(self.pairs)),
                "n_val_queries": int((hq.part == "val").sum()),
                **self.base,
                "candidate_pool_upper_bound": ranker.recall_at_k(cv.groupby("qi").item_id.apply(list), rel_val),
                "final_recall@50": ranker.recall_at_k(ranker.top_k(cv, sc), rel_val),
                "ranker_rounds": int(best),
            }
            imp = pd.Series(model_v.feature_importance("gain"), self.feats).sort_values(ascending=False)
            report["feature_importance"] = (imp / imp.sum()).round(4).to_dict()
            _save_parquet(cv.assign(score=sc), self.f["val_scored"])
            _save_json(report, self.f["report"])
        self.report = json.load(open(self.f["report"]))
        self.log("[val_ranker] Recall@50: " + json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in self.report.items() if k != "feature_importance"}, ensure_ascii=False))
        return self

    # ------------------------------------------------------------------ 4. финальный ранкер
    def final_ranker(self):
        if not self._done("model"):
            model, best_all = ranker.train(self.cand, self.feats)
            model.save_model(self.f["model"])
            self.log(f"[final_ranker] обучен ({best_all} деревьев)")
        self.model = lgb.Booster(model_file=self.f["model"])
        return self

    # ------------------------------------------------------------------ 5. кандидаты бенчмарка
    def bench_candidates(self):
        self.corpus = self.items[self.items.in_bench]
        if not self._done("cand_bench"):
            ret = Retriever().fit(self.corpus, self.pairs.merge(self.loc, on="item_id", how="left"), self.items)
            _save_parquet(ret.candidates(self.bq), self.f["cand_bench"])
            del ret
        self.cand_b = pd.read_parquet(self.f["cand_bench"])
        self.log(f"[bench_cands] {len(self.cand_b)} строк")
        return self

    # ------------------------------------------------------------------ 6. answer.csv
    def answer(self, out="answer.csv"):
        pred = ranker.top_k(self.cand_b, self.model.predict(self.cand_b[self.feats]))
        ans = pd.DataFrame({"query_id": self.bq.query_id.values,
                            "answer": [" ".join(pred.get(i, [])) for i in range(len(self.bq))]})
        check_answer(ans, self.bq, set(self.corpus.item_id))
        ans.to_csv(out, index=False)
        rep = dict(self.report, answer_rows=len(ans))
        _save_json(rep, os.path.join(os.path.dirname(os.path.abspath(out)), "validation_report.json"))
        self.log(f"[answer] готово: {out}")
        return ans

    def run(self, out="answer.csv"):
        self.prep().val_candidates().val_ranker()
        self.final_ranker()
        del self.cand  # освобождаем память перед этапом бенчмарка
        return self.bench_candidates().answer(out)
