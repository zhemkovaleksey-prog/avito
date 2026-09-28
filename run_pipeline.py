"""Полный пайплайн: данные -> валидация -> обучение ранкера -> answer.csv.

Запуск:
    python run_pipeline.py --data_dir path/to/NLP_avito_interns-dataset --out answer.csv

Каждый этап сохраняет результат в cache/ и при повторном запуске пропускается —
если процесс прервался, достаточно запустить команду ещё раз. Этапы описаны в src/pipeline.py,
пошаговый разбор с пояснениями — в solution.ipynb.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from pipeline import Pipeline  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--out", default="answer.csv")
    p.add_argument("--n_holdout", type=int, default=10000)
    p.add_argument("--train_rows", type=int, default=0, help="0 = весь train")
    a = p.parse_args()
    Pipeline(a.data_dir, a.cache_dir, a.n_holdout, a.train_rows).run(a.out)
