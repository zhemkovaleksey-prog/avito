"""Нормализация текста: нижний регистр, ё->е, токенизация, лемматизация (pymorphy3) с кэшем."""
from __future__ import annotations

import re
from functools import lru_cache

import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")

# Служебные слова, которые не несут смысла для поиска услуг
STOP = set("""и в во на с со к ко по о об от до за из у для не ни а но или же ли бы то как что это
при под над без через про весь все всё свой который мы вы он она они я ты""".split())


@lru_cache(maxsize=2_000_000)
def lemma(tok: str) -> str:
    # Для чисел/латиницы pymorphy ничего полезного не даст — оставляем как есть
    if not tok.isalpha() or tok.isascii():
        return tok
    return _morph.parse(tok)[0].normal_form.replace("ё", "е")


def normalize(text: str) -> str:
    """Строка -> нормализованная строка (без лемматизации), для char n-gram."""
    if not isinstance(text, str):
        return ""
    return " ".join(_TOKEN_RE.findall(text.lower().replace("ё", "е")))


def lemmas(text: str, max_tokens: int | None = None) -> str:
    """Строка -> строка лемм через пробел, без стоп-слов."""
    toks = _TOKEN_RE.findall(text.lower().replace("ё", "е")) if isinstance(text, str) else []
    if max_tokens:
        toks = toks[:max_tokens]
    return " ".join(lemma(t) for t in toks if t not in STOP and len(t) > 1)
