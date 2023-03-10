from dataclasses import dataclass
from functools import cache
from typing import Any, Callable, List
from article_utils import get_statistics
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd

from preprocess_utils import is_cz

toktok = ToktokTokenizer()


def between(a, b, keep_na=True):
    def _between(x):
        if x is None:
            return keep_na
        if a is not None and x < a:
            return False

        if b is not None and x > b:
            return False

        return True

    return _between


def create_config(config, default={}):
    cfg = default.copy()
    cfg.update(config)
    return cfg


def create_filter(filter, fc: Callable[[Any, Any], bool], lower=True):
    if lower:
        filter = [f.lower() for f in filter]

    def _filter(head):
        if head is None:
            return True
        head = head.lower()
        return not any(fc(head, x) for x in filter)

    return _filter

def create_filter_by_stats(config):
    def filter_js(df):
        stats = get_statistics(df)
        return all(fc(stats.__dict__[key]) for key, fc in config.items())

    return filter_js


def create_tokenized_filter(fc: Callable[[List[str]], bool], col):
    def filter_toktok(js: dict) -> dict:
        data = js[col]
        if data is None:
            return js

        tokenized: List[str] = toktok.tokenize(data)
        if not fc(tokenized):
            js[col] = None

        return js

    return filter_toktok


def create_filter_by_cz_lang(ratio=1.0):
    def filter_cz(content):
        return is_cz(content) >= ratio

    return filter_cz
