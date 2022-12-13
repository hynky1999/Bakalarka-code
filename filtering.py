from article_utils import get_statistics
from enum import Enum

from preprocess_utils import is_cz


def between(a, b):
    def _between(x):
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


def create_filter_by_stats(config):
    def filter_js(js):
        stats_dict = get_statistics(js)
        if all(config[key](stats_dict.__dict__[key]) for key in config):
            return True

        return False

    return filter_js


def create_filter_by_cz_lang(ratio=1.0):
    def filter_cz(js):
        return is_cz(js["content"]) >= ratio

    return filter_cz
