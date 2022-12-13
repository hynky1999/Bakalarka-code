import json
import fasttext
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names = [
    "aktualne",
    "denik",
    "idnes",
    "ihned",
    "irozhlas",
    "novinky",
    "seznamzpravy",
]


def save_jsonb(l, filename, mode="w", show_progress=True):
    iterable = l
    if show_progress:
        iterable = tqdm(l)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, mode) as f:
        for item in iterable:
            f.write(json.dumps(item))
            f.write("\n")


def load_jsonb(filename, show_progress=True):
    total = 0
    if show_progress:
        total = num_of_lines(filename)
    with open(filename, "r") as f:
        iterable = f
        if show_progress:
            iterable = tqdm(f, total=total)
        for line in iterable:
            yield json.loads(line)


def num_of_lines(filename):
    with open(filename, "r") as f:
        return sum(1 for _ in f)


def articles_num(folder):
    for file in folder.iterdir():
        if file.is_file():
            yield file.name, num_of_lines(file)


# CZ checking
def filter_by_cz_lang(json_file, ratio=1.0, yield_false=False, article_only=False):
    for article in load_jsonb(json_file):
        content = article["content"]
        content_ratio = is_cz(content)
        data = None
        if content_ratio >= ratio:
            data = article, content_ratio, True

        elif yield_false:
            data = article, content_ratio, False
        if data is not None:
            if article_only:
                yield data[0]
            else:
                yield data


model = fasttext.load_model(str(Path(__file__).parent / "lid.176.ftz"))


def is_cz(article):
    lines = list(filter(lambda x: len(x) > 0, article.strip().split("\n")))
    if len(lines) == 0:
        return False

    czech_lines = [
        1 if model.predict(line, k=1)[0][0][9:] == "cs" else 0 for line in lines
    ]
    return sum(czech_lines) / len(lines)


## Misc
def create_handler(log_path, name, format_str):
    logg = logging.getLogger(str((log_path / name).absolute()))
    logg.handlers = []
    logg.setLevel(logging.ERROR)
    fh = logging.FileHandler(log_path / f"{name}.log", mode="w")
    fh.setFormatter(logging.Formatter(format_str))
    logg.addHandler(fh)
    return logg


def create_handlers(log_path, format_str):
    pth = Path("logs") / log_path
    pth.mkdir(parents=True, exist_ok=True)
    return {name: create_handler(pth, name, format_str) for name in names}


def show_outliers(df, col, threshold, mod, limit=100, random=False):

    if mod == "lower":
        tresholded = df[df[col] < threshold]
        asc = False
    else:
        tresholded = df[df[col] > threshold]
        asc = True

    if random:
        return tresholded.sample(limit)
    return tresholded.sort_values(col, ascending=asc).head(limit)


def show_outlier_by_percentiles(df, col, percentile, limit=100, random=False):
    if percentile > 1:
        percentile = percentile / 100

    side = "lower"
    if percentile > 0.5:
        side = "higher"
    return show_outliers(
        df, col, np.percentile(df[col], percentile * 100), side, limit, random
    )


def pick_indexes(indexes, file):
    for i, js in enumerate(load_jsonb(file)):
        if i in indexes:
            yield js


def show_df_lines(df, filename, mod=lambda x: x):
    index = df.index.to_list()
    chosen_data = pick_indexes(index, filename)
    for i in chosen_data:
        print(mod(i))


def get_unique(file, col):
    authors = dict()
    length = num_of_lines(file)
    for js in load_jsonb(file):
        if js[col] == None:
            continue

        for js_val in js[col]:
            val = authors.get(js_val, 0)
            authors[js_val] = val + 1
    return authors


get_unique_authors = lambda file: get_unique(file, "author")
get_unique_topics = lambda file: get_unique(file, "keywords")
