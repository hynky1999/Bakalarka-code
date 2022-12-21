import datetime
import json
from typing import List
from article_utils import published_date_to_date
from dataclasses import dataclass
from filtering import create_tokenized_filter
from enum import IntEnum
import numpy as np
import pandas as pd
import re


class Gender(IntEnum):
    MAN = 1
    WOMAN = 2
    MIXED = 3


@dataclass
class Article:
    url: str
    server: str
    headline: str
    brief: str | None
    content: str
    category: str | None
    authors: List[str] | None
    author_genders: List[Gender] | None
    date: str | None
    day: int | None


# AUTHORS
multispace = re.compile(r"\s+")


post_sub = ["připravila?", "přeložila?"]
post_sub_re = re.compile(r"(" + "|".join(post_sub) + r")\s+")


def postprocess_author(author):
    author = preprocess_author(author)
    author = author.lower()
    author = multispace.sub(" ", author)
    author_split = author.split(" ")
    author_split_capitalized = [x.capitalize() for x in author_split]
    author = " ".join(author_split_capitalized)
    return author


def postprocess_authors(js):
    js["author"] = (
        [postprocess_author(author) for author in js["author"]]
        if js["author"]
        else None
    )
    js["author_genders"] = (
        list(map(guess_gender, js["author"])) if js["author"] else None
    )
    return js


def guess_gender(author):
    if author == None:
        return None
    if author.lower().strip().endswith("á"):
        return Gender.WOMAN

    return Gender.MAN


def filter_author(js):
    authors = js["author"]
    if authors == None:
        return js

    new_authors = []
    for author in authors:
        preproc_auth = preprocess_author(author)
        if is_human_author(preproc_auth):
            new_authors.append(author)

    if len(new_authors) == 0:
        js["author"] = None

    js["author"] = new_authors
    return js


non_human_contain = [
    "redakce",
    "redaktor",
    "middlesearch",
    "center",
    "global",
    "naše",
    "investements",
    "telegraph",
    "parkhotel",
    "capital",
    "europe",
    "digest",
    "czech",
    "čtk",
    "usa",
    "chronicle",
    "story",
    "zeitung",
    "/",
    "(",
    ")",
    "škola",
    "seznam",
    "agency",
    "post",
    "s.r.o",
    "scientist",
    "washington",
    "banka",
    "manažer",
    "www",
    "journal",
    "komerč",
    "české",
    "akademie",
    "blue",
    "avon",
    "consultancy",
    "group",
    "times",
    "international",
    "ftv",
    "news",
    "rádio",
    "program",
    "poradce",
    "york",
    "broker",
    "čtenář",
    "materiál",
    "reporté",
    "návod",
    "tým",
    "český",
    "rozhlas",
    "press",
    "generál",
    "předseda",
    "mail",
    "novinky",
    "aktualne",
    "čr",
    "bank",
    "pojištovna",
    "idnes",
    "online",
    "deník",
    "časopis",
    "irozhlas",
    "společnost",
    "tv",
    "radio",
    "rádio",
    "swiss",
    "město",
]
contains_number = re.compile("[0-9]")


def preprocess_author(author):
    author = author.strip()
    return author.strip().split("|")[0]


def is_capitalized(word):
    return len(word) == 0 or (word[0].isupper() and word[1:].islower())


def is_upper(word):
    return word.isupper()


def is_human_author(author):
    splitted = list(filter(lambda x: len(x) > 0, author.split(" ")))
    non_whitespace_author = author.replace(" ", "")

    if len(splitted) > 4 or len(splitted) < 2:
        return False

    # Either all upper caps or capilized first letter
    if not all([is_capitalized(word) for word in splitted]) and not all(
        [is_upper(word) for word in splitted]
    ):
        return False

    # At least 2 letters in every word
    if not all([len(word) > 2 for word in splitted]):
        return False

    if not non_whitespace_author.isalnum():
        return False

    if contains_number.search(author):
        return False

    if any([x in author.lower() for x in non_human_contain]):
        return False

    return True


def cap_with_dot(headline):
    # Capitalize and adds dot
    if headline == None:
        return None

    headline = headline.strip()
    if len(headline) >= 2 and headline[-1] == "." and headline[-2] != ".":
        headline = headline[:-1]

    if len(headline) > 1 and headline[0].islower():
        headline = headline[0].upper() + headline[1:]

    return headline


# DATE
def postprocess_date(js):
    date = published_date_to_date(js["publication_date"])
    day = date.weekday() if date else None
    js["date"] = date
    js["day"] = day
    return js


# HEADLINE
def postprocess_headline(js):
    js["headline"] = cap_with_dot(js["headline"])
    return js


# BRIEF
def postprocess_brief(js):
    js["brief"] = cap_with_dot(js["brief"])
    return js


filters_category = [
    create_tokenized_filter(lambda x: len(x) <= 5, "category"),
    lambda x: len(x["category"]) <= 35,
]


def postprocess_category(js):
    category = cap_with_dot(js["category"])
    if category != None:
        # Toktok filter works with dict
        js["category"] = category
        if all([f(js) for f in filters_category]):
            js["category"] = js["category"].lower()

        else:
            js["category"] = None

    return js


def add_server(server):
    def add_server_inner(js):
        js["server"] = server
        return js

    return add_server_inner


def as_Article(js):
    return js


def LowerTopXWhiten(col, limit):
    def topX_inner(jss):
        colled = [js[col] for js in jss]
        as_pd = pd.Series(colled)
        selected_cats = as_pd.value_counts().head(limit)
        selected_rows = as_pd.isin(selected_cats.index)
        for js, selected in zip(jss, selected_rows):
            if not selected:
                js[col] = None

        return jss

    return topX_inner


class JSONArticleEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Gender):
            return obj.value

        if isinstance(obj, datetime.date):
            return str(obj)
        return json.JSONEncoder.default(self, obj)
