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
    MAN = 0
    WOMAN = 1
    MIXED = 2


def filter_by_list(col, filter_list, lower=False):
    l = filter_list
    if lower:
        l = [x.lower() for x in filter_list]

    def filter_by_list_inner(js):
        val = js[col]
        if val is None:
            return js

        val = val.lower() if lower else val
        if not val in l:
            js[col] = None
        return js

    return filter_by_list_inner


def translate(col, translate_dict, lower=False):
    def translate_inner(js):
        if js[col] is None:
            return js
        val = js[col].lower() if lower else js[col]
        if val in translate_dict:
            js[col] = translate_dict[val]

        return js

    return translate_inner


@dataclass
class Article:
    url: str
    server: str
    headline: str
    brief: str | None
    content: str
    category: str | None
    authors: List[str] | None
    authors_gender: List[Gender] | None
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
    js["authors_gender"] = (
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


def postprocess_category(js):
    category = js["category"]
    if category != None:
        category = category.lower()

    category = cap_with_dot(category)
    js["category"] = category
    return js


def add_server(server):
    def add_server_inner(js):
        js["server"] = server
        return js

    return add_server_inner


def add_cum_gender(js):
    genders = js["authors_gender"]
    g_type: Gender | None = None
    if genders is not None:
        for g in Gender:
            if all(gender == g for gender in genders):
                g_type = g
                break
        if g_type == None:
            g_type = Gender.MIXED
    js["cum_gender"] = g_type
    return js


def as_Article(js):
    return js


class JSONArticleEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Gender):
            return obj.value

        if isinstance(obj, datetime.date):
            return str(obj)
        return json.JSONEncoder.default(self, obj)
