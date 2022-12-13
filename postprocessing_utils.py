from article_utils import published_date_to_date
from dataclasses import dataclass
from enum import Enum
import re


@dataclass
class Gender(Enum):
    MAN = 1
    WOMAN = 2


@dataclass
class Article:
    url: str
    server: str
    headline: str
    brief: str | None
    content: str
    category: str | None
    authors: str | None
    author_geders: Gender | None
    date: str | None
    day: int | None


# AUTHORS
multispace = re.compile(r"\s+")


post_sub = ["připravila?", "přeložila?"]
post_sub_re = re.compile(r"(" + "|".join(post_sub) + r")\s+")


def postprocess_author(author):
    author = author.lower()
    author = multispace.sub(" ", author)
    author_split = author.split(" ")
    author_split_capitalized = [x.capitalize() for x in author_split]
    author = " ".join(author_split_capitalized)
    return author


def postprocess_authors(js):
    js["author"] = [postprocess_author(author) for author in js["author"]]
    js["author_genders"] = list(map(guess_gender, js["author"]))
    return js


def guess_gender(author):
    if author == None:
        return None
    if author.endswith("ová"):
        return Gender.WOMAN

    # TODO: improve
    return Gender.MAN


def filter_author(js):
    authors = js["author"]
    if authors == None:
        return js

    new_authors = []
    for author in authors:
        preproc_auth = preprocess_author(author)
        if is_human_author(preproc_auth):
            new_authors.append(preproc_auth)

    if len(new_authors) == 0:
        js["author"] = None

    js["author"] = new_authors
    return js


non_human_contain = [
    "redakce",
    "redaktor",
    "middlesearch",
    "naše",
    "čtk",
    "usa",
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
    if headline[-1] == "." and headline[-2] != ".":
        headline = headline[:-1]

    if headline[0].islower():
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
    js["category"] = [cap_with_dot(x) for x in js["category"]]
    return js


def add_server(server):
    def add_server_inner(js):
        js["server"] = server
        return js

    return add_server_inner


def as_Article(js):
    return Article(
        url=js["url"],
        server=js["server"],
        headline=js["headline"],
        brief=js["brief"],
        content=js["content"],
        category=js["category"],
        authors=js["author"],
        author_geders=js["author_genders"],
        date=js["date"],
        day=js["day"],
    )
