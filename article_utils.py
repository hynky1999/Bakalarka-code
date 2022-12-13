import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize.toktok import ToktokTokenizer
import functools
from datetime import datetime
from preprocess_utils import num_of_lines, load_jsonb, show_df_lines
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from datetime import date
import pytz


@dataclass
class Stats:
    url: str
    article_length: int
    headline_length: int | None
    brief_length: int | None
    brief_non_alpha_ratio: float | None
    num_words: int
    num_words_ratio: float
    num_words_per_line: float
    avg_word_length: float
    non_alpha: int
    non_alpha_ratio: float
    date: date | None


used_cols = [
    "article_length",
    "headline_length",
    "brief_length",
    "brief_non_alpha_ratio",
    "num_words",
    "num_words_ratio",
    "num_words_per_line",
    "avg_word_length",
    "non_alpha",
    "non_alpha_ratio",
]
plot_col = 3
fig_size = (20, 10)


# Articles

IMAGE_FOLDER = Path("images") / "analysis"
HIST_NAME = "hist.png"
WHISKER_NAME = "whisker.png"
BY_DATE_NAME = "by_date.png"


def create_hist_plots(df: pd.DataFrame, save: bool):
    rows = (len(used_cols) - 1) // plot_col + 1
    fig, axes = plt.subplots(rows, plot_col, figsize=fig_size)
    fig.suptitle(f"{df.Name} histogram plots")
    for d in range(len(used_cols)):
        d_row_idx = d // plot_col
        d_col_idx = d % plot_col
        ax = axes[d_row_idx][d_col_idx]
        df[used_cols[d]].hist(ax=ax, bins=150, legend=True)

    p = IMAGE_FOLDER / df.Name
    p.mkdir(parents=True, exist_ok=True)
    if save:
        fig.savefig(str((p / HIST_NAME).absolute()))


def create_whisker_plots(df: pd.DataFrame, save: bool):
    rows = (len(used_cols) - 1) // plot_col + 1
    fig, axes = plt.subplots(rows, plot_col, figsize=fig_size)
    fig.suptitle(f"{df.Name} whisker plots")
    for d in range(len(used_cols)):
        d_row_idx = d // plot_col
        d_col_idx = d % plot_col
        ax = axes[d_row_idx][d_col_idx]
        df.boxplot(used_cols[d], ax=ax)
    p = IMAGE_FOLDER / df.Name
    p.mkdir(parents=True, exist_ok=True)
    if save:
        fig.savefig(str((p / WHISKER_NAME).absolute()))


def create_date_plot(df: pd.DataFrame, save: bool):
    rows = (len(used_cols)) // plot_col + 1
    fig, axes = plt.subplots(rows, plot_col, figsize=fig_size)
    fig.suptitle(f"{df.Name} date plots")
    groupby_date = df.groupby("date")
    for d in range(len(used_cols)):
        d_row_idx = d // plot_col
        d_col_idx = d % plot_col
        ax = axes[d_row_idx][d_col_idx]
        groupby_date[used_cols[d]].mean().plot.area(ax=ax, legend=True)

    d_row_idx = len(used_cols) // plot_col
    d_col_idx = len(used_cols) % plot_col
    ax = axes[d_row_idx][d_col_idx]
    groupby_date["url"].count().plot.area(ax=ax, legend=True)
    p = IMAGE_FOLDER / df.Name
    p.mkdir(parents=True, exist_ok=True)
    if save:
        fig.savefig(str((p / BY_DATE_NAME).absolute()))


def create_exploratory_plots(df: pd.DataFrame, save=False):
    create_hist_plots(df, save=save)
    create_whisker_plots(df, save=save)
    create_date_plot(df, save=save)


toktok = ToktokTokenizer()


def flatten(l):
    return [item for sublist in l for item in sublist]


def published_date_to_date(date: str | None):
    if date is None:
        return None

    date_from = datetime.fromisoformat(date)
    if date_from.tzinfo == None:
        date_from = date_from.replace(tzinfo=pytz.UTC)

    date_date = date_from.date()
    return date_date


def get_statistics(js: dict):
    url = js["url"]
    article = js["content"].strip()
    brief = js["brief"].strip() if js["brief"] else None
    headline = js["headline"].strip()
    article_length = len(article)
    brief_length = len(brief) if brief else None
    headline_length = len(headline)

    # Tok tok is speedy unlike the others
    tokenized = toktok.tokenize(article)
    num_words = len(tokenized)
    num_words_ratio = num_words / article_length
    num_words_per_line = num_words / len(article.split("\n"))
    avg_word_length = get_average_word_length(tokenized)
    non_alpha = count_non_alpha(tokenized)
    brief_non_alpha_ratio = (
        count_non_alpha(toktok.tokenize(brief)) / article_length if brief else None
    )
    non_alpha_ratio = non_alpha / article_length
    date = published_date_to_date(js["publication_date"])
    return Stats(
        url,
        article_length,
        headline_length,
        brief_length,
        brief_non_alpha_ratio,
        num_words,
        num_words_ratio,
        num_words_per_line,
        avg_word_length,
        non_alpha,
        non_alpha_ratio,
        date,
    )


@functools.cache
def create_df(file: Path):
    l = []
    for js in load_jsonb(file):
        l.append(
            get_statistics(js).__dict__,
        )

    df = pd.DataFrame(l)
    df.Name = file.name

    return df


def get_average_word_length(tokenized_article):
    return sum([len(x) for x in tokenized_article]) / len(tokenized_article)


def count_non_alpha(article):
    # Should new line also count ?
    return sum([1 for char in article if not char.isalnum()])


### INSPECT TOOLS


def inspect_drop_date(
    df,
    col,
    file,
    mod,
    start: datetime,
    end: datetime,
    middle: datetime | None = None,
    type: str = "up",
    num=10,
):
    start_date = start.date()
    end_date = end.date()
    middle_date = middle.date() if middle else None
    dates_adjusted = df[(df["date"] > start_date) & (df["date"] < end_date)]

    dates_adjusted.groupby("date").mean()[col].plot()

    if middle_date:
        asc = False if type == "up" else True

        left = (
            dates_adjusted[dates_adjusted["date"] < middle_date]
            .sort_values(col, ascending=asc)
            .head(num)
        )

        right = (
            dates_adjusted[dates_adjusted["date"] > middle_date]
            .sort_values(col, ascending=asc)
            .head(num)
        )

        show_df_lines(left, file, mod)
        print("TAIL\n\n\n")
        show_df_lines(right, file, mod)
