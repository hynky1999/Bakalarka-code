import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize.toktok import ToktokTokenizer
import functools
from datetime import datetime
from tqdm import tqdm
from preprocess_utils import num_of_lines, load_jsonb, show_df_lines
from pathlib import Path
import pytz


used_cols = [
    "article length",
    "headline length",
    "brief length",
    "num words",
    "num words ratio",
    "num words per line",
    "avg word length",
    "non-alpha",
    "non-alpha ratio",
]
plot_col = 3
fig_size = (20, 10)


# Articles

IMAGE_FOLDER = Path("images") / "analysis"


def create_hist_plots(df: pd.DataFrame, save):
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
        fig.savefig(str((p / "histograms.png").absolute()))


def create_whisker_plots(df: pd.DataFrame, save):
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
        fig.savefig(str((p / "whisker.png").absolute()))


def create_date_plot(df: pd.DataFrame, save):
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
        fig.savefig(str((p / "dates.png").absolute()))


def create_exploratory_plots(df, save=False):
    create_hist_plots(df, save=save)
    create_whisker_plots(df, save=save)
    create_date_plot(df, save=save)


toktok = ToktokTokenizer()


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_statistics(js):
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
    non_alpha_ratio = non_alpha / article_length
    date = (
        datetime.fromisoformat(js["publication_date"])
        if js["publication_date"] != None
        else None
    )
    if date != None and date.tzinfo == None:
        date = date.replace(tzinfo=pytz.UTC)

    if date != None:
        date = date.date()
    comments_num = js["comments_num"]
    return (
        url,
        article_length,
        headline_length,
        brief_length,
        num_words,
        num_words_ratio,
        num_words_per_line,
        avg_word_length,
        non_alpha,
        non_alpha_ratio,
        date,
        comments_num,
    )


@functools.cache
def create_df(file):
    length = num_of_lines(file)
    header = [
        "url",
        "article length",
        "headline length",
        "brief length",
        "num words",
        "num words ratio",
        "num words per line",
        "avg word length",
        "non-alpha",
        "non-alpha ratio",
        "date",
        "comments_num",
    ]
    l = []
    for js in tqdm(load_jsonb(file), total=length):
        l.append(
            [
                *get_statistics(js),
            ]
        )

    df = pd.DataFrame(l, columns=header)
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
