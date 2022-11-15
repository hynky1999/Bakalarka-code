import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize.toktok import ToktokTokenizer
import functools
from datetime import datetime
from tqdm import tqdm
from preprocess_utils import num_of_lines, load_jsonb
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


def create_hist_plots(df: pd.DataFrame):
    rows = (len(used_cols) - 1) // plot_col + 1
    fig, axes = plt.subplots(rows, plot_col, figsize=fig_size)
    fig.suptitle(f"{df.Name} histogram plots")
    for d in range(len(used_cols)):
        d_row_idx = d // plot_col
        d_col_idx = d % plot_col
        ax = axes[d_row_idx][d_col_idx]
        df[used_cols[d]].hist(ax=ax, bins=150, legend=True)


def create_whisker_plots(df: pd.DataFrame):
    rows = (len(used_cols) - 1) // plot_col + 1
    fig, axes = plt.subplots(rows, plot_col, figsize=fig_size)
    fig.suptitle(f"{df.Name} whisker plots")
    for d in range(len(used_cols)):
        d_row_idx = d // plot_col
        d_col_idx = d % plot_col
        ax = axes[d_row_idx][d_col_idx]
        df.boxplot(used_cols[d], ax=ax)


def create_date_plot(df: pd.DataFrame):
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


def create_exploratory_plots(df):
    create_hist_plots(df)
    create_whisker_plots(df)
    create_date_plot(df)


toktok = ToktokTokenizer()


def flatten(l):
    return [item for sublist in l for item in sublist]


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
        l.append(
            [
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
