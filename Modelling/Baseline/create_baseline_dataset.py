from typing import List
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import partial
from pathlib import Path
from datasets import load_dataset, Dataset
from stop_words import get_stop_words
from datetime import datetime
from sacremoses import MosesTokenizer
from baseline_utils import dummy, preprocess_tokenized
import argparse
import scipy
import pickle
import pandas as pd

def get_tokenizer(tokenizer):
    print(f"Using tokenizer {tokenizer}")
    match tokenizer:
        case "moses":
            return MosesTokenizer('cs')

    raise ValueError(f"Unknown tokenizer {tokenizer}")

def tokenize_batch(tokenizer, preprocess, batch):
    content = batch["content"]
    content = [tokenizer.tokenize(preprocess(x)) for x in content]
    return content


def extract_metadata(col, extract_fc):
    return [extract_fc(x) for x in col]

def add_metadata(metadata, batch):
    args = None
    match metadata:
        case "words":
            args = (batch["tokenized"], len)

        case "non_alpha":
            args = (batch["tokenized"], (lambda c: sum((map(lambda x: not x.isalpha(), c)))))

        case "upercase":
            args = (batch["tokenized"], (lambda c: sum(list(map(str.isupper, c)))))

        case "digits":
            args = (batch["tokenized"], (lambda c: sum(list(map(str.isdigit, c)))))

        case "capitalized":
            args = (batch["tokenized"], (lambda c: sum(list(map(str.istitle, c)))))

    if args is None:
        raise ValueError(f"Unknown metadata {metadata}")
    
    return extract_metadata(*args)

def add_tokenized(dataset: Dataset, tokenizer: str, metadata: List[str], num_proc: int, batch_size=1024):
    batch_tokenizer = partial(tokenize_batch, get_tokenizer(tokenizer))
    
    dataset = dataset.map(lambda batch: {
        "tokenized": batch_tokenizer(dummy, batch)
    }, batched=True, batch_size=batch_size, num_proc=num_proc)

    dataset = dataset.map(lambda batch:
                           { k: add_metadata(k, batch) for k in metadata}
                           , batched=True, batch_size=batch_size, num_proc=num_proc)
    return dataset


def create_save_folder(args, path):
    folder = path / args.id
    folder.mkdir(parents=True, exist_ok=False)
    with open(folder / "args.txt", "w") as f:
        f.write(str(args))
    return folder

def save_tfidf(tfidf, save_folder):
    if not save_folder.exists():
        save_folder.mkdir(parents=True)
    with open(save_folder / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)


def create_tfdif_vectorizer(args):
    tfidf = TfidfVectorizer(
        min_df=args.tfidf_min_df,
        max_df=args.tfidf_max_df,
        preprocessor=partial(preprocess_tokenized, args.tfidf_lower),
        lowercase=False,
        tokenizer=dummy,
        ngram_range=args.tfidf_ngram_range,
        stop_words=get_stop_words("czech")
    )

    return tfidf
    

def get_tfidf(tfidf: TfidfVectorizer, df: pd.DataFrame):
    content = df["tokenized"]
    print(f"Converting {len(content)} documents to TF-IDF")
    if hasattr(tfidf, "vocabulary_"):
        tf_features = tfidf.transform(content)
    else:
        tf_features = tfidf.fit_transform(content)
    return tf_features


def save_vectorizer(tfidf, save_folder):
    with open(save_folder / "vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

def run(args):
    save_folder = create_save_folder(args, args.output_path)
    tfidf = create_tfdif_vectorizer(args)

    for split in args.splits:
        dataset = load_dataset(str(args.dataset_path), split=split)
        if args.limit is not None:
            dataset = dataset.select(range(args.limit))
        dataset = add_tokenized(dataset, args.tokenizer, args.metadata, args.num_proc)
        df = dataset.to_pandas()
        tfidf_csr = get_tfidf(tfidf, df)
        metadata = df[args.metadata]
        scipy.sparse.save_npz(save_folder / f"{split}_tfidf.npz", tfidf_csr)
        metadata.to_parquet(save_folder / f"{split}_metadata.parquet")

    save_vectorizer(tfidf, save_folder)

    
def parse_ngram_range(s):
    return tuple(map(int, tuple(s.split("-"))))

def tfidf_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("TF-IDF")
    group.add_argument("--tfidf_min_df", type=int, default=1)
    group.add_argument("--tfidf_max_df", type=float, default=1.0)
    group.add_argument("--tfidf_ngram_range", type=parse_ngram_range, default=(1,2))
    group.add_argument("--tfidf_lower", type=bool, default=True)
    return parser

def tokenize_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Tokenization")
    group.add_argument("--tokenizer", type=str, default="moses")
    group.add_argument("--metadata", type=str, nargs="+", default=["words", "non_alpha", "upercase", "digits", "capitalized"])
    return parser

def run_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Run")
    group.add_argument("--splits", type=str, default="train,validation,test")
    group.add_argument("--limit", type=int, default=None)
    group.add_argument("--id", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    group.add_argument("--num_proc", type=int, default=None)
    return parser

# load arguments
def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser = tokenize_args(parser)
    parser = tfidf_args(parser)
    parser = run_args(parser)
    return parser.parse_args()

if __name__ == "__main__":
    args = load_args()
    args.splits = args.splits.split(",")
    run(args)
