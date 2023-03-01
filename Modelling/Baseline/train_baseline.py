from typing import List
from datasets import load_from_disk, Dataset
from pathlib import Path
from datasets import concatenate_datasets
from sklearn.model_selection import PredefinedSplit
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn import FunctionSampler
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from scipy.sparse import vstack, hstack
from stop_words import get_stop_words
from datetime import datetime
from functools import partial
from baseline_utils import dummy, preprocess_tokenized, removeZero, pandas_reshaper, to_pandas
from create_baseline_dataset import get_tokenizer, add_tokenized
from datasets import load_dataset
import argparse
import pickle
import numpy as np
import pandas as pd
import joblib
import json
import wandb

def count_non_zero(x):
    return np.count_nonzero(x)

def filter_zero(x: Dataset, col, n_proc):
    n_proc = n_proc if n_proc > 0 else None
    return x.filter(lambda batch: [x!= 0 for x in batch[col]], batched=True, num_proc=n_proc )
    


def train(x_train, y_train, x_eval, y_eval, model: Pipeline):
    train_eval_x = concatenate_datasets([x_train, x_eval])
    train_eval_y = np.concatenate([y_train, y_eval])

    model = model.fit(train_eval_x, train_eval_y)
    report_cv_results(model.named_steps["search"].cv_results_)
    return model

def report_cv_results(cv_results):
    df = pd.DataFrame(cv_results)
    # iterate over df rows and return dict
    for index, row in df.iterrows():
        log_dict = {}
        for k, v in row.items():
            if k.startswith("param"):
                continue
            k = k.replace("split0_", "")
            k = k.replace("test", "val")
            for split in ["val", "train"]:
                if f"{split}_" in k:
                    k = k.replace(f"{split}_", "")
                    k = f"{split}/{k}"

            log_dict[k] = v

        wandb.log(log_dict)


def test(x_test: Dataset, y_test, model: Pipeline, labels):
    predictions_prob = model.predict_proba(x_test)
    predictions_labels = np.argmax(predictions_prob, axis=1)
    wandb.sklearn.plot_roc(y_test, predictions_prob, labels)
    wandb.sklearn.plot_confusion_matrix(y_test, predictions_labels, labels)
    wandb.log({f"test/{s}": score_fc(model, x_test, y_test) for s, score_fc in model.named_steps["search"].scorer_.items()})










def create_columns(used_features, lowercase, max_features, ngram_range, min_df, max_df):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=get_stop_words("czech"),
        tokenizer=dummy,
        preprocessor=partial(preprocess_tokenized, lowercase),
    )
    possible_features = [
        ("words", make_pipeline(pandas_reshaper, StandardScaler()), "words"),
        ("digits", make_pipeline(pandas_reshaper, StandardScaler()), "digits"),
        ("capitalized", make_pipeline(pandas_reshaper, StandardScaler()), "capitalized"),
        ("non_alpha", make_pipeline(pandas_reshaper, StandardScaler()), "non_alpha"),
        (
            "upercase",
            make_pipeline(pandas_reshaper, StandardScaler()),
            "upercase",
        ),
    ]

    chosen_features = [col for col in possible_features if col[0] in used_features]
    if len(used_features) != len(chosen_features):
        raise ValueError(
            f"Used features {used_features} do not match {chosen_features}"
        )

    if len(chosen_features) == 0:
        raise ValueError(f"No features were chosen")

        
    columns = ColumnTransformer(
        chosen_features + [("tfidf", tfidf, "tokenized")], remainder="drop", n_jobs=-1, verbose=True
    )
    return columns


def create_model(columns, cv, params, tokenizer, metadata, scores: List[str], n_proc=-1):
    lr = LogisticRegression(verbose=2, solver='saga', random_state=42, multi_class='multinomial')
    cv_search = GridSearchCV(
        lr,
        cv=cv,
        param_grid=params,
        return_train_score=True,
        n_jobs=n_proc,
        verbose=1,
        scoring=scores,
        refit=scores[0],
    )
    tok_n_proc = n_proc if n_proc > 0 else None

    model = Pipeline([
                      ("tokenize", FunctionTransformer(partial(add_tokenized, tokenizer, metadata, tok_n_proc), validate=False)),
                      ("to_pandas", FunctionTransformer(to_pandas, validate=False)),
                      ("columns", columns),
                      ("drop_None", FunctionSampler(func=removeZero)),
                       ("search", cv_search)], verbose=True)
    return model


def save_model(model, output_path):
    with open(output_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    artifact = wandb.Artifact("tfidf", type="model")
    artifact.add_dir(output_path)

    wandb.log_artifact(artifact)


def prepare_y(dataset, col):
    return np.array(dataset[col])


def run(args):
    wandb.init(project=f"baseline_{args.col}", config=vars(args))
    train_dataset = load_dataset(str(args.dataset_path), split="train")
    eval_dataset = load_dataset(str(args.dataset_path), split="validation")
    test_dataset = load_dataset(str(args.dataset_path), split="test")


    if args.limit is not None:
        train_dataset = train_dataset.select(range(args.limit))
        eval_dataset = eval_dataset.select(range(args.limit))
        test_dataset = test_dataset.select(range(args.limit))


    train_eval_filtered = filter_zero(train_dataset, args.col, n_proc=args.n_proc)
    test_filtered = filter_zero(test_dataset, args.col, n_proc=args.n_proc)
    wandb.sklearn.plot_class_proportions(train_eval_filtered[args.col], test_filtered[args.col], train_dataset.features[args.col].names)


    y_train = prepare_y(train_dataset, args.col)
    y_eval = prepare_y(eval_dataset, args.col)

    columns = create_columns(args.features, lowercase=args.lowercase, max_features=args.max_features, ngram_range=args.ngram_range, min_df=args.min_df, max_df=args.max_df)
    non_zero_train, non_zero_eval = np.count_nonzero(y_train), np.count_nonzero(y_eval)
    train_eval_split = PredefinedSplit(
        test_fold=[-1] * non_zero_train + [1] * non_zero_eval
    )
    wandb.log({"Real size": non_zero_train + non_zero_eval})
    cv_params = prepare_cv_args(args)

    model = create_model(
        columns,
        train_eval_split,
        cv_params,
        args.tokenizer,
        args.features,
        args.score_type,
        args.n_proc,
    )

    # We use pretokenized data
    trained_model = train(train_dataset, y_train, eval_dataset, y_eval, model)
    save_model(trained_model, args.output_path)
    labels = train_dataset.features[args.col].names

    test(test_dataset, prepare_y(test_filtered, args.col), trained_model, labels)


def parse_ngram_range(s):
    return tuple(s.split("-"))


def prepare_cv_args(args):
    params = {}
    for key, value in vars(args).items():
        if (not isinstance(value, list)) or len(value) == 0:
            continue

        if key.startswith("lr__"):
            params[key[4:]] = value

    return params


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("col", type=str)
    parser.add_argument("--limit", type=int, nargs="?", default=None)
    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        default=["words", "non_alpha", "digits", "upercase", "capitalized"],
    )
    parser.add_argument(
        "--model_id",
        type=str,
        nargs="?",
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    parser.add_argument("--score_type", type=str, nargs="+", default=["f1_macro"])
    parser.add_argument("--tokenizer", type=str, nargs="?", default="moses")
    parser.add_argument("--n_proc", type=int, nargs="?", default=-1)
    parser.add_argument("--lr__C", type=float, nargs="*", default=[0.1, 1, 10, 100])
    parser.add_argument("--lr__tol", type=float, nargs="*", default=[1e-4])
    parser.add_argument("--lr__max_iter", type=int, nargs="*", default=[350])
    parser.add_argument("--lowercase", type=bool, nargs="?", default=True)
    parser.add_argument("--max_features", type=int, nargs="?", default=None)
    parser.add_argument(
        "--ngram_range", type=parse_ngram_range, nargs="?", default=(1, 2)
    )
    parser.add_argument("--min_df", type=int, nargs="?", default=70)
    parser.add_argument("--max_df", type=float, nargs="?", default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args.features)
    print(args.score_type)
    args.output_path = args.output_path / f"{args.model_id}"
    args.output_path.mkdir(parents=True, exist_ok=True)
    run(args)
