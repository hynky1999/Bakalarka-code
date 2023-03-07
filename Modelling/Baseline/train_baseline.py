from typing import Any, Callable, Iterable, List
from datasets import load_from_disk, Dataset
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from datasets import concatenate_datasets
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import confusion_matrix, get_scorer
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
from create_baseline_dataset import get_tokenizer, add_tokenized
from datasets import load_dataset
import argparse
import scipy
import pickle
import numpy as np
import pandas as pd
import wandb
from dataclasses import dataclass
from hydra.utils import to_absolute_path, instantiate
from baseline_utils import dummy, preprocess_tokenized, removeMinus,  pandas_reshaper, to_pandas
from omegaconf import OmegaConf
import json




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
    
    all_columns = df.columns
    scores = [x for x in all_columns if x.startswith("split0_")]
    stats = ["mean_fit_time", "std_fit_time"]
    df = df[["params"] + scores + stats]
    df = df.rename(columns={x: x.replace("split0_", "") for x in scores})


    wandb.log({"cv_results": df})


class DummyEstimator(BaseEstimator):
    def fit(self, x, y):
        return self

    def predict(self, x):
        return x

    def predict_proba(self, x):
        return x

def log_confusion_matrix(y_test, predictions_labels, labels, title):
    cfs = confusion_matrix(y_test, predictions_labels, labels=range(len(labels)))
    data = [[labels[i], labels[j] ,cfs[i, j]] for i,j in np.ndindex(cfs.shape)]
    fields = [
        "Actual",
        "Predicted",
        "nPredictions",
    ]
       
    wandb.log({title: wandb.Table(data=data, columns=fields)})
    



def test(x_test: Dataset, y_test, model: Pipeline, labels, col, n_proc):
    # Must be filtred as the minusDrop is not applied to predictions
    # Still zero in dataset
    x_test = filter_zero(x_test, col, n_proc)
    # Already -1 in y
    y_test = y_test[y_test >= 0]



    predictions_prob = model.predict_proba(x_test)
    predictions_labels = np.argmax(predictions_prob, axis=1)
    log_confusion_matrix(y_test, predictions_labels, labels, f"{x_test.split}/confusion matrix")
    wandb.log({f"{x_test.split}/{s}": score_fc(DummyEstimator(), y_test, predictions_labels) for s, score_fc in model.named_steps["search"].scorer_.items()})

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


def create_model(model_args, columns, cv, params, tokenizer, metadata, scores: List[str], n_proc=-1):
    model = instantiate(model_args)
    cv_search = GridSearchCV(
        model,
        cv=cv,
        param_grid=params,
        return_train_score=True,
        n_jobs=n_proc,
        verbose=1,
        scoring=tuple(scores),
        refit=scores[0],
    )
    tok_n_proc = n_proc if n_proc > 0 else None

    model = Pipeline([
                      ("tokenize", FunctionTransformer(partial(add_tokenized, tokenizer=tokenizer, metadata=metadata, num_proc=tok_n_proc), validate=False)),
                      ("to_pandas", FunctionTransformer(to_pandas, validate=False)),
                      ("columns", columns),
                      ("drop_None", FunctionSampler(func=removeMinus)),
                       ("search", cv_search)], verbose=True)
    return model


def save_model(model, output_path):
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    artifact = wandb.Artifact("tfidf", type="model")
    artifact.add_dir(output_path)

    wandb.log_artifact(artifact)


def prepare_y(dataset, col):
    return np.array(dataset[col]) - 1


@hydra.main(version_base="1.1", config_path="./config", config_name="config")
def main(cfg: DictConfig):

    wandb.init(project=f"Baseline-{cfg.task.column.capitalize()}", config=vars(cfg), mode=cfg.logger.mode)
    print("Loading data...")
    train_dataset = load_dataset(str(cfg.data.dataset_path), split="train")
    eval_dataset = load_dataset(str(cfg.data.dataset_path), split="validation")
    test_dataset = load_dataset(str(cfg.data.dataset_path), split="test")


    if cfg.data.limit is not None:
        train_dataset = train_dataset.select(range(cfg.data.limit))
        eval_dataset = eval_dataset.select(range(cfg.data.limit))
        test_dataset = test_dataset.select(range(cfg.data.limit))


    train_eval_filtered = filter_zero(train_dataset, cfg.task.column, n_proc=cfg.n_proc)
    test_filtered = filter_zero(test_dataset, cfg.task.column, n_proc=cfg.n_proc)
    wandb.sklearn.plot_class_proportions(train_eval_filtered[cfg.task.column], test_filtered[cfg.task.column], train_dataset.features[cfg.task.column].names)


    y_train = prepare_y(train_dataset, cfg.task.column)
    y_eval = prepare_y(eval_dataset, cfg.task.column)

    columns = create_columns(cfg.tfidf.features, lowercase=cfg.tfidf.lowercase, max_features=cfg.tfidf.max_features, ngram_range=tuple(cfg.tfidf.ngram_range), min_df=cfg.tfidf.min_df, max_df=cfg.tfidf.max_df)
    train_split_size, eval_split_size = np.sum(y_train != -1), np.sum(y_eval != -1)
    train_eval_split = PredefinedSplit(
        test_fold=[-1] * train_split_size + [1] * eval_split_size
    )
    wandb.log({"Real size": train_split_size + eval_split_size})

    model = create_model(
        cfg.model.model,
        columns,
        train_eval_split,
        OmegaConf.to_container(cfg.search.cv_params),
        cfg.tfidf.tokenizer,
        cfg.tfidf.features,
        cfg.score_types,
        cfg.n_proc,
    )

    # We use pretokenized data
    labels = train_dataset.features[cfg.task.column].names[1:]
    if cfg.run.train:
        trained_model = train(train_dataset, y_train, eval_dataset, y_eval, model)
        save_model(trained_model, Path(to_absolute_path(cfg.output_path)) / cfg.task.column / wandb.run.id)
        test(train_dataset, y_train, trained_model, labels, cfg.task.column, cfg.n_proc)
        test(eval_dataset, y_eval, trained_model, labels, cfg.task.column, cfg.n_proc)

    else:
        trained_model = load_model(Path(to_absolute_path(cfg.model_checkpoint)))

    if cfg.run.test:
        test(test_dataset, prepare_y(test_filtered, cfg.task.column), trained_model, labels, cfg.task.column, cfg.n_proc)



def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    main()