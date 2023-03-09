from typing import Any, Callable, Iterable, List
from datasets import load_from_disk, Dataset
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from datasets import concatenate_datasets
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import get_scorer
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
from datasets import load_dataset
import argparse
import scipy
import pickle
import numpy as np
import pandas as pd
import wandb
from dataclasses import dataclass
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@dataclass
class ScoreResult:
    name: str
    score: Any


def report_eval(results: Iterable[ScoreResult], set_name: str):
    for result in results:
        wandb.log({f"{set_name}/{result.name}": result.score})


def train(model: BaseEstimator, x, y):
    # Measuring the time of training
    print("Training")
    start = datetime.now()
    model = model.fit(x, y)
    end = datetime.now()
    wandb.log({"training_time": (end - start).total_seconds()})
    return model

def evaluate(
    model: Pipeline, x: scipy.sparse.csr_matrix, y: np.ndarray, score_fcs: dict, label_names: Iterable[Callable]
):
    prediction_labels = model.predict(x)
    scores = [
        ScoreResult(s, score_fc(DummyEstimator(), prediction_labels, y))
        for s, score_fc in score_fcs.items()

    ]
    wandb.sklearn.plot_confusion_matrix(
        y, prediction_labels, labels=label_names
    )
    return scores

class DummyEstimator(BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X


def create_model(model_cfg):
    model = hydra.utils.instantiate(model_cfg)
    return model


def save_model(model, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    artifact = wandb.Artifact("tfidf", type="model")
    artifact.add_dir(output_path)

    wandb.log_artifact(artifact)


def load_and_preprocess(
    df_path: Path,
    tfidf_path: Path,
    col: str,
    split: str,
):
    labels = load_dataset(df_path, split=split)[col]
    # None to -1
    labels = np.array(labels) - 1
    sparse_input = scipy.sparse.load_npz(tfidf_path / f"{split}_tfidf_with_metadata.npz")
    print(f"Loaded {split} with {len(labels)} samples and {sparse_input.shape[1]} features and {sparse_input.shape[0]} samples")
    if sparse_input.shape[0] != len(labels):
        limit = min(sparse_input.shape[0], len(labels))
        sparse_input = sparse_input[:, :limit]
        labels = labels[:limit]

    # Remove samples with no label
    keep_idxs = np.where(labels != -1)[0]

    labels = labels[keep_idxs]
    sparse_input = sparse_input[keep_idxs, :]


    print(f"Loaded {split} with {len(labels)} samples")
    print(f"Memory usage: {sparse_input.data.nbytes / 1024 / 1024} MB")
    return sparse_input, labels

def get_metdata_columns(metadata_path: Path):
    metadata = pd.read_parquet(metadata_path / "test_metadata.parquet")
    return list(metadata.columns)

def get_tfidf_columns(vectorizer_path: Path):
    with open(vectorizer_path / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer.get_feature_names_out()

def create_score_fc(score_names):
    return get_scorer(score_names)

def report_cv_results(cv_results):
    df = pd.DataFrame(cv_results)
    # iterate over df rows and return dict
    
    all_columns = df.columns
    scores = [x for x in all_columns if x.startswith("split0_")]
    stats = ["mean_fit_time", "std_fit_time"]
    df = df[["params"] + scores + stats]
    df = df.rename(columns={x: x.replace("split0_", "") for x in scores})


    wandb.log({"cv_results": df})

def get_best_params(cv_results, metric):
    best_idx = np.argmax(cv_results[f"mean_test_{metric}"])
    return cv_results["params"][best_idx]

def find_best_model(
        model: BaseEstimator,
        x: scipy.sparse.csr_matrix,
        y: np.ndarray,
        score_fcs: dict,
        split: PredefinedSplit,
        cv_params: dict,
):
    grid_search = GridSearchCV(
        model,
        param_grid=cv_params,
        scoring=score_fcs,
        return_train_score=True,
        cv=split,
        refit=False,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(x, y)
    report_cv_results(grid_search.cv_results_)
    best_params = get_best_params(grid_search.cv_results_, metric=list(score_fcs.keys())[0])
    model.set_params(**best_params)
    return model






@hydra.main(version_base="1.1", config_path="./config", config_name="config")
def main(cfg: DictConfig):
    partial_load = partial(
        load_and_preprocess,
        cfg.data.dataset_path,
        Path(to_absolute_path(cfg.data.tfidf_path)),
        cfg.task.column,
    )

    wandb.init(project=f"{cfg.task.column.capitalize()}-ML", config=vars(cfg), mode=cfg.logger.mode)
    print("Loading data...")
    train_x, train_y = partial_load(split="train")
    eval_x, eval_y = partial_load(split="validation")
    test_x, test_y = partial_load(split="test")

    print("Creating model...")
    model = create_model(cfg.model.model)

    score_fcs = {score: create_score_fc(score) for score in cfg.score_types}
    label_names = load_dataset(cfg.data.dataset_path, split="test").features[cfg.task.column].names[1:]
    # We use pretokenized data
    if cfg.run.load_model is not None:
        model = pickle.load(open(cfg.run.load_model, "rb"))
    if cfg.run.train is not None:
        # Measure time of training

        train_eval_x = scipy.sparse.vstack([train_x, eval_x])
        train_eval_y = np.concatenate([train_y, eval_y])
        cv_split = PredefinedSplit([-1] * len(train_y) + [1] * len(eval_y))
        model = find_best_model(model, train_eval_x, train_eval_y, score_fcs, cv_split, OmegaConf.to_container(cfg.search.cv_params))
        model = train(model, train_x, train_y)

        report_eval(evaluate(model, train_x, train_y, score_fcs, label_names), "train")
        report_eval(evaluate(model, eval_x, eval_y, score_fcs, label_names), "eval")
        save_model(model, Path(to_absolute_path(cfg.output_path)) / cfg.task.column / wandb.run.id)

    if cfg.run.test:
        report_eval(evaluate(model, test_x, test_y, score_fcs, label_names), "test")


if __name__ == "__main__":
    main()
