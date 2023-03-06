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
from create_baseline_dataset import get_tokenizer, add_tokenized
from datasets import load_dataset
import argparse
import scipy
import pickle
import numpy as np
import pandas as pd
import wandb
from dataclasses import dataclass
from hydra.utils import to_absolute_path
from baseline_utils import to_ndarray


@dataclass
class ScoreResult:
    name: str
    score: Any


def report_eval(results: Iterable[ScoreResult], set_name: str):
    for result in results:
        wandb.log({f"{set_name}/{result.name}": result.score})


def train(model: Pipeline, x, y):
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


def create_columns(
    used_features,
    metadata_names,
    tfidf_names,
):
    chosen_features = []
    print(metadata_names)
    for name in used_features:
        if name in metadata_names:
            chosen_features.append(
                (name, make_pipeline(FunctionTransformer(to_ndarray), StandardScaler()), [metadata_names.index(name)])
            )
        else:
            raise ValueError(f"Unknown feature {name}")

    if len(chosen_features) == 0:
        raise ValueError(f"No features were chosen")

    columns = ColumnTransformer(
        chosen_features, n_jobs=-1, verbose=True, remainder="passthrough", sparse_threshold=1.0
    )
    return columns


def create_model(columns, model_cfg):
    model = hydra.utils.instantiate(model_cfg)
    pipeline = Pipeline([("columns", columns), ("model", model)], verbose=True)
    return pipeline


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
    metadata_path: Path,
    col: str,
    split: str,
):
    metadata = pd.read_parquet(metadata_path / f"{split}_metadata.parquet")
    labels = load_dataset(df_path, split=split).select(range(len(metadata))).map(lambda batch: {col: [x-1 for x in batch[col]]}, batched=True)[col]
    labels = np.array(labels)
    metadata_sparse = scipy.sparse.csr_matrix(metadata.values)
    tfidf_sparse = scipy.sparse.load_npz(tfidf_path / f"{split}_tfidf.npz")

    sparse_input = scipy.sparse.hstack([metadata_sparse, tfidf_sparse])
    keep_idxs = np.where(labels != -1)[0]
    labels = labels[keep_idxs]
    sparse_input = sparse_input[keep_idxs]
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


@hydra.main(version_base="1.1", config_path="./config", config_name="config")
def main(cfg: DictConfig):
    partial_load = partial(
        load_and_preprocess,
        cfg.data.dataset_path,
        Path(to_absolute_path(cfg.data.tfidf_path)),
        Path(to_absolute_path(cfg.data.metadata_path)),
        cfg.task.column,
    )

    wandb.init(project=f"Baseline-{cfg.task.column.capitalize()}", config=vars(cfg), mode=cfg.logger.mode)
    print("Loading data...")
    train_x, train_y = partial_load(split="train")
    eval_x, eval_y = partial_load(split="validation")
    test_x, test_y = partial_load(split="test")

    metadata_names = get_metdata_columns(Path(to_absolute_path(cfg.data.metadata_path)))
    tfidf_names = get_tfidf_columns(Path(to_absolute_path(cfg.data.tfidf_path)))
    columns = create_columns(cfg.features, metadata_names, tfidf_names)


    print("Creating model...")
    model = create_model(columns, cfg.model.model)

    score_fcs = {score: create_score_fc(score) for score in cfg.score_types}
    label_names = np.unique(np.concatenate([train_y, eval_y, test_y]))
    # We use pretokenized data
    if cfg.run.load_model is not None:
        model = pickle.load(open(cfg.load_model, "rb"))
    if cfg.run.train is not None:
        # Measure time of training
        model = train(model, train_x, train_y)

        report_eval(evaluate(model, train_x, train_y, score_fcs, label_names), "train")
        report_eval(evaluate(model, eval_x, eval_y, score_fcs, label_names), "eval")
        save_model(model, Path(to_absolute_path(cfg.data.output_path)) / cfg.task.column / wandb.run.id)

    if cfg.run.test:
        report_eval(evaluate(model, test_x, test_y, score_fcs, label_names), "test")


if __name__ == "__main__":
    main()
