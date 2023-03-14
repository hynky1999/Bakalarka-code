from typing import Any, Callable, Dict, Iterable, List
from pathlib import Path
from sklearn.base import BaseEstimator
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import get_scorer
from datetime import datetime
from functools import partial
from callbacks import Callback

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from utils import ScoreResult
from datamodules import DataModule
import scipy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig, OmegaConf
from models import Trainer

def evaluate(dataset: DataModule, trainer: Trainer, set_name: str):
    scores, preds = trainer.evaluate(dataset)
    wandb.sklearn.plot_confusion_matrix(
        dataset.get_target(), preds, labels=dataset.get_label_names()
    )
    for score in scores:
        wandb.run.summary[f"{set_name}/best_{score.name}"] = score.score
    
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

def create_score_fc(score_names):
    return get_scorer(score_names)


@hydra.main(version_base="1.3", config_path="./config", config_name="config")
def main(cfg: DictConfig):
    cfg_as_dict = OmegaConf.to_container(cfg)
    wandb.init(project=f"{cfg.task.column.capitalize()}-ML", config=cfg_as_dict, mode=cfg.logger.mode, settings=wandb.Settings(start_method="thread"))

    print("Creating model...")
    model = create_model(cfg.model.model)


    score_fcs = {score: create_score_fc(score) for score in cfg.score_types}
    # We use pretokenized data
    if cfg.run.load_model is not None:
        model = pickle.load(open(cfg.run.load_model, "rb"))
    
    
    callbacks = [instantiate(callback) for callback in cfg.callbacks.callbacks]
    trainer = Trainer(
        model=model,
        score_fcs=score_fcs,
        batch_size=cfg.run.train.batch_size,
        epochs=cfg.run.train.epoch,
        log_steps=cfg.run.train.log_steps,
        callbacks=callbacks,
        out_dir=Path(to_absolute_path(cfg.output_path)) / cfg.task.column,
    )

    eval_set = None
    if cfg.run.train:
        train_set = instantiate(cfg.data.dataset, split="train", column=cfg.task.column)
        eval_set = instantiate(cfg.data.dataset, split="validation", column=cfg.task.column)
        trainer.train(train_set, eval_set)

    if cfg.run.eval:
        if eval_set is None:
            eval_set = instantiate(cfg.data.dataset, split="validation", column=cfg.task.column)
        evaluate(eval_set, trainer, "eval")

    if cfg.run.test:
        test_set = instantiate(cfg.data.dataset, split="test", column=cfg.task.column)
        evaluate(test_set, trainer, "test")



if __name__ == "__main__":
    main()
