from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Tuple
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
    EarlyStopping,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.tuner.tuning import Tuner
from pathlib import Path

from metrics import MetricWithMetadata
from metrics import MetricMetadata
from CzechNewsDatamodule import NewsDataModule
from FineTunedRobeCzech import FineTunedRobeCzech
from callbacks import SimpleLayersFreezerCallback


def create_metrics(class_names) -> Tuple[List[MetricWithMetadata], List[MetricWithMetadata], List[MetricWithMetadata]]:
    metric_defs = {
        "accuracy": (Accuracy, {"task": "multiclass", "num_classes": len(class_names)}),
        "f1_macro": (F1Score, {"task": "multiclass", "num_classes": len(class_names), "average": "macro"}),
        "f1_micro": (F1Score, {"task": "multiclass", "num_classes": len(class_names), "average": "micro"}),
        "precision_macro": (Precision, {"task": "multiclass", "num_classes": len(class_names), "average": "macro"}),
        "precision_micro": (Precision, {"task": "multiclass", "num_classes": len(class_names), "average": "micro"}),
        "confusion_matrix": (ConfusionMatrix, {"task": "multiclass", "num_classes": len(class_names)}),
    }

    def create_metric(metric_name, epoch: bool, step: bool):
        metric_def = metric_defs[metric_name]
        metric = metric_def[0](**metric_def[1])
        metadata = MetricMetadata(metric_name, epoch=epoch, step=step)
        return MetricWithMetadata(metric, metadata)

    train_metrics = [
        create_metric("accuracy", epoch=True, step=False),
        create_metric("f1_macro", epoch=True, step=True),
        create_metric("f1_micro", epoch=True, step=False),
        create_metric("precision_macro", epoch=True, step=False),
        create_metric("precision_micro", epoch=True, step=False),
    ]

    test_metrics = [
        create_metric("accuracy", epoch=True, step=False),
        create_metric("f1_macro", epoch=True, step=False),
        create_metric("f1_micro", epoch=True, step=False),
        create_metric("precision_macro", epoch=True, step=False),
        create_metric("precision_micro", epoch=True, step=False),
    ]

    val_metrics = [
        create_metric("accuracy", epoch=True, step=False),
        create_metric("f1_macro", epoch=True, step=False),
        create_metric("f1_micro", epoch=True, step=False),
        create_metric("precision_macro", epoch=True, step=False),
        create_metric("precision_micro", epoch=True, step=False),
        create_metric("confusion_matrix", epoch=True, step=False),
    ]

    return train_metrics, test_metrics, val_metrics


def run(mode):
    column = "server"
    logger = WandbLogger(project=f"{column}_NN", log_model=True)
    data_module = NewsDataModule(
        column,
        "ufal/robeczech-base",
        max_length=512,
        batch_size=64,
        num_proc=1,
        trunc_type="start",
        cache_dir=Path("~/.cache/czech_news_proc").expanduser(),
        reload_cache=False,
    )
    data_module.prepare_data()
    data_module.setup()
    class_names = data_module.train_dataset.features["labels"].names[1:]
    train_metrics, test_metrics, val_metrics = create_metrics(class_names)


    model = FineTunedRobeCzech(
        "ufal/robeczech-base",
        class_names,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
    )
    directory = Path("logs") / f"{column}_NN" / str(logger.experiment.id)

    trainer = Trainer(
        max_epochs=4,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        callbacks=[
            SimpleLayersFreezerCallback(2),
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val/loss", patience=3, mode="min"),
            ModelCheckpoint(
                monitor="val/loss",
                save_top_k=5,
                mode="min",
                dirpath=directory / "checkpoints",
            ),
            Timer(duration="02:00:00:00"),
        ],
        enable_model_summary=True,
        enable_progress_bar=True,
        logger=logger,
        log_every_n_steps=1000,
        val_check_interval=0.1,
    )

    match mode:
        case "train":
            trainer.fit(model, datamodule=data_module)
            trainer.test(model, datamodule=data_module)

        case "tune":
            tuner = Tuner(trainer)
            lr_result = tuner.lr_find(
                model, datamodule=data_module, min_lr=1e-7, max_lr=1e-1, num_training=100
            )
            # lr
            wandb.log({"lr_find_graph": lr_result.plot(suggest=True)})
            print(f"Suggested learning rate: {lr_result.suggestion()}")


            # batch size
            batch_size_result = tuner.scale_batch_size(
                model, datamodule=data_module
            )
            print(f"Suggested batch size: {batch_size_result}")

        case "test":
            trainer.test(model, datamodule=data_module)
            

def get_trainer_args():
    parser = ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "tune", "test"])
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_trainer_args()
    run(args.mode)
