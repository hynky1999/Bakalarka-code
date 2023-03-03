from lightning.pytorch.callbacks import BaseFinetuning
from dataclasses import dataclass
from typing import List, Tuple
from lightning import LightningModule
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix, Metric
from torchmetrics.classification import MulticlassConfusionMatrix
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
    EarlyStopping,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner.tuning import Tuner
from transformers import PreTrainedTokenizerBase, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from pathlib import Path
from lightning.pytorch import LightningDataModule
from torch.nn import ModuleList, ModuleDict
import shutil

@dataclass
class MetricMetadata:
    readable_name: str
    step: bool
    epoch: bool

# Would be nice if it would be real metric but I would have to make an effort to ensure synchronizaiton
@dataclass
class MetricWithMetadata():
    Metric: Metric
    metadata: MetricMetadata






class SimpleLayersFreezer(BaseFinetuning):
    def __init__(self, last_unfreeze_layers: int):
        super().__init__()
        self.last_unfreeze_layers = last_unfreeze_layers

    def finetune_function(
        self, pl_module: "pl.LightningModule", epoch: int, optimizer, opt_idx: int
    ):
        pass

    def freeze_before_training(self, model):
        self.freeze(model)
        # Classifier unfreeze
        if hasattr(model.pretrained_model, "classifier"):
            self.make_trainable(model.pretrained_model.classifier)
        # Last freeze_layers layers unfreeze
        if hasattr(model.pretrained_model, "roberta") and hasattr(
            model.pretrained_model.roberta, "encoder"
        ):
            self.make_trainable(
                model.pretrained_model.roberta.encoder.layer[
                    -self.last_unfreeze_layers :
                ]
            )


def prepare_confussion_matrix_for_logging(confusion_matrix, class_names):
    data = []
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            data.append((class_names[i], class_names[j], confusion_matrix[i, j]))
    return data, ["Actual", "Predicted", "Count"]


class FineTunedRobeCzech(LightningModule):
    def __init__(
        self,
        model_chp,
        class_names,
        train_metrics: List[Metric],
        val_metrics: List[Metric],
        test_metrics: List[Metric],
        lr=2e-5,
        warmup_steps=1000,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pretrained_model = self.get_model(model_chp, len(class_names))
        self.metrics = ModuleDict()
        self.metric_metadatas = {}

        for metrics, split in zip(
            [train_metrics, val_metrics, test_metrics], ["train", "val", "test"]
        ):
            print(split)
            module_metrics, metadata = self.preprocess_metrics(metrics)
            self.metrics[split] = ModuleList(module_metrics)
            self.metric_metadatas[split] = metadata

    def preprocess_metrics(self, metrics) -> Tuple[ModuleList, List[MetricMetadata]]:
        module_metrics = []
        metadata = []
        for metric in metrics:
            module_metrics.append(metric.Metric)
            metadata.append(metric.metadata)
        return ModuleList(module_metrics), metadata
        
        



    def get_model(self, model_chp, num_dim):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_chp, num_labels=num_dim
        )
        return model

    def log_metrics(self, output: SequenceClassifierOutput, labels, split):
        predicted_labels = torch.argmax(output.logits, dim=1)
        self.log(
            f"{split}/loss",
            output.loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        metrics = self.metrics[split]
        for i, metric_metadata in enumerate(self.metric_metadatas[split]):
            metric = metrics[i]
            if metric_metadata.step:
                step_value = metric(predicted_labels, labels)
                self.log(
                    f"{split}/{metric_metadata.readable_name}_step",
                    step_value,
                    logger=True,
                )
            else:
                metric.update(predicted_labels, labels)

    def log_confussion_matrix(self, cfs_matrix, split):
        if not isinstance(self.logger, WandbLogger):
            raise Exception(
                "Confusion matrix logging is only supported for wandb logger"
            )
        data, cols = prepare_confussion_matrix_for_logging(
            cfs_matrix, self.hparams.class_names
        )
        self.logger.log_table(
            f"{split}/confusion_matrix_{self.current_epoch}", data=data, columns=cols
        )

    def reset_and_log_metrics(self, split):
        metrics = self.metrics[split]
        for i, metric_metadata in enumerate(self.metric_metadatas[split]):
            metric = metrics[i]
            if metric_metadata.epoch:
                if isinstance(metric, MulticlassConfusionMatrix):
                    cm = metric.compute()
                    self.log_confussion_matrix(cm, split)
                else:
                    self.log(
                        f"{split}/{metric_metadata.readable_name}_epoch",
                        metric.compute(),
                        logger=True,
                    )
            metric.reset()

    def training_step(self, batch, batch_idx):
        output = self.pretrained_model(**batch)
        self.log_metrics(output, batch["labels"], "train")
        return output.loss

    def training_epoch_end(self, outputs):
        self.reset_and_log_metrics("train")

    def validation_step(self, batch, batch_idx):
        output = self.pretrained_model(**batch)
        labels = batch["labels"]
        self.log_metrics(output, labels, "val")
        return output.loss

    def validation_epoch_end(self, outputs):
        self.reset_and_log_metrics("val")

    def test_step(self, batch, batch_idx):
        output = self.pretrained_model(**batch)
        labels = batch["labels"]
        self.log_metrics(output, labels, "test")
        return output.loss

    def test_epoch_end(self, outputs):
        self.reset_and_log_metrics("test")

    def configure_optimizers(self):
        scheduler = None
        # Use hugging face settings
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.pretrained_model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.pretrained_model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            eps=self.hparams.eps,
            betas=self.hparams.betas,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


class NewsDataModule(LightningDataModule):
    def __init__(
        self,
        column,
        tokenizer,
        max_length=512,
        batch_size=12,
        num_proc=4,
        trunc_type="start",
        cache_dir=Path("cache/czech_news_proc"),
        reload_cache=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.max_length = max_length
        self.column = column
        self.trunc_type = trunc_type
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True
        )
        self.cache_dir = cache_dir
        self.reload_cache = reload_cache

    def prepare_split(self, split, reload_cache):
        split_cache_dir = self.cache_dir / self.column / str(self.max_length) / split
        if split_cache_dir.exists():
            if not reload_cache:
                return
            else:
                shutil.rmtree(split_cache_dir)
                split_cache_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(str("hynky/czech_news_dataset"), split=split)
        dataset = dataset.rename_column(self.column, "labels")
        dataset = dataset.map(
            lambda batch: self.tokenizer(
                batch["content"], truncation=True, max_length=self.max_length
            ),
            keep_in_memory=True,
            batched=True,
        )
        cols = {"labels", "attention_mask", "input_ids"}
        # Remove "Nones"
        dataset = dataset.map(
            lambda batch: {"labels": [x - 1 for x in batch["labels"] if x != 0]},
            batched=True,
        )
        dataset = dataset.remove_columns(set(dataset.column_names) - cols)
        dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"])

        print("Saving")
        dataset.save_to_disk(str(split_cache_dir), num_proc=self.num_proc)

    def load_split(self, split):
        dataset = load_from_disk(
            self.cache_dir / self.column / str(self.max_length) / split
        )
        return dataset

    def prepare_data(self):
        self.prepare_split("train", reload_cache=self.reload_cache)
        self.prepare_split("validation", reload_cache=self.reload_cache)
        self.prepare_split("test", reload_cache=self.reload_cache)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_split("train")
            self.val_dataset = self.load_split("validation")
        if stage == "test" or stage is None:
            self.test_dataset = self.load_split("test")

    def create_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.create_dataloader(self.test_dataset)


# trainer.fit(model, datamodule=data_module)
# trainer.test(model, datamodule=data_module)


def create_metrics(class_names) -> Tuple[List[MetricWithMetadata], List[MetricWithMetadata], List[MetricWithMetadata]]:
    train_metrics, test_metrics, val_metrics = [
        [
            MetricWithMetadata(
                Accuracy(task="multiclass", num_classes=len(class_names)),
                MetricMetadata(
                    "accuracy",
                    step=False,
                    epoch=True,
                )
            ),
            MetricWithMetadata(
                F1Score(
                    task="multiclass", num_classes=len(class_names), average="macro"
                ),
                MetricMetadata(
                    "f1_macro",
                    step=True,
                    epoch=True,
                ),
            ),
            MetricWithMetadata(
                F1Score(
                    task="multiclass", num_classes=len(class_names), average="micro"
                ),
                MetricMetadata(
                    "f1_micro",
                    step=True,
                    epoch=True,
                ),
            ),
            MetricWithMetadata(
                Precision(
                    task="multiclass", num_classes=len(class_names), average="macro"
                ),
                MetricMetadata(
                    "precission",
                    step=False,
                    epoch=True,
                ),
            ),
            MetricWithMetadata(
                Recall(
                    task="multiclass", num_classes=len(class_names), average="macro"
                ),
                MetricMetadata(
                    "recall",
                    step=False,
                    epoch=True,
                ),
            ),
            MetricWithMetadata(
                ConfusionMatrix(task="multiclass", num_classes=len(class_names)),
                MetricMetadata(
                    "confusion_matrix",
                    step=False,
                    epoch=True,
                ),
            )
        ]
        for _ in range(3)
    ]

    return train_metrics, test_metrics, val_metrics


def run():
    column = "server"
    logger = WandbLogger(project=f"{column}_NN", log_model=True)
    data_module = NewsDataModule(
        column,
        "ufal/robeczech-base",
        max_length=512,
        batch_size=12,
        num_proc=4,
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
    directory = Path("logs") / f"{column}_NN" / logger.experiment.id
    torch.cuda.is_available()

    trainer = Trainer(
        max_epochs=4,
        accelerator="gpu",
        strategy="ddp",
        devices=1,
        callbacks=[
            SimpleLayersFreezer(2),
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
        limit_train_batches=500,
        limit_val_batches=500,
        limit_test_batches=10,
        log_every_n_steps=10,
        val_check_interval=100,
    )
    tuner = Tuner(trainer)
    tuner.lr_find(
        model, datamodule=data_module, min_lr=1e-7, max_lr=1e-1, num_training=100
    )


print(1111)
if __name__ == "__main__":
    run()
