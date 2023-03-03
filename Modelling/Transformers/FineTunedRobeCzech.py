from typing import List
from lightning import LightningModule
import torch
from torch.nn import ModuleList, ModuleDict
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup
from lightning.pytorch.loggers.wandb import WandbLogger
from metrics import MetricWithMetadata
from torchmetrics.classification import MulticlassConfusionMatrix


def prepare_confussion_matrix_for_logging(confusion_matrix, class_names):
    data = []
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            data.append((class_names[i], class_names[j], confusion_matrix[i, j]))
    return data, ["Actual", "Predicted", "Count"]


class FineTunedRobeCzech(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FineTunedRobeCzech")
        parser.add_argument("--lr", type=float, default=2e-5)
        parser.add_argument("--warmup_steps", type=int, default=1000)
        parser.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.98))
        parser.add_argument("--eps", type=float, default=1e-08)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--model_chp", type=str, default="ufal/robeczech-roberta-base")
        return parent_parser



    def __init__(
        self,
        model_chp,
        class_names,
        train_metrics: List[MetricWithMetadata],
        val_metrics: List[MetricWithMetadata],
        test_metrics: List[MetricWithMetadata],
        lr=2e-5,
        warmup_steps=1000,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pretrained_model = self.get_model(model_chp, len(class_names))
        self.metrics = ModuleDict({
            "train_metrics": ModuleList(train_metrics),
            "val_metrics": ModuleList(val_metrics),
            "test_metrics": ModuleList(test_metrics)
        })

    def get_model(self, model_chp, num_dim):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_chp, num_labels=num_dim
        )
        return model

    def log_metrics(self, output: SequenceClassifierOutput, labels, split):
        predicted_labels = torch.argmax(output.logits, dim=1)
        for metric in self.metrics[split + "_metrics"]:
            if metric.metadata.step:
                step_value = metric(predicted_labels, labels)
                self.log(
                    f"{split}/{metric.metadata.readable_name}_step",
                    step_value,
                    logger=True,
                )
            else:
                metric.update(predicted_labels, labels)

    def log_confussion_matrix(self, cfs_matrix, split):
        if not isinstance(self.logger, WandbLogger):
            raise ValueError("Confusion matrix logging is only supported for wandb")

        data, cols = prepare_confussion_matrix_for_logging(
            cfs_matrix, self.hparams.class_names
        )
        self.logger.log_table(
            f"{split}/confusion_matrix_{self.current_epoch}", data=data, columns=cols
        )

    def reset_and_log_metrics(self, split):
        for metric in self.metrics[split + "_metrics"]:
            metric: MetricWithMetadata
            if metric.metadata.epoch:
                # Kinda monkey patched
                if metric.real_type == MulticlassConfusionMatrix:
                    cm = metric.compute()
                    self.log_confussion_matrix(cm, split)
                else:
                    self.log(
                        f"{split}/{metric.metadata.readable_name}_epoch",
                        metric.compute(),
                        logger=True,
                    )
            metric.reset()

    def training_step(self, batch, batch_idx):
        output = self.pretrained_model(**batch)
        self.log("train/loss", output.loss, logger=True, on_step=True, on_epoch=False)
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
