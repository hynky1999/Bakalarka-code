from typing import Any, List
from lightning import LightningModule
import torch
from torch.nn import ModuleList, ModuleDict
from transformers import AutoModelForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup
from metrics import create_test_metrics
from metrics import create_val_metrics
from metrics import create_train_metrics
from metrics import reset_and_log_metrics, log_metrics
from optims import TransformerAdamW




class FineTunedClassifier(LightningModule):
    def __init__(
        self,
        pretrained_model,
        class_num,
        lr=2e-5,
        warmup_steps=1000,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pretrained_model = self.get_model(pretrained_model, class_num)
        self.metrics = ModuleDict({
            "train_metrics": ModuleList(create_train_metrics(class_num)),
            "val_metrics": ModuleList(create_val_metrics(class_num)),
            "test_metrics": ModuleList(create_test_metrics(class_num))
        })

    def get_model(self, model_chp, num_dim):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_chp, num_labels=num_dim
        )
        return model
        

    def training_step(self, batch, batch_idx):
        output = self.pretrained_model(**batch)
        self.log("train/loss_step", output.loss, logger=True, on_step=True, on_epoch=False)
        predicted_labels = torch.argmax(output.logits, dim=1)
        log_metrics(self, predicted_labels, batch["labels"], "train")
        return output.loss

    def training_epoch_end(self, outputs):
        reset_and_log_metrics(self, "train")

    def validation_step(self, batch, batch_idx):
        output = self.pretrained_model(**batch)
        labels = batch["labels"]
        predicted_labels = torch.argmax(output.logits, dim=1)
        log_metrics(self, predicted_labels, labels, "val")
        self.log("val/loss_epoch", output.loss, logger=True, on_epoch=True, on_step=False)
        return output.loss

    def validation_epoch_end(self, outputs):
        reset_and_log_metrics(self, "val")

    def test_step(self, batch, batch_idx):
        output = self.pretrained_model(**batch)
        labels = batch["labels"]
        predicted_labels = torch.argmax(output.logits, dim=1)
        log_metrics(self, predicted_labels, labels, "test")
        self.log("val/loss_epoch", output.loss, logger=True, on_epoch=True, on_step=False)
        return output.loss

    def test_epoch_end(self, outputs):
        reset_and_log_metrics(self, "test")

    def configure_optimizers(self):
        scheduler = None
        optimizer = TransformerAdamW(
            self,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
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
