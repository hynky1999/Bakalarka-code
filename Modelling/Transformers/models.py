from dataclasses import dataclass
from typing import Any, List
from lightning import LightningModule
import torch
from torch.nn import ModuleList, ModuleDict
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM
from typing import Callable
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.modeling_outputs import MaskedLMOutput
from metrics import create_test_metrics
from metrics import create_val_metrics
from metrics import create_train_metrics
from metrics import log_metrics, PerplexityFromLossMetric
from optims import CreateableOptimizer
from schedulers import CreateableScheduler


@dataclass
class LogitsOutput:
    logits: torch.Tensor
    loss: torch.Tensor


class BaseModel(LightningModule):
    def __init__(
        self,
        optimizer: CreateableOptimizer | None,
        scheduler: CreateableScheduler | None,
    ):
        super().__init__()
        self.optim = optimizer
        self.scheduler = scheduler


    def configure_optimizers(self):
        optimizer = self.optim(self) if self.optim else None
        if optimizer is None:
            return None

        scheduler = self.scheduler(optimizer, self.trainer) if self.scheduler else None
        if scheduler is None:
            return optimizer

        return [optimizer], [scheduler.__dict__]



class ClassificationModel(BaseModel):
    def __init__(
        self,
        num_classes,
        optimizer: CreateableOptimizer | None,
        scheduler: CreateableScheduler | None,
    ):
        super().__init__(optimizer, scheduler)
        self.metrics = ModuleDict(
            {
                "train_metrics": ModuleList(create_train_metrics(num_classes)),
                "val_metrics": ModuleList(create_val_metrics(num_classes)),
                "test_metrics": ModuleList(create_test_metrics(num_classes)),
            }
        )
        self.optim = optimizer
        self.scheduler = scheduler

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        self.log(
            "train/loss_step", output.loss, logger=True, on_step=True, on_epoch=False
        )
        predicted_labels = torch.argmax(output.logits, dim=1)
        log_metrics(self, predicted_labels, batch["labels"], "train")
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        labels = batch["labels"]
        predicted_labels = torch.argmax(output.logits, dim=1)
        log_metrics(self, predicted_labels, labels, "val")
        self.log(
            "val/loss_epoch", output.loss, logger=True, on_epoch=True, on_step=False
        )
        return output.loss


    def test_step(self, batch, batch_idx):
        output = self(**batch)
        labels = batch["labels"]
        predicted_labels = torch.argmax(output.logits, dim=1)
        log_metrics(self, predicted_labels, labels, "test")
        self.log(
            "val/loss_epoch", output.loss, logger=True, on_epoch=True, on_step=False
        )
        return output.loss

    def forward(self) -> LogitsOutput:
        raise NotImplementedError


class FineTunedClassifier(ClassificationModel):
    def __init__(
        self,
        pretrained_model,
        num_classes,
        scheduler,
        optimizer,
    ):
        super().__init__(num_classes, optimizer, scheduler)
        self.model = self.get_model(pretrained_model, num_classes)
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

    @staticmethod
    def get_model(model_chp, num_dim):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_chp, num_labels=num_dim
        )
        return model

    def forward(self, **kwargs):
        return self.model(**kwargs)


class LinearRegressionModel(ClassificationModel):
    def __init__(self, num_features, num_classes, scheduler, optimizer):
        super().__init__(num_classes, optimizer, scheduler)
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(num_features, num_classes)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, labels) -> LogitsOutput:
        logits = self.linear(input_ids)
        loss = self.cross_entropy(logits, labels)
        return LogitsOutput(logits, loss)


class LMModel(BaseModel):
    def __init__(self, pretrained_model, optimizer, scheduler):
        super().__init__(optimizer, scheduler)
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
        self.metrics = ModuleDict(
            {
                "train_metrics": PerplexityFromLossMetric(),
                "val_metrics": PerplexityFromLossMetric(),
                "test_metrics": PerplexityFromLossMetric(),
            }
        )
    
    def forward(self, **kwargs) -> MaskedLMOutput:
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx): 
        output = self(**batch)
        self.log("train/perplexity", self.metrics["train_metrics"](output.loss), logger=True, on_step=True, on_epoch=True)
        self.log("train/loss", output.loss, logger=True, on_step=True, on_epoch=False)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.log("val/perplexity", self.metrics["val_metrics"](output.loss), logger=True, on_epoch=True, on_step=False)
        return output.loss

    def test_step(self, batch, batch_idx):
        output = self(**batch)
        self.log("test/perplexity", self.metrics["test_metrics"](output.loss), logger=True, on_epoch=True, on_step=False)
        return output.loss


