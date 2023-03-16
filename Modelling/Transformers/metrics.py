from functools import partial
import torch
from torchmetrics import Metric
from dataclasses import dataclass
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision
from torchmetrics.classification import MulticlassConfusionMatrix
from lightning.pytorch.loggers.wandb import WandbLogger


@dataclass
class MetricMetadata:
    readable_name: str
    step: bool
    epoch: bool

# Would be nice if it would be real metric but I would have to make an effort to ensure synchronizaiton
class MetricWithMetadata(Metric):
    def __init__(self, metric, metadata: MetricMetadata, **kwargs):
        super().__init__(**kwargs)
        self._metric = metric
        self.real_type = type(metric)
        self.metadata = metadata

    def update(self, *args, **kwargs):
        self._metric.update(*args, **kwargs)

    def compute(self):
        return self._metric.compute()

    def reset(self):
        super().reset()
        self._metric.reset()



def create_train_metrics(num_classes: int):
    partial_create = partial(create_metric, num_classes=num_classes)
    train_metrics = [
        partial_create("f1_macro", epoch=True, step=False),
        partial_create("f1_micro", epoch=True, step=False),
    ]
    return train_metrics

def create_test_metrics(num_classes: int):
    partial_create = partial(create_metric, num_classes=num_classes)
    test_metrics = [
        partial_create("f1_macro", epoch=True, step=False),
        partial_create("f1_micro", epoch=True, step=False),
    ]
    return test_metrics

def create_val_metrics(num_classes: int):
    partial_create = partial(create_metric, num_classes=num_classes)
    val_metrics = [
        partial_create("f1_macro", epoch=True, step=False),
        partial_create("f1_micro", epoch=True, step=False),
    ]
    return val_metrics


metric_defs = {
    "accuracy": (Accuracy, {"task": "multiclass"}),
    "f1_macro": (F1Score, {"task": "multiclass", "average": "macro"}),
    "f1_micro": (F1Score, {"task": "multiclass", "average": "micro"}),
    "precision_macro": (Precision, {"task": "multiclass", "average": "macro"}),
    "precision_micro": (Precision, {"task": "multiclass", "average": "micro"}),
    "confusion_matrix": (ConfusionMatrix, {"task": "multiclass"}),
}

def create_metric(metric_name, num_classes , epoch: bool, step: bool):
    metric_def = metric_defs[metric_name]
    metric = metric_def[0](**metric_def[1], num_classes=num_classes)
    metadata = MetricMetadata(metric_name, epoch=epoch, step=step)
    return MetricWithMetadata(metric, metadata)



def log_confussion_matrix(model, cfs_matrix, split):
    print(type(model.logger))
    if not isinstance(model.logger, WandbLogger):
        raise ValueError("Confusion matrix logging is only supported for wandb")

    data, cols = prepare_confussion_matrix_for_logging(
        cfs_matrix 
    )
    model.logger.log_table(
        f"{split}/confusion_matrix_{model.current_epoch}", data=data, columns=cols
    )

def prepare_confussion_matrix_for_logging(confusion_matrix):
    data = []
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            data.append((i, j, confusion_matrix[i, j]))
    return data, ["Actual", "Predicted", "Count"]


def log_metrics(model, predicted_labels, labels, split):
        for metric in model.metrics[split + "_metrics"]:
            if metric.metadata.step:
                step_value = metric(predicted_labels, labels)
                model.log(
                    f"{split}/{metric.metadata.readable_name}_step",
                    step_value,
                    logger=True,
                )
            else:
                metric.update(predicted_labels, labels)

def reset_and_log_metrics(model, split):
    for metric in model.metrics[split + "_metrics"]:
        metric: MetricWithMetadata
        if metric.metadata.epoch:
            # Kinda monkey patched
            if metric.real_type == MulticlassConfusionMatrix:
                cm = metric.compute()
                log_confussion_matrix(model, cm, split)
            else:
                model.log(
                    f"{split}/{metric.metadata.readable_name}_epoch",
                    metric.compute(),
                    logger=True,
                )
        metric.reset()


class PerplexityFromLossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("crossentropy", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss):
        self.crossentropy += loss
        self.total += loss.numel()

    def compute(self):
        return torch.exp(self.crossentropy/self.total)
    