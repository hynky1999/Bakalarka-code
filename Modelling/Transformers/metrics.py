from torchmetrics import Metric
from dataclasses import dataclass


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