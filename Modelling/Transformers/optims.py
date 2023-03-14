from torch.optim import AdamW, Optimizer
from typing import Callable
from lightning.pytorch import LightningModule
def get_no_bias_weight_adamw(model: LightningModule, lr: float, weight_decay: float, eps: float, betas: tuple):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    adamw = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay, eps=eps, betas=betas)
    return adamw

CreateableOptimizer = Callable[[LightningModule], Optimizer]