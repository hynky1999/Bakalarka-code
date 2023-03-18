from torch.optim import AdamW, Optimizer
from typing import Callable
from lightning.pytorch import LightningModule
def get_no_bias_weight_adamw(model: LightningModule, lr: float, weight_decay: float, eps: float, betas: tuple):
    no_decay = ["bias", "LayerNorm.weight"]
    # Only the unfrozen layers
    named_params = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in named_params
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in named_params
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    adamw = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay, eps=eps, betas=betas)
    return adamw

CreateableOptimizer = Callable[[LightningModule], Optimizer]