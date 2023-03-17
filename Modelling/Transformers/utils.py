from typing import List
from lightning.pytorch.callbacks import BaseFinetuning
import torch.nn as nn

def get_named_params(modules: nn.Module | List[nn.Module], requires_grad: bool = True):
    modules = BaseFinetuning.flatten_modules(modules)
    for mod in modules:
        for param in mod.named_parameters(recurse=False):
            if param[1].requires_grad == requires_grad:
                yield param

def get_no_decay_groups(modules, lr, weight_decay, no_decay_names):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in get_named_params(modules)
                if not any(nd in n for nd in no_decay_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in get_named_params(modules)
                if any(nd in n for nd in no_decay_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    return optimizer_grouped_parameters