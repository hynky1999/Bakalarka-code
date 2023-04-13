from typing import List, Union
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch import LightningModule
from torch.optim.optimizer import Optimizer

import torch.nn as nn
from utils import get_no_decay_groups



def count_params(opt: Optimizer, grad: bool = True):
    return sum(p.numel() for group in opt.param_groups for p in group["params"] if p.requires_grad == grad)


class GradualUnfreezingCallback(BaseFinetuning):
    @staticmethod
    def get_bone(model):
        if hasattr(model.model, "roberta"):
            return model.model.roberta
        elif hasattr(model.model, "bert"):
            return model.model.bert
        else:
            raise ValueError("No bone found")


    def __init__(self, unfreeze_per_epoch: int, div_lr: float = 2.6, min_unfreeze_layer: int = 0, start_lr=None, start_decay=None):
        super().__init__()
        self.unfreeze_per_epoch = unfreeze_per_epoch
        self.total_layers = 0
        self.start_lr = start_lr
        self.start_decay = start_decay
        self.div_lr = div_lr
        self.min_unfreeze_layer = min_unfreeze_layer


    def freeze_before_training(self, model):
        self.freeze(model)
        # Classifier unfreeze
        self.make_trainable(model.model.classifier)

        self.total_layers = len(self.get_bone(model).encoder.layer)

    def finetune_function(
        self, model: LightningModule, epoch: int, optimizer: Optimizer
    ):

        r_unfreeze_l = self.total_layers - epoch * self.unfreeze_per_epoch
        l_unfreeze_l = max(self.min_unfreeze_layer, r_unfreeze_l - self.unfreeze_per_epoch)
        # Nothing to unfreeze
        if r_unfreeze_l <= self.min_unfreeze_layer:
            return


        # Optims should have this
        lr = self.start_lr
        decay = self.start_decay
        if lr is None:
            lr = optimizer.param_groups[0]["initial_lr"]
        if decay is None:
            decay = optimizer.param_groups[0]["weight_decay"]


        # If not unfreeze divide lr
        if r_unfreeze_l != self.total_layers:
            lr = lr / self.div_lr**epoch


        new_layers = self.get_bone(model).encoder.layer[
            l_unfreeze_l:r_unfreeze_l
        ]
        print(f"Unfreezing {l_unfreeze_l}-{r_unfreeze_l} layers with lr {lr} and decay {decay}")
        self.make_trainable(new_layers)
        param_groups = get_no_decay_groups(new_layers, lr, decay, ["bias", "LayerNorm.weight"])
        print(len(optimizer.param_groups))
        for param_group in param_groups:
            optimizer.add_param_group(param_group)
        print(f"Total trainable params: {count_params(optimizer)}")

            
