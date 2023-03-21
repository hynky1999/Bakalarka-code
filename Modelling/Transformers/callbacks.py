from typing import List, Union
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch import LightningModule
from torch.optim.optimizer import Optimizer

import torch.nn as nn
from utils import get_no_decay_groups



def count_params(opt: Optimizer, grad: bool = True):
    return sum(p.numel() for group in opt.param_groups for p in group["params"] if p.requires_grad == grad)


class GradualUnfreezingCallback(BaseFinetuning):
    def __init__(self, unfreeze_per_epoch: int, div_lr: float = 2.6, min_unfreeze_layer: int = 0, remove_batches: int = 0):
        super().__init__()
        self.unfreeze_per_epoch = unfreeze_per_epoch
        self.remove_batches = remove_batches
        self.last_unfrozen_layer = 0
        self.total_layers = 0
        self.div_lr = div_lr
        self.min_unfreeze_layer = min_unfreeze_layer


    def freeze_before_training(self, model):
        self.freeze(model)
        # Classifier unfreeze
        self.make_trainable(model.model.classifier)
        self.total_layers = len(model.model.roberta.encoder.layer)
        self.last_unfrozen_layer = self.total_layers

    def finetune_function(
        self, model: LightningModule, epoch: int, optimizer: Optimizer
    ):

        # Nothing to unfreeze
        if self.last_unfrozen_layer <= self.min_unfreeze_layer:
            return


        old_last_unfrozen_layer = self.last_unfrozen_layer
        self.last_unfrozen_layer = max(self.min_unfreeze_layer, self.last_unfrozen_layer - self.unfreeze_per_epoch)

        
        # Optims should have this
        lr = optimizer.param_groups[-1]["initial_lr"]
        decay = optimizer.param_groups[-1]["weight_decay"]

        if old_last_unfrozen_layer != self.total_layers:
            lr /= self.div_lr


        new_layers = model.model.roberta.encoder.layer[
            self.last_unfrozen_layer:old_last_unfrozen_layer
        ]
        print(f"Unfreezing {self.last_unfrozen_layer}-{old_last_unfrozen_layer} layers with lr {lr} and decay {decay}")
        self.make_trainable(new_layers)
        param_groups = get_no_decay_groups(new_layers, lr, decay, ["bias", "LayerNorm.weight"])
        for param_group in param_groups:
            optimizer.add_param_group(param_group)
        print(f"Total trainable params: {count_params(optimizer)}")

            
