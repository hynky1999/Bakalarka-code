from typing import List, Union
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch import LightningModule
import torch.nn as nn
from utils import get_no_decay_groups
class SimpleLayersFreezerCallback(BaseFinetuning):
    def __init__(self, last_unfreeze_layers: int):
        super().__init__()
        self.last_unfreeze_layers = last_unfreeze_layers

    def finetune_function(
        self, pl_module: LightningModule, epoch: int, optimizer, opt_idx: int
    ):
        pass

    def freeze_before_training(self, model):
        self.freeze(model)
        # Classifier unfreeze
        if hasattr(model.model, "classifier"):
            self.make_trainable(model.model.classifier)
        # Last freeze_layers layers unfreeze
        if hasattr(model.model, "roberta") and hasattr(
            model.model.roberta, "encoder"
        ):
            self.make_trainable(
                model.model.roberta.encoder.layer[
                    -self.last_unfreeze_layers :
                ]
            )

class GradualUnfreezingCallback(BaseFinetuning):
    def __init__(self, unfreeze_per_epoch: int, div_lr: float = 2.6):
        super().__init__()
        self.unfreeze_per_epoch = unfreeze_per_epoch
        self.last_unfrozen_layer = None
        self.total_pos_layers = None
        self.div_lr = div_lr


    def freeze_before_training(self, model):
        self.freeze(model)
        # Classifier unfreeze
        if hasattr(model.model, "classifier"):
            self.make_trainable(model.model.classifier)
        # Last freeze_layers layers unfreeze
        if hasattr(model.model, "roberta") and hasattr(
            model.model.roberta, "encoder"
        ):
            self.total_pos_layers = len(model.model.roberta.encoder.layer)
            self.last_unfrozen_layer = max(0, self.total_pos_layers - self.unfreeze_per_epoch)
            self.make_trainable(
                model.model.roberta.encoder.layer[
                    self.last_unfrozen_layer:
                ]
            )

    def finetune_function(
        self, model: LightningModule, epoch: int, optimizer, opt_idx: int
    ):
        if self.total_pos_layers is None or self.last_unfrozen_layer == 0 or self.last_unfrozen_layer is None:
            return

        old_last_unfrozen_layer = self.last_unfrozen_layer
        self.last_unfrozen_layer = max(0, self.last_unfrozen_layer - self.unfreeze_per_epoch)


        new_layers = model.model.roberta.encoder.layer[
            self.last_unfrozen_layer:old_last_unfrozen_layer
        ]
        print(f"Unfreezing {len(new_layers)} layers")
        print(new_layers)
        self.make_trainable(new_layers)
        new_lr = optimizer.param_groups[-1]["lr"] / self.div_lr
        decay = optimizer.param_groups[0]["weight_decay"]
        param_groups = get_no_decay_groups(new_layers, new_lr, decay, ["bias", "LayerNorm.weight"])
        optimizer.add_param_group(param_groups)
            
