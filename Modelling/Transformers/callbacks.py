from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch import LightningModule
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

