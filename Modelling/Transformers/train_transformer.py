from argparse import ArgumentParser
import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig
import wandb
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.tuner.tuning import Tuner
from callbacks import SimpleLayersFreezerCallback
from datamodules import NewsDataModule
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping, Timer, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger 


def fit(trainer: Trainer, model, datamodule, ckpt_path=None):
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)


def validate(trainer: Trainer, model, datamodule):
    trainer.validate(model, datamodule)


def test(trainer, model, datamodule):
    trainer.test(model, datamodule)


def tune(trainer: Trainer, model, datamodule):
    tuner = Tuner(trainer)
    batch_size = tuner.scale_batch_size(
        model=model, datamodule=datamodule, mode="binsearch"
    )
    print(f"Found Batch size: {batch_size}")

    lr_finder = tuner.lr_find(model=model, datamodule=datamodule)
    print(f"Found lr: {lr_finder.suggestion()}")
    wandb.log({"lr_graph": lr_finder.plot(suggest=True)})


    
@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = WandbLogger(project=f"{cfg.task.column.capitalize()}-Deep-Learning", log_model=True, save_dir= cfg.logger.save_dir, config=vars(cfg))

    datamodule = instantiate(cfg.data, column=cfg.task.column, num_classes=cfg.task.num_classes)
    model = instantiate(cfg.model, num_classes=cfg.task.num_classes)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val/f1_macro_epoch",
            save_top_k=3,
            mode="max",
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/f1_macro_epoch",
            patience=3,
            mode="max",
            verbose=True,
        ),
        Timer(interval="step",
            duration="00:14:00:00"
        )
    ]
    additional_callbacks = instantiate(cfg.trainer.callbacks)
    print(datamodule)


    trainer: Trainer = Trainer(
        callbacks=callbacks + additional_callbacks,
        logger=logger,
        enable_model_summary=True,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,
        enable_progress_bar=True,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        accelerator=cfg.trainer.accelerator,
    )


    print(trainer.strategy)

    if cfg.run.tune == True:
        tune(trainer, model, datamodule)

    if cfg.run.fit == True:
        fit(trainer, model, datamodule, ckpt_path=cfg.chckpt_path)

    if cfg.run.validate == True:
        validate(trainer, model, datamodule)

    if cfg.run.test == True:
        test(trainer, model, datamodule)
    print("Done")


if __name__ == "__main__":
    main()