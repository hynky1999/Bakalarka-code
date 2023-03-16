from argparse import ArgumentParser
import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
import wandb
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.tuner.tuning import Tuner
from callbacks import SimpleLayersFreezerCallback
from datamodules import NewsDataModule
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping, Timer, LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger 


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


    
@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = None
    if "logger" in cfg:
        logger = instantiate(cfg.logger, project=f"{cfg.task.name.capitalize()}-Deep-Learning")(config=vars(cfg)) if cfg.logger else None

    datamodule_kwargs = OmegaConf.to_container(cfg.task.setting) if "setting" in cfg.task else {}
    datamodule = instantiate(cfg.data, num_proc=cfg.num_proc, batch_size=cfg.batch_size, **datamodule_kwargs)
    optimizer = instantiate(cfg.optimizer, _partial_=True) if "optimizer" in cfg else None
    scheduler = instantiate(cfg.scheduler, _partial_=True) if "scheduler" in cfg else None
    
    model_kwargs = {
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    if hasattr(datamodule, "num_features"):
        model_kwargs["num_features"] = datamodule.num_features

    model = instantiate(cfg.model, **model_kwargs)

    callbacks = [
        ModelCheckpoint(
            monitor="val/f1_macro_epoch",
            save_top_k=2,
            mode="max",
            verbose=True,
        ),
        # EarlyStopping(
        #     monitor="val/f1_macro_epoch",
        #     patience=3,
        #     mode="max",
        #     verbose=True,
        # ),
        Timer(interval="step",
            duration="00:14:00:00"
        )
    ]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))


    additional_callbacks = instantiate(cfg.callbacks)


    trainer: Trainer = Trainer(
        callbacks=callbacks + additional_callbacks,
        logger=logger,
        enable_model_summary=True,
        strategy=cfg.accelerator.strategy,
        devices=cfg.accelerator.devices,
        max_epochs=cfg.max_epochs,
        enable_progress_bar=True,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        accelerator=cfg.accelerator.accelerator,
    )


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