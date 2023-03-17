from argparse import ArgumentParser
import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
import wandb
from lightning.pytorch.tuner.tuning import Tuner
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping, Timer, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger


def fit(trainer: Trainer, model, datamodule, ckpt_path=None):
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)


def validate(trainer: Trainer, model, datamodule, ckpt_path=None):
    trainer.validate(model, datamodule, ckpt_path=ckpt_path)


def test(trainer, model, datamodule, ckpt_path=None):
    trainer.test(model, datamodule, ckpt_path=ckpt_path)


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
    if "logger" in cfg:
        logger = instantiate(cfg.logger, project=f"{cfg.task.name.capitalize()}-Deep-Learning")(config=vars(cfg))
    else:
        logger = TensorBoardLogger("tb_logs", name=cfg.task.name)

    

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
        Timer(interval="step",
            duration="02:00:00:00"
        ),
        LearningRateMonitor(logging_interval="step")
    ]
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
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        limit_test_batches=cfg.limit_test_batches,
        fast_dev_run=cfg.fast_dev_run,
        accelerator=cfg.accelerator.accelerator,
    )
    print(cfg.run.mode)


    if cfg.run.mode == "tune":
        tune(trainer, model, datamodule)

    if cfg.run.mode == "fit":
        fit(trainer, model, datamodule, ckpt_path=cfg.chckpt_path)

    if cfg.run.mode == "validate":
        validate(trainer, model, datamodule)

    if cfg.run.mode == "test":
        test(trainer, model, datamodule)


if __name__ == "__main__":
    main()