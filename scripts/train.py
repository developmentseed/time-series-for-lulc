import os
import sys
from pathlib import Path

import torch
import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset import PixelLDM
from model import PixelLM
from wikidata import S2A_LULC_CLS, S2A_LULC_COLORS
from evaluate import log_metrics


def train(
    num_bands: int = 10,
    num_classes: int = 10,
    timestep: int = 13,
    hidden_dims: int = 64,
    drop: float = 0.2,
    lr: float = 1e-3,
    wd: float = 1e-6,
    milestones: list = [8, 12, 15, 18],
    gamma: float = 0.5,
    cubes_dir: Path = Path("./data/cubesxy"),
    lc_klass: list = S2A_LULC_CLS,
    lc_colors: list = S2A_LULC_COLORS,
    batch_size: int = 8,
    num_workers: int = 12,
    fast_dev_run: bool = False,
    debug: bool = False,
    precision: int = 16,
    accelerator: str = "gpu",
    devices: int = 1,
    max_epochs: int = 20,
    accumulate_grad_batches: int = 1,
):
    # Create logs/ - wandb logs to an already existing directory only
    cwd = os.getcwd()
    (Path(cwd) / "logs").mkdir(exist_ok=True)

    # LOGGERs
    name = sys.argv[1]
    wandb_logger = WandbLogger(
        project="pixel",
        name=name,
        save_dir="logs",
        log_model=False,
    )
    csv_logger = CSVLogger(save_dir="logs/pixel", name=name)

    ckpt_cb = ModelCheckpoint(
        monitor="val_f1_score",
        mode="max",
        save_top_k=2,
        verbose=True,
        filename="epoch:{epoch}-step:{step}-loss:{val_loss:.3f}-f1score:{val_f1_score:.3f}",
        auto_insert_metric_name=False,
    )

    # early_stop_cb = EarlyStopping(
    #     monitor="val_loss",
    #     mode="min",
    #     min_delta=0.01,
    #     patience=5,
    #     verbose=True,
    # )

    lr_cb = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
    )

    dm = PixelLDM(
        cubes_dir=cubes_dir,
        num_bands=num_bands,
        timestep=timestep,
        lc_klass=lc_klass,
        lc_colors=lc_colors,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = PixelLM(
        num_bands=num_bands,
        num_classes=num_classes,
        timestep=timestep,
        hidden_dims=hidden_dims,
        drop=drop,
        lr=lr,
        wd=wd,
        milestones=milestones,
        gamma=gamma,
    )

    trainer = L.Trainer(
        fast_dev_run=fast_dev_run,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=max_epochs,
        logger=[csv_logger, wandb_logger],
        callbacks=[ckpt_cb, lr_cb],
        limit_train_batches=2 if debug else 1.0,
        limit_val_batches=2 if debug else 1.0,
        limit_test_batches=2 if debug else 1.0,
        log_every_n_steps=1,
    )

    # FIT
    print("TRAINING...")
    trainer.fit(model=model, datamodule=dm)

    # EVALUATE
    device = "cuda"
    print("EVALUATING...")
    # Load the best weights for the model from training
    model = PixelLM.load_from_checkpoint(ckpt_cb.best_model_path)
    model = model.to(device)
    model.eval()
    log_metrics(dm, model, device, wandb=True)


if __name__ == "__main__":
    train()
