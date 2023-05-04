from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as L
from torchmetrics.classification import F1Score


def _conv(ni, nf, ks):
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=1, padding=ks // 2)


def _conv_layer(ni, nf, ks, drop=0.0):
    return nn.Sequential(
        _conv(ni, nf, ks), nn.BatchNorm1d(nf), nn.ReLU(), nn.Dropout(p=drop)
    )


def stacked_conv(ni, nf, ks, drop):
    return nn.Sequential(
        _conv_layer(ni, nf, ks, drop=drop),
        nn.MaxPool1d(kernel_size=2, stride=2),
        _conv_layer(nf, nf, ks, drop=0.0),
        # nn.AdaptiveAvgPool1d(output_size=1),
        nn.Flatten(),
    )


class Pixel(nn.Module):
    def __init__(self, num_bands, num_classes, timestep, hidden_dims, drop):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=num_bands)
        self.conv1 = stacked_conv(ni=num_bands, nf=hidden_dims, ks=3, drop=drop)
        self.conv2 = stacked_conv(ni=num_bands, nf=hidden_dims, ks=5, drop=drop)
        self.head = nn.Sequential(
            nn.Linear(
                in_features=2 * (hidden_dims * (timestep // 2)),
                out_features=2 * hidden_dims,
            ),
            nn.BatchNorm1d(2 * hidden_dims),
            nn.ReLU(),
            nn.Linear(in_features=2 * hidden_dims, out_features=num_classes),
        )
        self.drop = drop

    def forward(self, x):
        # normalize the input
        x = self.bn(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)

        # bs x (2 * nf)
        x3 = torch.cat([x1, x2], dim=1)
        x3 = F.dropout(x3, p=self.drop)

        logits = self.head(x3)

        return logits


class PixelLM(L.LightningModule):
    def __init__(
        self,
        num_bands,
        num_classes,
        timestep,
        hidden_dims,
        drop,
        lr,
        wd,
        milestones,
        gamma,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Pixel(num_bands, num_classes, timestep, hidden_dims, drop)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.f1_score = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )
        multistep_scheduler = {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma
            ),
            "interval": "epoch",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": multistep_scheduler,
        }

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage, log=True):
        x, y = batch["X"], batch["y"]
        # TOFO: compute pred before passing to F1Score
        logits = self(x)
        y_pred = F.softmax(logits, dim=1)

        loss = self.criterion(logits, y)
        f1_score = self.f1_score(y_pred, y)

        if log:
            self._log(loss, f1_score, stage)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def _log(self, loss, f1_score, stage):
        on_step = True if stage == "train" else False

        self.log(
            f"{stage}_loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_f1_score",
            f1_score,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path, device):
        module = cls.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
        ).to(device)
        module.eval()

        return module
