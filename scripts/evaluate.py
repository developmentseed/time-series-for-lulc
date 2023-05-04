from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
import wandb
from torchmetrics.classification import ConfusionMatrix, F1Score, Accuracy
from model import PixelLM
from dataset import PixelLDM
from wikidata import S2A_LULC_CLS, S2A_LULC_COLORS


@torch.no_grad()
def log_metrics(dm, model, device="cuda", wandb=False):
    cmat = ConfusionMatrix(
        task="multiclass",
        num_classes=len(dm.train_ds.klass),
        normalize="true",
    ).to(device)
    f1 = F1Score(
        task="multiclass",
        num_classes=len(dm.train_ds.klass),
        average="none",
    ).to(device)
    acc = Accuracy(
        task="multiclass",
        num_classes=len(dm.train_ds.klass),
        average="none",
    ).to(device)

    val_dl = dm.val_dataloader()
    for idx, batch in enumerate(val_dl):
        batch["X"] = batch["X"].to(device=device)
        batch["y"] = batch["y"].to(device=device)
        logits = model(batch["X"])

        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        mask = batch["y"]

        # Confusion Matrix
        cmat(pred.view(-1), mask.view(-1))

        # F1 Score
        f1(logits, mask)

        # Accuracy
        acc(logits, mask)

    # Confusion Matrix
    cmat = cmat.compute().detach().cpu().numpy()
    f1 = f1.compute().detach().cpu().numpy()
    acc = acc.compute().detach().cpu().numpy()

    sns.set(font_scale=0.6)
    heatmap = sns.heatmap(
        cmat,
        annot=True,
        cmap="bwr",
        fmt=".2f",
        xticklabels=dm.train_ds.klass,
        yticklabels=dm.train_ds.klass,
    )
    fig = heatmap.get_figure()
    fig.savefig("heatmap.png")

    scores = {k: [v1, v2] for k, v1, v2 in zip(dm.train_ds.klass, f1, acc)}
    scores = pd.DataFrame(scores, index=["F1 Score", "Accuracy"]).reset_index()
    print(scores)

    # Log to wandb
    if wandb:
        wandb.log({"Confusion Matrix": wandb.Image("heatmap.png")})
        wandb.log({"scores": wandb.Table(dataframe=scores)})


if __name__ == "__main__":
    CKPT = "logs/pixel/baseline/version_0/checkpoints/epoch:19-step:2320-loss:0.936-f1score:0.814.ckpt"

    dm = dm = PixelLDM(
        cubes_dir=Path("data/cubes"),
        lc_klass=S2A_LULC_CLS,
        lc_colors=S2A_LULC_COLORS,
        batch_size=8,
        num_workers=12,
    )
    dm.setup(stage="fit")

    model = PixelLM.load_eval_checkpoint(CKPT, device=torch.device("cuda"))

    log_metrics(dm, model)
