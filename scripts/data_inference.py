from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
# import seaborn as sns
import torch
from dataset import PixelLDM
# import matplotlib.animation as animation
# from IPython.display import HTML
from matplotlib import colors
# from torchmetrics.classification import ConfusionMatrix, F1Score, Accuracy
from model import PixelLM
# import wandb
from rasterio.features import sieve
from rasterio.merge import merge
from wikidata import S2A_LULC_CLS, S2A_LULC_COLORS

CKPT = "/home/tam/Downloads/epoch_18-step_2204-loss_0.939-f1score_0.813.ckpt"
model = PixelLM.load_eval_checkpoint(CKPT, device=torch.device("cuda"))

dm = PixelLDM(
    cubes_dir=Path("data/cubes"),
    num_bands=10,
    timestep=13,
    lc_klass=S2A_LULC_CLS,
    lc_colors=S2A_LULC_COLORS,
    batch_size=8,
    num_workers=12,
)
dm.setup("fit")
colors_list = list(dm.train_ds.id2colors.values())
colors_list = [[band / 255 for band in color] for color in colors_list]
lulc_cmap = colors.ListedColormap(colors_list)


wd = Path("/home/tam/Desktop/aoi/tuxtla")

for xyz in wd.glob("cubesxy/*.npz"):
    print("Working on", xyz)

    sample = np.load(xyz, allow_pickle=True)

    X = sample["X"]

    attrs = sample["attrs"].item()

    print(X.shape)

    h, w = X.shape[:2]

    X = X.reshape(-1, 13, 10)

    result = []
    for dat in np.split(X, h):
        dat = torch.from_numpy(dat.astype(np.float32)).transpose(1, 2).to("cuda")
        logits = model(dat)
        dat.detach()
        y_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        logits.detach()
        y_pred = y_pred.detach().cpu().numpy().astype(np.uint8)
        result.append(y_pred)

    y_pred = np.vstack(result)

    y_pred = sieve(y_pred, size=10)  # remove small polygons < 10 pixels size

    # fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(10, 30))
    # for i, ax in enumerate(axes.flatten()):
    #     if i < 13:
    #         ax.imshow((sample["X"][:,:,i,[2,1,0]]/3000))
    #     else:
    #         ax.imshow(y_pred, cmap=lulc_cmap, vmin=0, vmax=9)

    # plt.tight_layout()

    # plt.savefig("/home/tam/Desktop/lulc_inference.pdf")

    with rasterio.open(
        wd / "inferences" / f"{xyz.stem}.tif",
        "w",
        height=h,
        width=w,
        crs=attrs["crs"],
        transform=attrs["transform"],
        count=1,
        dtype="uint8",
    ) as dst:
        dst.write(y_pred, 1)
        dst.write_colormap(1, dm.train_ds.id2colors)

print("Done inferencing")

to_merge = [tif for tif in wd.glob("inferences/*.tif")]
merge(to_merge, dst_path=wd / "lulc_merged.tif")
