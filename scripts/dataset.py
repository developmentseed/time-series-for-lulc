import torch
import pytorch_lightning as L
import numpy as np
from torch.utils.data import Dataset, DataLoader


class S2ACube(Dataset):
    def __init__(self, cubes_dir, lc_colors, lc_klass, tfm=None):
        self.cubes = list(cubes_dir.glob("*.npz"))
        self.xyz = [cube.stem for cube in self.cubes]
        self.klass2colors = lc_colors
        self.klass2id = dict()
        self.id2label = dict()
        self.klass2label = dict()
        self.klass = list()
        for i, (k, v) in enumerate(lc_klass.items()):
            self.klass2id[k] = i
            self.id2label[i] = v
            self.klass2label[k] = v
            self.klass.append(v)
        self.id2colors = {
            id: self.klass2colors[klass] for klass, id in self.klass2id.items()
        }

    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, idx):
        data = np.load(self.cubes[idx])
        return data["X"], data["y"]


class S2ACubeDataModule(L.LightningDataModule):
    def __init__(
        self,
        cubes_dir,
        lc_colors,
        lc_klass,
        batch_size=3,
        num_workers=4,
        pin_memory=True,
        tfm=None,
    ):
        super().__init__()
        self.cubes_dir = cubes_dir
        self.lc_colors = lc_colors
        self.lc_klass = lc_klass
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tfm = tfm

    def setup(self, stage=None):
        self.train_ds = S2ACube(
            self.cubes_dir / "train", self.lc_colors, self.lc_klass, self.tfm
        )
        self.val_ds = S2ACube(
            self.cubes_dir / "val", self.lc_colors, self.lc_klass, self.tfm
        )
        self.test_ds = S2ACube(
            self.cubes_dir / "test", self.lc_colors, self.lc_klass, self.tfm
        )

    def timeseries_collate(self, data):
        Xs, ys = zip(*data)
        X = np.vstack(Xs)
        y = np.hstack(ys)

        # handle nan's
        X = torch.from_numpy(X).transpose(1, 2).float()
        X = torch.nan_to_num(X, nan=0.0)
        return {
            "X": X,
            "y": torch.from_numpy(y),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            collate_fn=self.timeseries_collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            collate_fn=self.timeseries_collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            collate_fn=self.timeseries_collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
