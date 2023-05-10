import torch
import lightning.pytorch as L
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PixelDS(Dataset):
    def __init__(self, cubes_dir, num_bands, timestep, lc_colors, lc_klass, tfm=None):
        self.cubes = list(cubes_dir.glob("*.npz"))
        self.timestep = timestep
        self.num_bands = num_bands
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
        data = np.load(self.cubes[idx], allow_pickle=True)
        # data["X"] = w, h, time, band; data["y"] = w, h
        X, y = data["X"], data["y"]
        X = X.reshape(-1, self.timestep, self.num_bands)
        y = y.reshape(-1)

        y = self.label_tfm(y)
        return (X, y)

    def label_tfm(self, y):
        _y = np.copy(y)
        for k, v in self.klass2id.items():
            _y[y == k] = v
        return _y


class PixelLDM(L.LightningDataModule):
    def __init__(
        self,
        cubes_dir,
        num_bands,
        timestep,
        lc_colors,
        lc_klass,
        batch_size,
        num_workers,
        pin_memory=True,
        tfm=None,
    ):
        super().__init__()
        self.cubes_dir = cubes_dir
        self.num_bands = num_bands
        self.timestep = timestep
        self.lc_colors = lc_colors
        self.lc_klass = lc_klass
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tfm = tfm

    def setup(self, stage=None):
        self.train_ds = PixelDS(
            self.cubes_dir / "train",
            self.num_bands,
            self.timestep,
            self.lc_colors,
            self.lc_klass,
            self.tfm,
        )
        self.val_ds = PixelDS(
            self.cubes_dir / "val",
            self.num_bands,
            self.timestep,
            self.lc_colors,
            self.lc_klass,
            self.tfm,
        )
        self.test_ds = PixelDS(
            self.cubes_dir / "test",
            self.num_bands,
            self.timestep,
            self.lc_colors,
            self.lc_klass,
            self.tfm,
        )

    def timeseries_collate(self, data):
        Xs, ys = zip(*data)
        X = np.vstack(Xs)
        y = np.hstack(ys)

        # handle nan's
        X = torch.from_numpy(X.astype(np.float32)).transpose(
            1, 2
        )  # => (pixel, band, time)
        # X = torch.nan_to_num(X, nan=0.0) - handled in new dataset pre-processing
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
