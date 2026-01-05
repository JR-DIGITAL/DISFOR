from pathlib import Path
from typing import List, Unpack
import json

import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from disfor.const import CLASSES
from disfor.data import GenericDataset, DatasetParams


class DisturbanceDataset(GenericDataset, Dataset):
    """PyTorch Dataset that loads image chips from stored GeoTIFFs."""

    def __init__(
        self, sample_ids: List[int] | None = None, **kwargs: Unpack[DatasetParams]
    ):
        """
        Args:
            sample_ids: List of sample_ids that should be included. Used for example to subset train and test splits
        """
        super().__init__(**kwargs)
        if self.data_folder is None:
            from disfor.data_fetcher import fetch_s2_chips

            self.tiff_folder = fetch_s2_chips()
        else:
            self.tiff_folder = Path(self.data_folder) / "tiffs"
        if sample_ids is not None:
            self.pixel_data = self.pixel_data.filter(pl.col.sample_id.is_in(sample_ids))
        samples = self.pixel_data.select(
            "label",
            path=pl.format(
                "{}/{}.tif",
                pl.col.sample_id,
                pl.col.timestamps.dt.strftime("%Y-%m-%d"),
            ),
        )

        # Pre-compute paths, labels, and chip indices to avoid string ops in __getitem__
        tiff_half_size = 32 // 2
        self.chip_range = (
            tiff_half_size - self.chip_size // 2,
            tiff_half_size + self.chip_size // 2,
        )
        self.file_paths = [self.tiff_folder / path for path in samples["path"]]
        self.labels = self.encoder.transform(samples["label"].to_list())
        match self.label_strategy:
            case "LabelEncoder":
                class_counts = np.unique_counts(self.labels).counts
                self.class_weights = torch.from_numpy(
                    class_counts.sum() / class_counts
                ).float()
            case "LabelBinarizer":
                self.class_weights = torch.from_numpy(
                    self.labels.sum() / self.labels.sum(axis=0)
                ).float()
            case "Hierarchical":
                # don't have weights yet
                self.class_weights = torch.ones(self.labels[0].shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        scale_factor = 10000
        arr = (
            tifffile.imread(self.file_paths[index])[
                self.chip_range[0] : self.chip_range[1],
                self.chip_range[0] : self.chip_range[1],
                self.band_idxs,
            ]
            / scale_factor
        )
        return {
            "image": torch.from_numpy(arr).permute(2, 0, 1).float(),
            "label": torch.tensor(self.labels[index]),
            "path": str(self.file_paths[index]),
        }

    def plot_chip(self, idx: int, ax: plt.Axes = None):
        """
        Plot a true-color visualization of the chip.

        Args:
            idx: Index of the chip to plot
            ax: Optional matplotlib axis. If None, creates new figure.
        """
        sample = self[idx]
        img = sample["image"].numpy()
        label_idx = sample["label"].numpy()

        class_id = self.encoder.inverse_transform(label_idx[np.newaxis, ...])[0]
        label_name = CLASSES[class_id]

        try:
            r = self.bands.index("B04")
            g = self.bands.index("B03")
            b = self.bands.index("B02")
        except ValueError:
            raise ValueError(
                "Bands B02, B03, B04 must be present for true-color visualization."
            )

        rgb = np.stack([img[r], img[g], img[b]], axis=-1)
        gain = 5
        rgb = np.clip(rgb * gain, 0, 1)

        if ax is None:
            plt.figure(figsize=(4, 4))
            ax = plt.gca()

        ax.imshow(rgb)
        ax.set_title(f"{class_id} - {label_name}")
        ax.axis("off")

        if ax is None:
            plt.show()


class DisturbanceDataModule(L.LightningDataModule):
    """
    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of workers for data loading.
        persist_workers (bool): If workers should persist between epochs
        train_ids (List[int]): List of sample ids to include in training. If None, train test split from the dataset will be used
        val_ids (List[int]): List of sample ids to include in validation. If None, train test split from the dataset will be used
        **kwargs: Passed on to DisturbanceDataset
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        persist_workers: bool = True,
        data_folder: str | None = None,
        train_ids: List[int] | None = None,
        val_ids: List[int] | None = None,
        **kwargs: Unpack[DatasetParams],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persist_workers = persist_workers
        self.data_folder = data_folder
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.kwargs = kwargs

        required_data = [
            "train_ids.json",
            "val_ids.json",
        ]
        if data_folder is None:
            from disfor.data_fetcher import DATA_GETTER

            base_data_paths = {
                filename: DATA_GETTER.fetch(filename) for filename in required_data
            }
        else:
            base_data_paths = {
                filename: Path(data_folder) / filename for filename in required_data
            }

        if self.train_ids is None:
            with open(base_data_paths["train_ids.json"], "r") as f:
                self.train_ids = json.load(f)
        if self.val_ids is None:
            with open(base_data_paths["val_ids.json"], "r") as f:
                self.val_ids = json.load(f)

    def setup(self, stage=None) -> None:
        """
        Setup the datasets for training and validation.

        Args:
            stage (str): Stage of the training process ('fit', 'validate').
        """
        if stage in {"fit", None}:
            self.trn_ds = DisturbanceDataset(
                sample_ids=self.train_ids, data_folder=self.data_folder, **self.kwargs
            )
            self.val_ds = DisturbanceDataset(
                sample_ids=self.val_ids, data_folder=self.data_folder, **self.kwargs
            )
            self.class_weights = self.trn_ds.class_weights

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=self.persist_workers and self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=self.persist_workers and self.num_workers > 0,
        )
