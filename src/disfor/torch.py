from pathlib import Path
from typing import List, Literal, TypedDict, Unpack
import json

import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import lightning as L
import numpy as np
import matplotlib.pyplot as plt

from disfor.const import CLASSES
from disfor.data import GenericDataset

class TiffDataset(GenericDataset, Dataset):
    """PyTorch Dataset that loads pre-processed binary data."""

    def __init__(
        self,
        # TODO implement sample id subset
        sample_ids: List[int] | None = None,
        *args,
        **kwargs
    ):
        """
        Args:
            data_folder: Path to root data folder containng pixel_data.parquet, labels.parquet and samples.parquet
                if None, the data will by dynamically downloaded from Huggingface
            sample_ids: List of sample_ids that should be included. Used for example to subset train and test splits
            target_classes: Which classes should be included
            chip_size: Size of the image chip. Maximum of 32x32
            confidence: Logged confidence of label interpretation.
            sample_datasets: Data from which sampling campaign should be included. Includes data from all by default (None)
            min_clear_percentage: Minimum percent of pixels in the chip that has to be clear (SCL in 4,5,6) to be taken.
            max_days_since_event: Either an integer specifying the maximum duration in days to the start label. This can also be set separately for each target_class.
                For example if target_classes is [110, 211] (Mature Forest, Clear Cut) we can specify a maximum number of days only for Clear Cut by passing a dictionary
                with {211: 90}
            bands: Spectral bands to include
            months: List of months to sample acquisitions from. January is 1, December is 12.
            omit_border: Omit samples which have "border" in the comment. These are usually samples where the sample is a mixed pixel
            omit_low_tcd: Omit samples which have "TCD" in the comment. These are usually samples where the forest has a low tree cover density (for example olive plantations)
        """
        super().__init__(*args, **kwargs)
        if self.data_folder is None:
            from disfor.data_fetcher import fetch_s2_chips
            self.tiff_folder = fetch_s2_chips()
        else:
            self.tiff_folder = Path(self.data_folder) / "tiffs"
        if sample_ids is not None:
            self.pixel_data = self.pixel_data.filter(pl.col.sample_id.is_in(sample_ids))
        samples = (
            self.pixel_data
            .select(
                "label",
                path=pl.format(
                    "{}/{}.tif",
                    pl.col.sample_id,
                    pl.col.timestamps.dt.strftime("%Y-%m-%d"),
                ),
            )
        )

        # Pre-compute paths, labels, and chip indices to avoid string ops in __getitem__
        tiff_half_size = 32 // 2
        self.chip_range = (
            tiff_half_size - self.chip_size // 2,
            tiff_half_size + self.chip_size // 2,
        )
        self.file_paths = [self.tiff_folder / path for path in samples["path"]]
        self.labels = self.encoder.transform(samples["label"])
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

    def __getitem__(self, idx):
        scale_factor = 10000
        arr = (
            tifffile.imread(self.file_paths[idx])[
                self.chip_range[0] : self.chip_range[1],
                self.chip_range[0] : self.chip_range[1],
                self.band_idxs,
            ]
            / scale_factor
        )
        return {
            "image": torch.from_numpy(arr).permute(2, 0, 1).float(),
            "label": torch.tensor(self.labels[idx]),
            "path": str(self.file_paths[idx]),
        }

    def plot_chip(self, idx: int):
        """
        Plot a true-color visualization (B04, B03, B02) of the chip
        with the label as the title.
        """
        sample = self[idx]
        img = sample["image"].numpy()  # shape: (C, H, W)
        label_idx = sample["label"].numpy()

        # Map encoded label back to original class id + name
        class_id = self.encoder.inverse_transform(label_idx[np.newaxis, ...])[0]
        label_name = CLASSES[class_id]

        # original band order in self.bands
        try:
            r = self.bands.index("B04")
            g = self.bands.index("B03")
            b = self.bands.index("B02")
        except ValueError:
            raise ValueError(
                "Bands B02, B03, B04 must be present for true-color visualization."
            )

        rgb = np.stack([img[r], img[g], img[b]], axis=-1)

        # Normalize for display
        gain = 5
        rgb = np.clip(rgb * gain, 0, 1)

        plt.figure(figsize=(4, 4))
        plt.imshow(rgb)
        plt.title(f"{class_id} - {label_name}")
        plt.axis("off")
        plt.show()


class TiffDataModule(L.LightningDataModule):
    """
    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of workers for data loading.
        persist_workers (bool): If workers should persist between epochs
        train_ids (List[int]): List of sample ids to include in training. If None, train test split from the dataset will be used
        val_ids (List[int]): List of sample ids to include in validation. If None, train test split from the dataset will be used
        **kwargs: Passed on to TiffDataset
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        persist_workers: bool = True,
        data_folder: str | None = None,
        train_ids: List[int] | None = None,
        val_ids: List[int] | None = None,
        **kwargs,
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

    def setup(self, stage=None):
        """
        Setup the datasets for training and validation.

        Args:
            stage (str): Stage of the training process ('fit', 'validate',
            etc.).
        """
        if stage in {"fit", None}:
            self.trn_ds = TiffDataset(sample_ids=self.train_ids, data_folder=self.data_folder, **self.kwargs)
            self.val_ds = TiffDataset(sample_ids=self.val_ids, data_folder=self.data_folder, **self.kwargs)
            self.class_weights = self.trn_ds.class_weights

    def train_dataloader(self):
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

    def val_dataloader(self):
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
