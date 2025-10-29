from pathlib import Path
from typing import List, Literal
import json

import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import lightning as L
import numpy as np
import matplotlib.pyplot as plt

CLASSES = {
    100: "Alive Vegetation",
    110: "Mature Forest",
    120: "Revegetation",
    121: "With Trees (after clear cut)",
    122: "Canopy closing (after thinning/defoliation)",
    123: "Without Trees (shrubs and grasses, no reforestation visible)",
    200: "Disturbed",
    210: "Planned",
    211: "Clear Cut",
    212: "Thinning",
    213: "Forestry Mulching (Non Forest Vegetation Removal)",
    220: "Salvage",
    221: "After Biotic Disturbance",
    222: "After Abiotic Disturbance",
    230: "Biotic",
    231: "Bark Beetle (with decline)",
    232: "Gypsy Moth (temporary)",
    240: "Abiotic",
    241: "Drought",
    242: "Wildfire",
    244: "Wind",
    245: "Avalanche",
    246: "Flood",
}


class TiffDataset(Dataset):
    """PyTorch Dataset that loads pre-processed binary data."""

    def __init__(
        self,
        data_folder: str,
        sample_ids: List[int] | None = None,
        target_classes: List[
            Literal[
                100,
                110,
                120,
                121,
                122,
                123,
                200,
                210,
                211,
                212,
                213,
                220,
                221,
                222,
                230,
                231,
                232,
                240,
                241,
                242,
                244,
                245,
                246,
            ]
        ]
        | None = None,
        label_strategy: Literal[
            "LabelEncoder", "LabelBinarizer", "Hierarchical"
        ] = "LabelEncoder",
        chip_size: Literal[32, 16, 8, 4] = 32,
        confidence: List[Literal["high", "medium"]] | None = None,
        sample_datasets: List[Literal["Evoland", "HRVPP", "Windthrow"]] | None = None,
        min_clear_percentage: int = 99,
        max_days_since_event: int | dict | None = None,
        bands: List[
            Literal[
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B11",
                "B12",
                "SCL",
            ]
        ]
        | None = None,
        months: List[Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] | None = None,
        omit_border: bool = False,
        omit_low_tcd: bool = False,
    ):
        """
        Args:
            data_file: Path to the binary file containing data and labels
            sample_ids: List of sample_ids that should be included. Used for example to subset train and test splits
            target_classes: Which classes should be included
            chip_size: Size of the image chip. Maximum of 32x32
            confidence: Logged confidence of label interpretation. Either high or medium
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
        super().__init__()
        self.bands = bands or [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ]
        self.chip_size = chip_size

        # redo this, we don't need to filter at all if these are None
        self.target_classes = target_classes or list(CLASSES.keys())
        match label_strategy:
            case "LabelEncoder":
                from sklearn.preprocessing import LabelEncoder

                self.encoder = LabelEncoder()
            case "LabelBinarizer":
                from sklearn.preprocessing import LabelBinarizer

                self.encoder = LabelBinarizer()

        self.encoder.fit(self.target_classes)

        all_bands = [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
            "SCL",
        ]
        self.band_idxs = [all_bands.index(band) for band in self.bands]

        if months is None:
            months = list(range(1, 13))
        # Add quality filters
        group_filters = [pl.lit(True)]
        if sample_ids is not None:
            group_filters.append(pl.col.sample_id.is_in(sample_ids))
        if confidence is not None:
            group_filters.append(pl.col.confidence.is_in(confidence))
        if sample_datasets is not None:
            group_filters.append(pl.col.dataset.is_in(sample_datasets))
        if omit_low_tcd:
            group_filters.append(~pl.col.comment.str.contains("TCD"))
        if omit_border:
            group_filters.append(~pl.col.comment.str.contains("border"))

        # Load and filter groups data
        groups = (
            pl.read_parquet(
                Path(data_folder) / "samples.parquet",
                columns=["sample_id", "cluster_id", "comment", "dataset", "confidence"],
                use_pyarrow=True,
            )
            .with_columns(
                cluster_id_int=pl.col("cluster_id").rle_id(),
            )
            .filter(group_filters)
        )

        max_duration_filters = [pl.lit(False)]
        match max_days_since_event:
            case dict():
                for label, days in max_days_since_event.items():
                    if days is None:
                        continue
                    max_duration_filters.append(
                        (pl.col.duration_since_last_flag > pl.duration(days=days))
                        & (pl.col.label == label)
                    )
            case int():
                max_duration_filters.append(
                    pl.col.duration_since_last_flag
                    > pl.duration(days=max_days_since_event)
                )

        labels = pl.read_parquet(
            Path(data_folder) / "labels.parquet",
            columns=["sample_id", "label", "start"],
        ).with_columns(
            # TODO: fix this in the data+data pipeline
            pl.col.label.cast(pl.UInt16)
        )

        # get appropriate clear column for chip size
        clear_column = f"percent_clear_{chip_size}x{chip_size}"
        filtered_dates = (
            pl.read_parquet(
                Path(data_folder) / "pixel_data.parquet",
                columns=["sample_id", "label", "timestamps", clear_column],
            )
            .join(groups, left_on="sample_id", right_on="sample_id", how="inner")
            .join(labels, on=["sample_id", "label"], how="inner")
            .with_columns(
                duration_since_last_flag=(pl.col("timestamps") - pl.col("start")),
                path=pl.format(
                    "{}/{}/{}.tif",
                    pl.col.sample_id,
                    pl.col.label,
                    pl.col.timestamps.dt.strftime("%Y-%m-%d"),
                ),
            )
            .filter(
                pl.col("label").is_in(self.target_classes),
                pl.col(clear_column) > min_clear_percentage,
                pl.col("timestamps").dt.month().is_in(months),
                ~pl.any_horizontal(max_duration_filters),
            )
        )

        self.tiff_folder = Path(data_folder) / "tiffs"
        samples = filtered_dates.select(
            "label",
            "path",
        )

        # Pre-compute paths, labels, and chip indices to avoid string ops in __getitem__
        tiff_half_size = 32 // 2
        self.chip_range = (
            tiff_half_size - self.chip_size // 2,
            tiff_half_size + self.chip_size // 2,
        )
        self.file_paths = [self.tiff_folder / path for path in samples["path"]]
        self.labels = self.encoder.transform(samples["label"])
        match label_strategy:
            case "LabelEncoder":
                class_counts = np.unique_counts(self.labels).counts
                self.class_weights = torch.from_numpy(
                    class_counts.sum() / class_counts
                ).float()
            case "LabelBinarizer":
                self.class_weights = torch.from_numpy(
                    self.labels.sum() / self.labels.sum(axis=0)
                ).float()

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
            "label": torch.tensor(self.labels[idx]).float(),
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
        metadata_path (str): Path to the metadata file for normalization
        statistics.
    """

    def __init__(
        self,
        batch_size,
        num_workers,
        persist_workers=True,
        classes=None,
        label_strategy: Literal[
            "LabelEncoder", "LabelBinarizer", "Hierarchical"
        ] = "LabelBinarizer",
        train_ids=None,
        val_ids=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = classes if classes is not None else [110, 211, 231, 242, 244]
        self.persist_workers = persist_workers
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.label_strategy = label_strategy
        if self.train_ids is None:
            with open("./data/train_ids.json", "r") as f:
                self.train_ids = json.load(f)
        if self.val_ids is None:
            with open("./data/val_ids.json", "r") as f:
                self.val_ids = json.load(f)

    def setup(self, stage=None):
        """
        Setup the datasets for training and validation.

        Args:
            stage (str): Stage of the training process ('fit', 'validate',
            etc.).
        """
        if stage in {"fit", None}:
            self.trn_ds = TiffDataset(
                data_folder="./data/",
                target_classes=self.classes,
                sample_ids=self.train_ids,
                label_strategy=self.label_strategy,
                confidence=["high"],
                max_days_since_event={211: 90},
                omit_border=True,
                omit_low_tcd=True,
            )
            self.val_ds = TiffDataset(
                data_folder="./data/",
                target_classes=self.classes,
                sample_ids=self.val_ids,
                label_strategy=self.label_strategy,
                confidence=["high"],
                max_days_since_event={211: 90},
                omit_border=True,
                omit_low_tcd=True,
            )
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
