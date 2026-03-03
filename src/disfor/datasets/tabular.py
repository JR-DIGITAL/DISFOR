from typing import Unpack
import polars as pl

from .generic import GenericDataset, DatasetParams


class TabularDataset(GenericDataset):
    """Class providing data for sklearn style models

    For usage see the [dataloaders usage page](../usage/dataloaders).

    Args:
        **kwargs: keyword arguments being passed to [disfor.datasets.GenericDataset][]
    """

    def __init__(self, **kwargs: Unpack[DatasetParams]):
        super().__init__(**kwargs)
        train_df = self.pixel_data.filter(pl.col.sample_id.is_in(self._train_ids))
        test_df = self.pixel_data.filter(pl.col.sample_id.is_in(self._val_ids))

        # Train
        self.X_train = train_df[self.bands].to_numpy(writable=True)
        self.y_train = train_df["label_encoded"].to_numpy(writable=True)
        self.group_train = train_df["cluster_id_encoded"].to_numpy(writable=True)
        # Test
        self.X_test = test_df[self.bands].to_numpy(writable=True)
        self.y_test = test_df["label_encoded"].to_numpy(writable=True)
        self.group_test = test_df["cluster_id_encoded"].to_numpy(writable=True)
