from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Literal

import numpy as np
import polars as pl

from disfor.data_fetcher import DATA_GETTER
from disfor.const import CLASSES


def generate_folds(n_folds: int, data_folder="data"):
    from sklearn.model_selection import StratifiedGroupKFold

    groups = pl.read_parquet(
        Path(data_folder) / "samples.parquet",
        columns=["sample_id", "cluster_id", "comment", "dataset", "confidence"],
        use_pyarrow=True,
    )
    # sample_ids in HRVPP not highly correlated -> use sample_id as group
    # Evoland: Group by cluster_id
    # Windthrow: Group by Wind Event
    clusters = groups.with_columns(
        cluster=pl.when(dataset="HRVPP")
        .then(pl.col.sample_id.cast(pl.String))
        .otherwise(pl.format("{}{}", pl.col.dataset, pl.col.cluster_id))
    )
    samples_w_clusters = (
        pl.read_parquet(Path(data_folder) / "labels.parquet")
        .join(clusters.select("sample_id", "cluster"), on="sample_id")
        .sort("cluster")
        .with_columns(
            pl.col.label.cast(pl.Int16),
            cluster_int=pl.col("cluster").rle_id(),
        )
    )
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    splits = sgkf.split(
        X=samples_w_clusters["label"],
        y=samples_w_clusters["label"],
        groups=samples_w_clusters["cluster_int"],
    )
    folds = {}
    sample_ids = samples_w_clusters["sample_id"].to_numpy()
    for i, (train_index, test_index) in enumerate(splits):
        folds[i] = {}
        folds[i]["train_ids"] = set(sample_ids[train_index].tolist())
        folds[i]["val_ids"] = set(sample_ids[test_index].tolist())
    return folds

class HierarchicalLabelEncoder:
    """
    Sklearn-style encoder for hierarchical multi-class labels with multi-hot encoding.

    Assumes a 3-level hierarchy where:
    - Level 1: First digit (e.g., 1xx, 2xx)
    - Level 2: First two digits (e.g., 11x, 12x, 21x)
    - Level 3: All three digits (e.g., 110, 111, 211)
    """

    def __init__(self):
        self.level1_classes_ = None
        self.level2_classes_ = None
        self.level3_classes_ = None
        self.is_fitted_ = False

    def _extract_hierarchy(self, label: int) -> Tuple[int, int, int]:
        """Extract the three hierarchy levels from a label."""
        level1 = label // 100
        level2 = label // 10
        level3 = label
        return level1, level2, level3

    def fit(self, y: List[int]) -> "HierarchicalLabelEncoder":
        """
        Fit the encoder by discovering all unique classes at each hierarchy level.

        Parameters:
        -----------
        y : List[int]
            List of integer class labels

        Returns:
        --------
        self : HierarchicalLabelEncoder
        """
        level1_set = set()
        level2_set = set()
        level3_set = set()

        for label in y:
            l1, l2, l3 = self._extract_hierarchy(label)
            level1_set.add(l1)
            level2_set.add(l2)
            level3_set.add(l3)

        # Sort to ensure consistent ordering
        self.level1_classes_ = sorted(level1_set)
        self.level2_classes_ = sorted(level2_set)
        self.level3_classes_ = sorted(level3_set)

        self.is_fitted_ = True
        return self

    def transform(self, y: List[int]) -> np.ndarray:
        """
        Transform labels to hierarchical multi-hot encoding.

        Parameters:
        -----------
        y : List[int]
            List of integer class labels

        Returns:
        --------
        encoded : np.ndarray
            Multi-hot encoded array of shape (n_samples, n_features)
            where n_features = len(level1) + len(level2) + len(level3)
        """
        if not self.is_fitted_:
            raise ValueError(
                "Encoder must be fitted before transform. Call fit() first."
            )

        n_samples = len(y)
        n_level1 = len(self.level1_classes_)
        n_level2 = len(self.level2_classes_)
        n_level3 = len(self.level3_classes_)
        n_features = n_level1 + n_level2 + n_level3

        # Create mapping dictionaries for faster lookup
        level1_map = {cls: idx for idx, cls in enumerate(self.level1_classes_)}
        level2_map = {cls: idx for idx, cls in enumerate(self.level2_classes_)}
        level3_map = {cls: idx for idx, cls in enumerate(self.level3_classes_)}

        # Initialize output array
        encoded = np.zeros((n_samples, n_features), dtype=int)

        for i, label in enumerate(y):
            l1, l2, l3 = self._extract_hierarchy(label)

            # Set corresponding bits to 1
            if l1 in level1_map:
                encoded[i, level1_map[l1]] = 1
            if l2 in level2_map:
                encoded[i, n_level1 + level2_map[l2]] = 1
            if l3 in level3_map:
                encoded[i, n_level1 + n_level2 + level3_map[l3]] = 1

        return encoded

    def fit_transform(self, y: List[int]) -> np.ndarray:
        """
        Fit the encoder and transform labels in one step.

        Parameters:
        -----------
        y : List[int]
            List of integer class labels

        Returns:
        --------
        encoded : np.ndarray
            Multi-hot encoded array
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, encoded: np.ndarray) -> List[int]:
        """
        Transform multi-hot encoded labels back to original labels.

        Parameters:
        -----------
        encoded : np.ndarray
            Multi-hot encoded array of shape (n_samples, n_features)

        Returns:
        --------
        labels : List[int]
            List of integer class labels
        """
        if not self.is_fitted_:
            raise ValueError("Encoder must be fitted before inverse_transform.")

        n_level1 = len(self.level1_classes_)
        n_level2 = len(self.level2_classes_)

        labels = []
        for row in encoded:
            # Extract active indices for each level
            level3_idx = np.where(row[n_level1 + n_level2 :] == 1)[0]

            if len(level3_idx) > 0:
                # Use the most specific level (level 3)
                label = self.level3_classes_[level3_idx[0]]
            else:
                # Fallback to level 2 or level 1 if level 3 is not set
                level2_idx = np.where(row[n_level1 : n_level1 + n_level2] == 1)[0]
                if len(level2_idx) > 0:
                    label = self.level2_classes_[level2_idx[0]] * 10
                else:
                    level1_idx = np.where(row[:n_level1] == 1)[0]
                    if len(level1_idx) > 0:
                        label = self.level1_classes_[level1_idx[0]] * 100
                    else:
                        label = 0  # Default if no class is set

            labels.append(label)

        return labels

    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the encoded output.

        Returns:
        --------
        names : List[str]
            List of feature names in format "level1_X", "level2_XX", "level3_XXX"
        """
        if not self.is_fitted_:
            raise ValueError("Encoder must be fitted before getting feature names.")

        names = []
        names.extend([f"level1_{cls}" for cls in self.level1_classes_])
        names.extend([f"level2_{cls}" for cls in self.level2_classes_])
        names.extend([f"level3_{cls}" for cls in self.level3_classes_])

        return names
