from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict, field
import json
from typing import List, Dict, Optional, Literal
import polars as pl
from sklearn.preprocessing import LabelEncoder
import numpy as np


@dataclass
class ForestDisturbanceData:
    """Combined configuration and data preparation class"""

    # Class selection
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
    class_mapping_overrides: Dict[int, str] = field(
        default_factory=lambda: {}
    )

    # Filtering parameters
    confidence: List[Literal["high", "medium"]] | None = None
    valid_scl_values: List[Literal[0,1,2,3,4,5,6,7,8,9,10,11]] = field(
        default_factory=lambda: [4,5,6])
    months: List[Literal[1,2,3,4,5,6,7,8,9,10,11,12]] | None = None
    max_days_since_event: int | dict | None = None,
    sample_datasets: List[Literal["Evoland", "HRVPP", "Windthrow"]] | None = None

    # Sampling parameters
    max_samples_per_event: Optional[int] = None
    random_seed: Optional[int] = None

    # Balanced sampling parameters
    use_balanced_sampling: bool = False
    balance_method: str = "downsample_majority"
    target_majority_samples: Optional[int] = None

    # Quality filters
    omit_low_tcd: bool = True
    omit_border: bool = True

    # Feature selection
    bands: List[str] = field(
        default_factory=lambda: [
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
    )

    # Outlier removal parameters
    remove_outliers: bool = False
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    outlier_columns: Optional[List[str]] = None

    # Data path
    data_folder: str = "data/"

    # Private fields that will be populated after processing
    X_train: np.ndarray = field(init=False)
    y_train: np.ndarray = field(init=False)
    X_test: np.ndarray = field(init=False)
    y_test: np.ndarray = field(init=False)
    label_encoder: LabelEncoder = field(init=False)

    # Private data loading fields
    _class_mapping: Dict = field(init=False)
    _train_ids: List = field(init=False)
    _val_ids: List = field(init=False)

    def __post_init__(self):
        """Initialize default feature columns and process data"""
        # Process the data and populate the final attributes
        self._process_data()

    def _process_data(self):
        """Main processing pipeline - loads data and creates final arrays"""
        self._load_base_data()
        train_df, test_df = self._prepare_dataframes()
        train_df = self._apply_balanced_sampling(train_df)

        # Train
        self.X_train = train_df[self.bands].to_numpy(writable=True)
        self.y_train = train_df["labels_encoded"].to_numpy(writable=True)
        self.group_train = train_df["cluster_id_encoded"].to_numpy(writable=True)
        # Test
        self.X_test = test_df[self.bands].to_numpy(writable=True)
        self.y_test = test_df["labels_encoded"].to_numpy(writable=True)
        self.group_test = test_df["cluster_id_encoded"].to_numpy(writable=True)

    def _load_base_data(self):
        """Load base data files"""
        base_path = Path(self.data_folder)

        with open(base_path / "classes.json", "r") as f:
            self._class_mapping = {int(k): v for k, v in json.load(f).items()}

        with open(base_path / "train_ids.json", "r") as f:
            self._train_ids = json.load(f)

        with open(base_path / "val_ids.json", "r") as f:
            self._val_ids = json.load(f)

    def _prepare_dataframes(self):
        """Load and prepare dataframes according to configuration"""
        base_path = Path(self.data_folder)
        max_duration_filters = [pl.lit(False)]
        match self.max_days_since_event:
            case dict():
                for label, days in self.max_days_since_event.items():
                    if days is None:
                        continue
                    max_duration_filters.append(
                        (pl.col.duration_since_last_flag > pl.duration(days=days))
                        & (pl.col.label == label)
                    )
            case int():
                max_duration_filters.append(
                    pl.col.duration_since_last_flag
                    > pl.duration(days=self.max_days_since_event)
                )

        labels = pl.read_parquet(
            Path(base_path) / "labels.parquet",
            columns=["sample_id", "label", "start"],
        ).with_columns(
            # TODO: fix this in the data+data pipeline
            pl.col.label.cast(pl.UInt16)
        )

        # Load and filter pixel data
        signal_data = (
            pl.read_parquet(base_path / "pixel_data.parquet")
            .join(labels, on=["sample_id", "label"], how="inner")
            .with_columns(
                pl.col.labels.replace(
                    self.class_mapping_overrides, return_dtype=pl.String
                ),
                duration_since_last_flag=(pl.col("timestamps") - pl.col("start")),
            )
            .filter(
                pl.col("labels").is_in(self.target_classes),
                pl.col("timestamps").dt.month().is_in(self.months),
                pl.col.SCL.is_in(self.valid_scl_values),
                ~pl.any_horizontal(max_duration_filters),
            )
        )

        group_filters = [pl.lit(True)]

        # Add quality filters
        if self.confidence is not None:
            group_filters.append(pl.col.confidence.is_in(self.confidence))
        if self.sample_datasets is not None:
            group_filters.append(pl.col.dataset.is_in(self.sample_datasets))
        if self.omit_low_tcd:
            group_filters.append(~pl.col.comment.str.contains("TCD"))
        if self.omit_border:
            group_filters.append(~pl.col.comment.str.contains("border"))

        # Load and filter groups data
        groups = (
            pl.read_parquet(
                base_path / "samples.parquet",
                columns=["sample_id", "cluster_id", "comment", "dataset", "confidence"],
                use_pyarrow=True,
            )
            .with_columns(
                cluster_id_int=pl.col("cluster_id").rle_id(),
            )
            .filter(group_filters)
        )

        # Join signal data with groups
        signal_data_with_cluster = signal_data.join(
            groups, left_on="sample_id", right_on="sample_id", how="inner"
        ).with_columns(
            pl.col("cluster_id").rank("dense").cast(pl.Int64).name.suffix("_encoded")
        )

        signal_data_with_cluster, _ = self._remove_outliers(signal_data_with_cluster)
        signal_data_with_cluster = signal_data_with_cluster.sort(
            ["sample_id", "timestamps"]
        )

        # Create label encoder
        le = LabelEncoder()
        le.fit(signal_data_with_cluster["labels"])
        self.label_encoder = le

        # Split into train and test sets
        train_data_pl = signal_data_with_cluster.filter(
            pl.col.sample_id.is_in(self._train_ids)
        )
        test_data_pl = signal_data_with_cluster.filter(
            pl.col.sample_id.is_in(self._val_ids)
        )
        self.train_data_pl = train_data_pl

        # Apply sampling if configured
        if self.max_samples_per_event is not None:
            if self.random_seed is not None:
                pl.set_random_seed(self.random_seed)

            train_data_pl = train_data_pl.filter(
                pl.int_range(pl.len()).over(["sample_id", "labels"])
                < self.max_samples_per_event
            )

        # Prepare final dataframes with encoded labels
        train_df = train_data_pl.with_columns(
            pl.Series(
                "labels_encoded", le.transform(train_data_pl["labels"].to_list())
            ),
        )

        test_df = test_data_pl.with_columns(
            pl.Series("labels_encoded", le.transform(test_data_pl["labels"].to_list())),
        )

        return train_df, test_df

    def _apply_balanced_sampling(self, train_df: pl.DataFrame):
        """Apply balanced sampling by downsampling the majority class"""
        if not self.use_balanced_sampling:
            return train_df

        # print("Applying balanced sampling by downsampling majority class...")

        # Get class distribution
        class_counts = (
            train_df.group_by("labels_encoded")
            .agg(pl.count().alias("count"))
            .sort("labels_encoded")
        )

        # Find majority class and determine target size
        counts = [row[1] for row in class_counts.iter_rows()]
        class_ids = [row[0] for row in class_counts.iter_rows()]

        max_count = max(counts)
        majority_class_id = class_ids[counts.index(max_count)]

        # Determine target size for majority class
        if self.target_majority_samples is not None:
            target_size = self.target_majority_samples
        else:
            # Use 2x the second largest class, or 10k if that's smaller
            other_counts = [c for c in counts if c != max_count]
            if other_counts:
                second_largest = max(other_counts)
                target_size = min(max_count, max(second_largest * 2, 500))
            else:
                target_size = min(max_count, 500)

        # print(f"Downsampling majority class (ID {majority_class_id}) from {max_count:,} to {target_size:,}")

        # Set random seed
        if self.random_seed is not None:
            pl.set_random_seed(self.random_seed)

        # Downsample majority class
        majority_samples = train_df.filter(
            pl.col("labels_encoded") == majority_class_id
        )
        minority_samples = train_df.filter(
            pl.col("labels_encoded") != majority_class_id
        )

        # Sample from majority class
        downsampled_majority = majority_samples.sample(
            n=min(target_size, len(majority_samples))
        )

        # Combine with minority classes
        balanced_df = pl.concat([minority_samples, downsampled_majority])

        # Print new class distribution
        new_class_counts = (
            balanced_df.group_by("labels_encoded")
            .agg(pl.count().alias("count"))
            .sort("labels_encoded")
        )
        total_samples = 0
        for row in new_class_counts.iter_rows():
            class_id, count = row
            total_samples += count

        return balanced_df

    def _remove_outliers(self, df: pl.DataFrame):
        """Remove statistical outliers from specified feature columns per target class"""
        if not self.remove_outliers:
            return df, {"outliers_removed": 0, "total_samples": len(df)}

        # print(f"Removing outliers using {self.outlier_method} method per target class...")

        # Determine columns to check for outliers
        outlier_cols = (
            self.outlier_columns
            if self.outlier_columns is not None
            else self.bands
        )

        original_len = len(df)
        cleaned_dfs = []
        all_removal_stats = {}
        total_outliers_removed = 0

        # Process each target class separately
        for label in df["labels"].unique().to_list():
            class_df = df.filter(pl.col("labels") == label)
            class_original_len = len(class_df)

            if class_original_len == 0:
                continue

            outlier_mask = pl.lit(False)
            class_outlier_stats = {}

            if self.outlier_method == "iqr":
                # IQR method: Q1 - threshold * IQR, Q3 + threshold * IQR
                for col in outlier_cols:
                    q1 = class_df[col].quantile(0.25)
                    q3 = class_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - self.outlier_threshold * iqr
                    upper_bound = q3 + self.outlier_threshold * iqr

                    col_outliers = (pl.col(col) < lower_bound) | (
                        pl.col(col) > upper_bound
                    )
                    outlier_mask = outlier_mask | col_outliers

                    class_outlier_stats[col] = {
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    }

            elif self.outlier_method == "zscore":
                # Z-score method: |z| > threshold
                for col in outlier_cols:
                    mean_val = class_df[col].mean()
                    std_val = class_df[col].std()

                    if std_val > 0:  # Avoid division by zero
                        z_score = ((pl.col(col) - mean_val) / std_val).abs()
                        col_outliers = z_score > self.outlier_threshold
                        outlier_mask = outlier_mask | col_outliers

                        class_outlier_stats[col] = {
                            "mean": mean_val,
                            "std": std_val,
                            "threshold": self.outlier_threshold,
                        }

            elif self.outlier_method == "modified_zscore":
                # Modified Z-score using median absolute deviation
                for col in outlier_cols:
                    median_val = class_df[col].median()
                    mad = (class_df[col] - median_val).abs().median()

                    if mad > 0:  # Avoid division by zero
                        modified_z = 0.6745 * (pl.col(col) - median_val).abs() / mad
                        col_outliers = modified_z > self.outlier_threshold
                        outlier_mask = outlier_mask | col_outliers

                        class_outlier_stats[col] = {
                            "median": median_val,
                            "mad": mad,
                            "threshold": self.outlier_threshold,
                        }

            else:
                raise ValueError(f"Unknown outlier method: {self.outlier_method}")

            # Remove outliers for this class
            class_cleaned_df = class_df.filter(~outlier_mask)
            class_outliers_removed = class_original_len - len(class_cleaned_df)
            total_outliers_removed += class_outliers_removed

            cleaned_dfs.append(class_cleaned_df)

            # Store stats for this class
            all_removal_stats[label] = {
                "outliers_removed": class_outliers_removed,
                "total_samples": class_original_len,
                "samples_remaining": len(class_cleaned_df),
                "removal_percentage": (class_outliers_removed / class_original_len)
                * 100
                if class_original_len > 0
                else 0,
                "column_stats": class_outlier_stats,
            }

            # if class_outliers_removed > 0:
            # print(f"  {label}: Removed {class_outliers_removed:,} outliers ({all_removal_stats[label]['removal_percentage']:.2f}%) from {class_original_len:,} samples")

        # Combine all cleaned class dataframes
        cleaned_df = (
            pl.concat(cleaned_dfs) if cleaned_dfs else df.filter(pl.lit(False))
        )  # Empty dataframe with same schema

        removal_stats = {
            "outliers_removed": total_outliers_removed,
            "total_samples": original_len,
            "samples_remaining": len(cleaned_df),
            "removal_percentage": (total_outliers_removed / original_len) * 100
            if original_len > 0
            else 0,
            "method": self.outlier_method,
            "threshold": self.outlier_threshold,
            "columns_checked": outlier_cols,
            "per_class_stats": all_removal_stats,
        }

        # print(f"Total outliers removed: {total_outliers_removed:,} ({removal_stats['removal_percentage']:.2f}%) from {original_len:,} samples")
        # print(f"Total remaining samples: {len(cleaned_df):,}")

        return cleaned_df, removal_stats

    def to_dict(self):
        config_dict = asdict(self)
        # Remove the processed data fields from hashing
        config_dict.pop("X_train", None)
        config_dict.pop("y_train", None)
        config_dict.pop("X_test", None)
        config_dict.pop("y_test", None)
        config_dict.pop("label_encoder", None)
        config_dict.pop("_class_mapping", None)
        config_dict.pop("_train_ids", None)
        config_dict.pop("_val_ids", None)
        return config_dict

    def get_config_hash(self) -> str:
        """Generate a hash of the configuration for unique identification"""
        # Only hash the configuration parameters, not the processed data
        config_dict = self.to_dict()

        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
