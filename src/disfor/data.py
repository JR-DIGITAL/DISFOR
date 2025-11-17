from pathlib import Path
import json
from typing import TypedDict, Literal, List, Dict, NotRequired, Unpack
import polars as pl

from disfor.data_fetcher import DATA_GETTER
from disfor.const import CLASSES


class DatasetParams(TypedDict, total=False):
    """TypedDict for dataset configuration parameters.

    All fields are optional (using total=False), allowing partial parameter passing.
    """

    # Data source
    data_folder: NotRequired[str | None]

    # Class selection
    target_classes: NotRequired[
        List[
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
                243,
                244,
                245,
            ]
        ]
        | None
    ]
    class_mapping_overrides: NotRequired[Dict[int, int] | None]
    label_strategy: NotRequired[
        Literal["LabelEncoder", "LabelBinarizer", "Hierarchical"]
    ]

    # Filtering parameters
    confidence: NotRequired[List[Literal["high", "medium"]] | None]

    # Cloud masking parameters
    valid_scl_values: NotRequired[
        List[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] | None
    ]
    chip_size: NotRequired[Literal[32, 16, 8, 4]]
    min_clear_percentage_chip: NotRequired[int | None]
    months: NotRequired[List[Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] | None]
    max_days_since_event: NotRequired[int | dict | None]
    sample_datasets: NotRequired[List[Literal["Evoland", "HRVPP", "Windthrow"]] | None]

    # Sampling parameters
    max_samples_per_event: NotRequired[int | None]
    random_seed: NotRequired[int | None]

    # Balanced sampling parameters
    apply_downsampling: NotRequired[bool]
    target_majority_samples: NotRequired[int | None]

    # Quality filters
    omit_low_tcd: NotRequired[bool]
    omit_border: NotRequired[bool]

    # Feature selection
    bands: NotRequired[
        List[
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
        | None
    ]

    # Outlier removal parameters
    remove_outliers: NotRequired[bool]
    outlier_method: NotRequired[Literal["iqr", "zscore", "modified_zscore"]]
    outlier_threshold: NotRequired[float]
    outlier_columns: NotRequired[
        List[
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
        | None
    ]


class GenericDataset:
    """
    A generic class which serves to load, filter and pre-process the raw data.

    There are two classes which then bring the filtered and pre-processed data into formats which
    can be used with pytorch ([`disfor.torch.DisturbanceDataset`][]) and sklearn style classifiers ([`disfor.data.ForestDisturbanceData`][]) respectively.

    Args:
        data_folder: Path to root data folder containng pixel_data.parquet, labels.parquet and samples.parquet,
            if not specified, data will be fetched using [`disfor.data_fetcher.DATA_GETTER`][]
        target_classes: Which classes should be included
        class_mapping_overrides: Map classes to other classes for example {221: 211, 222: 211} would map both of the salvage classes to clear cut.
            This remapping happens before filtering of `target_classes`. This means that the items of the dict need to be specified in target_classes,
            otherwise they will be filtered out.
        confidence: Filters dataset to only include logged confidence of label interpretation.
        valid_scl_values: List of valid SCL values. Used to filter out cloudy or otherwise unusable observations
        chip_size: Size of the image chip. Maximum of 32x32. Used in `min_clear_percentage_chip`.
        min_clear_percentage_chip: Minimum percent (0-100) of pixels in the chip that has to be clear (SCL in 4,5,6) to be included.
        months: List of months to include acquisitions from. January is 1, December is 12.
        max_days_since_event: Either an integer specifying the maximum duration in days to the start label. This can also be set separately for each target_class.
            For example if target_classes is [110, 211] (Mature Forest, Clear Cut) we can specify a maximum number of 90 days after a Clear Cut by passing a dictionary
            with {211: 90}
        sample_datasets: Data from which sampling campaign should be included. Includes data from all by default (None)
        max_samples_per_event: Maximum number of acquisitions to include per event. Can be used to reduce number of samples
            drawn from segments with long durations. For example to reduce the number of healthy acquistions
        random_seed: Random seed used for reproducible subsampling operations
        apply_downsampling: Flag if downsampling sampling of the majority class should be used.
        target_majority_samples: How many samples the majority class should have after balancing. If None, the majority class will be reduced
            to 2 times the samples of the second largest class, or 500, whichever is less.
        omit_border: Omit samples which have "border" in the comment. These are usually samples where the sample is a mixed pixel
        omit_low_tcd: Omit samples which have "TCD" in the comment. These are usually samples where the forest has a low tree cover density (for example olive plantations)
        bands: Spectral bands to include
        remove_outliers: Flag if outliers should be removed. This is used to remove clouds or other data artifacts
            which were not masked through the SCL values.
        outlier_method: Statistical method used to determine outliers. This statistical measure is calculated for each unique
            `(sample_id, label)` group.
        outlier_threshold: Which threshold to apply, acquisitions greater than `(outlier_threshold*outlier_method)` will be removed.
        outlier_columns: Which columns (bands) to search for outliers. If an outlier is detected in any of the bands
            it will be removed. Default is all bands which are defined in the parameter `bands`
        label_strategy: How the values in `target_classes` should be encoded. LabelEncoder and LabelBinarizer correspond to the sklearn encoders.
            Hierarchical is a custom encoding implemented for the hierarchical classes. For more details see [`disfor.utils.HierarchicalLabelEncoder`][]

    Attributes:
        pixel_data: Polars dataframe containing filtered and pre-processed data.
    """

    def __init__(
        self,
        #
        data_folder: str | None = None,
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
                243,
                244,
                245,
            ]
        ]
        | None = None,
        class_mapping_overrides: Dict[int, int] | None = None,
        label_strategy: Literal[
            "LabelEncoder", "LabelBinarizer", "Hierarchical"
        ] = "LabelEncoder",
        # Filtering parameters
        confidence: List[Literal["high", "medium"]] | None = None,
        # Cloud masking parameters
        valid_scl_values: List[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        | None = None,
        chip_size: Literal[32, 16, 8, 4] = 32,
        min_clear_percentage_chip: int | None = None,
        months: List[Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] | None = None,
        max_days_since_event: int | dict | None = None,
        sample_datasets: List[Literal["Evoland", "HRVPP", "Windthrow"]] | None = None,
        # Sampling parameters
        max_samples_per_event: int | None = None,
        random_seed: int | None = None,
        # Balanced sampling parameters
        apply_downsampling: bool = False,
        target_majority_samples: int | None = None,
        # Quality filters
        omit_low_tcd: bool = True,
        omit_border: bool = True,
        # Feature selection
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
        # Outlier removal parameters
        remove_outliers: bool = False,
        outlier_method: Literal["iqr", "zscore", "modified_zscore"] = "iqr",
        outlier_threshold: float = 1.5,
        outlier_columns: List[
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
    ):
        self.random_seed = random_seed
        self.data_folder = data_folder
        self._load_base_data()
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
        self.bands = bands or all_bands[:-1]
        self.band_idxs = [all_bands.index(band) for band in self.bands]
        self.target_classes = target_classes or list(CLASSES.keys())
        self.valid_scl_values = valid_scl_values or [2, 4, 5, 6]
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.outlier_columns = outlier_columns
        self.class_mapping_overrides = class_mapping_overrides or {}
        self.chip_size = chip_size
        self.label_strategy = label_strategy

        # Filters for samples.parquet
        samples_filters = [pl.lit(True)]
        # TODO: sample_ids should be handled in the implementing classes
        if confidence is not None:
            samples_filters.append(pl.col.confidence.is_in(confidence))
        if sample_datasets is not None:
            samples_filters.append(pl.col.dataset.is_in(sample_datasets))
        if omit_low_tcd:
            samples_filters.append(~pl.col.comment.str.contains("TCD"))
        if omit_border:
            samples_filters.append(~pl.col.comment.str.contains("border"))

        # Filters for labels.parquet
        labels_filters = [pl.lit(True)]
        if target_classes is not None:
            labels_filters.append(pl.col("label").is_in(self.target_classes))
        else:
            # This is done because there are some labels with value 999 (artifact) in the labels dataset
            # TODO: Fix this directly in the dataset? i.e. exclude 999
            labels_filters.append(pl.col("label").is_in(CLASSES.keys()))

        # Filters for pixel_data.parquet
        pixel_data_filters = [pl.lit(True)]
        if months is not None:
            pixel_data_filters.append(pl.col("timestamps").dt.month().is_in(months))
        if self.valid_scl_values is not None:
            pixel_data_filters.append(pl.col.SCL.is_in(self.valid_scl_values))
        if min_clear_percentage_chip is not None:
            pixel_data_filters.append(
                pl.col(f"percent_clear_{chip_size}x{chip_size}")
                >= min_clear_percentage_chip
            )
        match max_days_since_event:
            case dict():
                max_duration_filters = []
                for label, days in max_days_since_event.items():
                    if days is None:
                        continue
                    max_duration_filters.append(
                        (
                            (pl.col("timestamps") - pl.col("start"))
                            > pl.duration(days=days)
                        )
                        & (pl.col.label == label)
                    )
                pixel_data_filters.append(~pl.any_horizontal(max_duration_filters))
            case int():
                pixel_data_filters.append(
                    (
                        (pl.col("timestamps") - pl.col("start"))
                        > pl.duration(days=max_days_since_event)
                    )
                )

        # Load and filter samples data
        samples = pl.read_parquet(
            self.base_data_paths["samples.parquet"],
            columns=["sample_id", "cluster_id", "comment", "dataset", "confidence"],
            use_pyarrow=True,
        ).filter(samples_filters)

        labels = (
            pl.read_parquet(
                self.base_data_paths["labels.parquet"],
                columns=["sample_id", "label", "start"],
            )
            .join(samples, on="sample_id", how="inner")
            .with_columns(
                pl.col.label.replace_strict(
                    self.class_mapping_overrides,
                    return_dtype=pl.UInt16,
                    default=pl.col.label,
                ),
            )
            .filter(labels_filters)
        )

        # Load and filter pixel data
        pixel_data = (
            pl.read_parquet(
                self.base_data_paths["pixel_data.parquet"],
                columns=set(
                    [
                        "sample_id",
                        "SCL",
                        "timestamps",
                        "label",
                        f"percent_clear_{chip_size}x{chip_size}",
                    ]
                    + self.bands
                ),
            )
            .join(labels, on=["sample_id", "label"], how="inner")
            .filter(pixel_data_filters)
        )

        # Outlier removal using statistical measures
        if remove_outliers and len(pixel_data) > 0:
            pixel_data = self._remove_outliers(pixel_data)

        # Apply sampling sub-sampling per event
        if max_samples_per_event is not None and len(pixel_data) > 0:
            if random_seed is not None:
                pl.set_random_seed(random_seed)
            if max_samples_per_event > 0:
                pixel_data = pixel_data.filter(
                    pl.int_range(pl.len()).over(["sample_id", "label"])
                    < max_samples_per_event
                )

        if apply_downsampling and len(pixel_data) > 0:
            pixel_data = self._apply_balanced_sampling(
                pixel_data, target_majority_samples
            )

        match label_strategy:
            case "LabelEncoder":
                from sklearn.preprocessing import LabelEncoder

                self.encoder = LabelEncoder()
            case "LabelBinarizer":
                from sklearn.preprocessing import LabelBinarizer

                self.encoder = LabelBinarizer()
            case "Hierarchical":
                from disfor.utils import HierarchicalLabelEncoder

                self.encoder = HierarchicalLabelEncoder()

        self.encoder.fit(self.target_classes)

        if len(pixel_data) > 0:
            pixel_data = (
                pixel_data.with_columns(
                    label_encoded=pl.Series(self.encoder.transform(pixel_data["label"]))
                )
                .sort("dataset", "cluster_id")
                .with_columns(
                    cluster_id_encoded=pl.struct("dataset", "cluster_id").rank("dense"),
                )
            )

        self.pixel_data = pixel_data

    def _load_base_data(self):
        """Load base data files"""
        required_data = [
            "classes.json",
            "train_ids.json",
            "val_ids.json",
            "labels.parquet",
            "pixel_data.parquet",
            "samples.parquet",
        ]
        if self.data_folder is None:
            self.base_data_paths = {
                filename: DATA_GETTER.fetch(filename) for filename in required_data
            }
        else:
            self.base_data_paths = {
                filename: Path(self.data_folder) / filename
                for filename in required_data
            }

        with open(self.base_data_paths["classes.json"], "r") as f:
            self._class_mapping = {int(k): v for k, v in json.load(f).items()}

        with open(self.base_data_paths["train_ids.json"], "r") as f:
            self._train_ids = json.load(f)

        with open(self.base_data_paths["val_ids.json"], "r") as f:
            self._val_ids = json.load(f)

    def _apply_balanced_sampling(
        self, df: pl.DataFrame, target_majority_samples: int | None = None
    ) -> pl.DataFrame:
        """Apply balanced sampling by downsampling the majority class"""
        counts = df["label"].value_counts(sort=True)[0]

        # return, if there's only one (or no) classes
        if len(counts) < 2:
            return df

        # Find majority class
        max_count = counts["count"][0]
        majority_class = counts["label"][0]

        # Determine target size for majority class,
        # If no target is set, the second largest class*2 is the maximum, if lower than 500
        if target_majority_samples is None:
            second_largest = counts["count"][1]
            target_majority_samples = min(max_count, max(second_largest * 2, 500))

        # Set random seed if specified
        if self.random_seed is not None:
            pl.set_random_seed(self.random_seed)

        # Split and downsample
        majority_mask = pl.col("label") == majority_class
        majority_samples = df.filter(majority_mask).sample(
            n=min(target_majority_samples, max_count)
        )
        minority_samples = df.filter(~majority_mask)

        return pl.concat([minority_samples, majority_samples])

    def _remove_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate outlier mask using the configured method"""
        outlier_cols = (
            self.outlier_columns if self.outlier_columns is not None else self.bands
        )

        mask = [pl.lit(False)]

        if self.outlier_method == "iqr":
            for col in outlier_cols:
                q1 = pl.col(col).quantile(0.25).over("sample_id", "label")
                q3 = pl.col(col).quantile(0.75).over("sample_id", "label")
                iqr = q3 - q1
                lower_bound = q1 - self.outlier_threshold * iqr
                upper_bound = q3 + self.outlier_threshold * iqr
                mask.append((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound))

        elif self.outlier_method == "zscore":
            for col in outlier_cols:
                mean = pl.col(col).mean().over("sample_id", "label")
                std = pl.col(col).std().over("sample_id", "label")
                z_score = ((pl.col(col) - mean) / std).abs()
                mask.append(z_score > self.outlier_threshold)

        elif self.outlier_method == "modified_zscore":
            for col in outlier_cols:
                median_val = pl.col(col).median().over("sample_id", "label")
                mad = (
                    ((pl.col(col) - pl.col(col).median()).abs())
                    .median()
                    .over("sample_id", "label")
                )
                modified_z = 0.6745 * (pl.col(col) - median_val).abs() / mad
                mask.append(modified_z > self.outlier_threshold)

        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")

        return df.filter(~pl.any_horizontal(mask))


class ForestDisturbanceData(GenericDataset):
    """Class providing data for sklearn style models

    For usage see the [dataloaders usage page](../usage/dataloaders).

    Args:
        **kwargs: keyword arguments being passed to [disfor.data.GenericDataset][]
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
