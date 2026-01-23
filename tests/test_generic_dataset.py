from hypothesis import given, strategies as st, settings, HealthCheck
from disfor.data import GenericDataset

# Define valid values as strategies
TARGET_CLASSES = [
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

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
SAMPLE_DATASETS = [1, 2, 3]
CONFIDENCE_LEVELS = ["high", "medium"]
CHIP_SIZES = [4, 8, 16, 32]
MONTHS = list(range(1, 13))


@given(
    target_classes=st.one_of(
        st.none(),
        st.lists(st.sampled_from(TARGET_CLASSES), min_size=1, max_size=5, unique=True),
    ),
    chip_size=st.sampled_from(CHIP_SIZES),
    confidence=st.one_of(
        st.none(),
        st.lists(
            st.sampled_from(CONFIDENCE_LEVELS), min_size=1, max_size=2, unique=True
        ),
    ),
    valid_scl_values=st.one_of(
        st.none(),
        st.lists(
            st.sampled_from(list(range(0, 12))), min_size=1, max_size=12, unique=True
        ),
    ),
    max_samples_per_event=st.one_of(
        st.none(),
        st.integers(0, max_value=1000000),
    ),
    apply_downsampling=st.booleans(),
    remove_outliers=st.booleans(),
    outlier_method=st.sampled_from(["iqr", "zscore", "modified_zscore"]),
    outlier_threshold=st.floats(0),
    target_majority_samples=st.one_of(
        st.none(),
        st.integers(0),
    ),
    sample_datasets=st.one_of(
        st.none(),
        st.lists(st.sampled_from(SAMPLE_DATASETS), min_size=1, max_size=3, unique=True),
    ),
    min_clear_percentage_chip=st.integers(min_value=0, max_value=100),
    max_days_since_event=st.one_of(st.none(), st.integers(min_value=1, max_value=365)),
    bands=st.one_of(
        st.none(),
        st.lists(st.sampled_from(BANDS), min_size=1, max_size=11, unique=True),
    ),
    months=st.one_of(
        st.none(),
        st.lists(st.sampled_from(MONTHS), min_size=1, max_size=12, unique=True),
    ),
    omit_border=st.booleans(),
    omit_low_tcd=st.booleans(),
    label_strategy=st.sampled_from(["LabelEncoder", "LabelBinarizer", "Hierarchical"]),
)
@settings(
    max_examples=50,  # Adjust based on your test runtime needs
    deadline=None,  # Disable deadline for potentially slow I/O operations
    suppress_health_check=[HealthCheck.too_slow],
)
def test_tiff_dataset_initialization(
    target_classes,
    chip_size,
    confidence,
    valid_scl_values,
    max_samples_per_event,
    apply_downsampling,
    remove_outliers,
    outlier_method,
    outlier_threshold,
    target_majority_samples,
    sample_datasets,
    min_clear_percentage_chip,
    max_days_since_event,
    bands,
    months,
    omit_border,
    omit_low_tcd,
    label_strategy,
):
    """
    Integration test that verifies TiffDataset can be initialized with
    various valid parameter combinations and perform basic operations.
    """
    # Initialize dataset with generated parameters
    _ = GenericDataset(
        target_classes=target_classes,
        chip_size=chip_size,
        confidence=confidence,
        sample_datasets=sample_datasets,
        min_clear_percentage_chip=min_clear_percentage_chip,
        max_days_since_event=max_days_since_event,
        bands=bands,
        months=months,
        omit_border=omit_border,
        omit_low_tcd=omit_low_tcd,
        valid_scl_values=valid_scl_values,
        max_samples_per_event=max_samples_per_event,
        apply_downsampling=apply_downsampling,
        remove_outliers=remove_outliers,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        target_majority_samples=target_majority_samples,
        label_strategy=label_strategy,
    )
