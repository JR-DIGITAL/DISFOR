import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from disfor.dataloader import TiffDataset


def test_dataset_init():
    test = TiffDataset(
        data_folder=r"C:\Users\Jonas.Viehweger\Documents\Projects\2025\disturbance-agent-data\data",
        sample_ids=None,
        target_classes=[110, 211],
        chip_size=32,
        confidence=["high"],
        sample_datasets=["HRVPP", "Windthrow"],
        max_days_since_event=None,
        bands=None,
        months=None,
        omit_border=True,
        omit_low_tcd=True,
    )
    assert len(test) > 0
    test.__getitem__(1)


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
SAMPLE_DATASETS = ["Evoland", "HRVPP", "Windthrow"]
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
    sample_datasets=st.one_of(
        st.none(),
        st.lists(st.sampled_from(SAMPLE_DATASETS), min_size=1, max_size=3, unique=True),
    ),
    min_clear_percentage=st.integers(min_value=0, max_value=100),
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
)
@settings(
    max_examples=50,  # Adjust based on your test runtime needs
    deadline=None,  # Disable deadline for potentially slow I/O operations
    suppress_health_check=[HealthCheck.too_slow],
)
def test_tiff_dataset_initialization_and_basic_operations(
    target_classes,
    chip_size,
    confidence,
    sample_datasets,
    min_clear_percentage,
    max_days_since_event,
    bands,
    months,
    omit_border,
    omit_low_tcd,
):
    """
    Integration test that verifies TiffDataset can be initialized with
    various valid parameter combinations and perform basic operations.
    """
    # Initialize dataset with generated parameters
    dataset = TiffDataset(
        data_folder=r"C:\Users\Jonas.Viehweger\Documents\Projects\2025\disturbance-agent-data\data",
        sample_ids=None,
        target_classes=target_classes,
        chip_size=chip_size,
        confidence=confidence,
        sample_datasets=sample_datasets,
        min_clear_percentage=min_clear_percentage,
        max_days_since_event=max_days_since_event,
        bands=bands,
        months=months,
        omit_border=omit_border,
        omit_low_tcd=omit_low_tcd,
    )

    # Basic sanity checks
    dataset_length = len(dataset)
    assert dataset_length >= 0, "Dataset length should be non-negative"

    # If dataset has items, test __getitem__
    if dataset_length > 0:
        # Test first item
        first_item = dataset[0]
        assert first_item is not None, "First item should not be None"

        # Test last item
        last_item = dataset[dataset_length - 1]
        assert last_item is not None, "Last item should not be None"

        # Test that out-of-bounds raises appropriate error
        with pytest.raises((IndexError, KeyError)):
            dataset[dataset_length]


@given(
    target_classes=st.lists(
        st.sampled_from(TARGET_CLASSES), min_size=1, max_size=3, unique=True
    ),
    max_days_values=st.lists(
        st.integers(min_value=10, max_value=365), min_size=1, max_size=3
    ),
)
@settings(max_examples=20, deadline=None)
def test_tiff_dataset_with_max_days_dict(target_classes, max_days_values):
    """
    Test that max_days_since_event can be specified as a dictionary
    mapping target classes to maximum days.
    """
    # Create a dictionary with some of the target classes
    # Pair each class with a max_days value
    num_classes_for_dict = min(len(target_classes), len(max_days_values))
    max_days_dict = {
        target_classes[i]: max_days_values[i] for i in range(num_classes_for_dict)
    }

    dataset = TiffDataset(
        data_folder="data",
        target_classes=target_classes,
        max_days_since_event=max_days_dict,
        chip_size=32,
    )

    assert len(dataset) >= 0


@given(
    sample_ids=st.lists(
        st.integers(min_value=0, max_value=4000), min_size=1, max_size=10, unique=True
    )
)
@settings(max_examples=10, deadline=None)
def test_tiff_dataset_with_sample_ids(sample_ids):
    """
    Test that dataset can be filtered by specific sample_ids.
    """
    dataset = TiffDataset(
        data_folder=r"C:\Users\Jonas.Viehweger\Documents\Projects\2025\disturbance-agent-data\data",
        sample_ids=sample_ids,
        chip_size=32,
    )

    # The dataset may be empty if none of the generated IDs exist
    # but should still initialize successfully
    assert len(dataset) >= 0


# A simpler test with common configurations for quick regression testing
@pytest.mark.parametrize("chip_size", CHIP_SIZES)
def test_tiff_dataset_chip_sizes(chip_size):
    """Simple parametrized test for different chip sizes."""
    dataset = TiffDataset(
        data_folder=r"C:\Users\Jonas.Viehweger\Documents\Projects\2025\disturbance-agent-data\data",
        chip_size=chip_size,
        target_classes=[110, 211],
        confidence=["high"],
    )

    assert len(dataset) >= 0

    if len(dataset) > 0:
        item = dataset[0]
        assert item["image"].shape == (10, chip_size, chip_size)
        # Assuming the item returns a tuple/dict with image data
        # Add specific assertions based on your actual data structure
        assert item is not None
