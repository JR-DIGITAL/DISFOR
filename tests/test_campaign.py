import datetime
import json

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from utils import (
    EVENT_LABELS,
    _salvage_issues,
    campaign_header,
    campaign_to_tables,
    flags_to_labels,
    labels_to_flags,
    tables_to_campaign,
    validate_campaign,
)

import disfor

SEGMENT_LABELS = [100, 110, 120, 121, 122, 123, 230, 231, 232, 241]

with disfor.get("classes.json").open() as f:
    CLASSES = json.load(f)

# JSON and arrow can't represent lone surrogates
text = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20)
non_blank_text = text.filter(str.strip)


@st.composite
def flags(draw):
    """Valid flags: events and segments in order, events may lie within segments"""
    items = draw(
        st.lists(
            st.one_of(
                st.tuples(st.sampled_from(sorted(EVENT_LABELS)), st.just([])),
                st.tuples(
                    st.sampled_from(SEGMENT_LABELS),
                    st.lists(st.sampled_from(sorted(EVENT_LABELS)), max_size=2),
                ),
            ),
            min_size=1,
            max_size=6,
        )
    )
    labels = []
    for label, inner_events in items:
        if label in EVENT_LABELS:
            labels.append(label)
        else:
            labels.extend([label, *inner_events, label])

    dates = draw(
        st.lists(
            st.dates(datetime.date(2015, 1, 1), datetime.date(2025, 1, 1)),
            min_size=len(labels),
            max_size=len(labels),
            unique=True,
        )
    )
    return {
        date.strftime("%Y-%m-%d"): str(label)
        for date, label in zip(sorted(dates), labels)
    }


@st.composite
def campaigns(draw):
    sample_ids = draw(st.lists(st.integers(0, 2**16 - 1), max_size=5, unique=True))
    coordinates = draw(
        st.lists(
            st.tuples(st.floats(-180, 180), st.floats(-90, 90)),
            min_size=len(sample_ids),
            max_size=len(sample_ids),
            unique=True,
        )
    )
    features = []
    for sample_id, (x, y) in zip(sorted(sample_ids), coordinates):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [x, y]},
                "properties": {
                    "sample_id": sample_id,
                    "flags": draw(flags().filter(lambda f: not _salvage_issues(f))),
                    "original_sample_id": draw(st.integers(-(2**63), 2**63 - 1)),
                    "interpreter": draw(non_blank_text),
                    "dataset": draw(st.integers(0, 2**8 - 1)),
                    "source": draw(text),
                    "source_description": draw(text),
                    "s2_tile": draw(text),
                    "cluster_id": draw(text),
                    "cluster_description": draw(text),
                    "comment": draw(text),
                    "confidence": draw(st.sampled_from(["high", "medium", "low"])),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "campaign": campaign_header(CLASSES),
        "features": features,
    }


@given(campaign=campaigns())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_campaign_round_trip(campaign):
    assert tables_to_campaign(*campaign_to_tables(campaign), CLASSES) == campaign


@given(campaign=campaigns())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_campaign_is_json_serializable(campaign):
    assert json.loads(json.dumps(campaign)) == campaign


def read_table(name):
    table = pq.read_table(disfor.get(name))
    table = table.drop_columns(
        [c for c in table.column_names if c.startswith("__index_level")]
    )
    return table.to_pandas(types_mapper=pd.ArrowDtype)


@pytest.fixture(scope="module")
def dataset_tables():
    samples = read_table("samples.parquet")
    samples = gpd.GeoDataFrame(
        samples.drop(columns="geometry"),
        geometry=gpd.GeoSeries.from_wkb(samples["geometry"], crs="EPSG:4326"),
    )
    samples = samples.sort_values("sample_id").reset_index(drop=True)
    labels = read_table("labels.parquet")
    labels = labels.sort_values(["sample_id", "start"]).reset_index(drop=True)
    return samples, labels


def normalize(samples, labels):
    """Apply the simplifications of the campaign format: flags are dates and
    comments are never null"""
    samples = samples.copy()
    samples["comment"] = samples["comment"].fillna("")

    def floor(column):
        dtype = column.dtype
        return column.astype("datetime64[ms, UTC]").dt.floor("D").astype(dtype)

    labels = labels.copy()
    is_segment = ~labels["label"].isin(EVENT_LABELS)
    labels["start"] = floor(labels["start"])
    labels.loc[is_segment, "end"] = floor(labels.loc[is_segment, "end"])
    labels["start_next_label"] = floor(labels["start_next_label"])
    return samples, labels


def test_dataset_round_trip(dataset_tables):
    expected_samples, expected_labels = normalize(*dataset_tables)
    campaign = tables_to_campaign(*dataset_tables, CLASSES)
    samples, labels = campaign_to_tables(campaign)

    pd.testing.assert_frame_equal(samples, expected_samples)
    pd.testing.assert_frame_equal(labels, expected_labels)


def test_dataset_round_trip_is_idempotent(dataset_tables):
    once = campaign_to_tables(tables_to_campaign(*dataset_tables, CLASSES))
    twice = campaign_to_tables(tables_to_campaign(*once, CLASSES))

    pd.testing.assert_frame_equal(once[0], twice[0])
    pd.testing.assert_frame_equal(once[1], twice[1])


def test_event_within_segment():
    flags = {"2020-01-01": "110", "2021-05-03": "213", "2024-12-31": "110"}
    assert [row["label"] for row in flags_to_labels(flags)] == [110, 213]


@pytest.mark.parametrize(
    "flags",
    [
        # segments overlap
        {"2020-01-01": "110", "2021-01-01": "120", "2022-01-01": "110"},
        # segment without end
        {"2020-01-01": "110", "2021-01-01": "211"},
    ],
)
def test_invalid_flags(flags):
    with pytest.raises(ValueError):
        flags_to_labels(flags)


def test_flag_date_collision():
    labels = pd.DataFrame(
        {
            "sample_id": [0, 0],
            "label": [110, 211],
            "start": pd.to_datetime(["2020-01-01", "2021-01-01"], utc=True),
            "end": pd.to_datetime(
                ["2021-01-01", "2021-01-01 23:59:59"], utc=True, format="ISO8601"
            ),
        }
    )
    with pytest.raises(ValueError):
        labels_to_flags(labels)


def test_validate_campaign_reports_all_issues():
    def feature(sample_id, coordinates, flags, interpreter="me", confidence="high"):
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": coordinates},
            "properties": {
                "sample_id": sample_id,
                "flags": flags,
                "interpreter": interpreter,
                "confidence": confidence,
            },
        }

    campaign = {
        "features": [
            feature(0, [1.0, 2.0], {"2020-01-01": "110"}),
            feature(1, [1.0, 2.0], {"2020-01-01": "211"}, interpreter=""),
            feature(2, [3.0, 4.0], {"2020-01-01": "110", "2021-01-01": "120"}),
            feature(3, [5.0, 6.0], {"2020-01-01": "211"}, confidence=""),
            feature(4, [7.0, 8.0], {"2020-01-01": "243", "2020-02-01": "221"}),
            feature(
                5,
                [9.0, 10.0],
                {"2020-01-01": "231", "2020-06-01": "231", "2020-07-01": "222"},
            ),
            # matching salvage is fine
            feature(6, [11.0, 12.0], {"2020-01-01": "243", "2020-02-01": "222"}),
        ]
    }
    issues = validate_campaign(campaign)
    assert len(issues) == 8
    assert sum("no end flag" in issue for issue in issues) == 3
    assert "Sample 1: missing interpreter" in issues
    assert "Sample 3: missing confidence" in issues
    assert any("Samples 0, 1: duplicated point" in issue for issue in issues)
    assert any(issue.startswith("Sample 4: biotic salvage") for issue in issues)
    assert any(issue.startswith("Sample 5: abiotic salvage") for issue in issues)
