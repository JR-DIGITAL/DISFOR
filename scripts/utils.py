import json
import urllib
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
import pyarrow as pa
from shapely.geometry import Point

import disfor

# Labels which are single day events. All other labels are segments with a
# start and end flag.
EVENT_LABELS = {200, 210, 211, 212, 213, 220, 221, 222, 240, 242, 243, 244, 245}

BIOTIC_LABELS = {230, 231, 232}
ABIOTIC_LABELS = {240, 241, 242, 243, 244, 245}
# Salvage labels and the disturbance labels which must not directly precede them
SALVAGE_MISMATCHES = {
    221: ("biotic", "abiotic", ABIOTIC_LABELS),
    222: ("abiotic", "biotic", BIOTIC_LABELS),
}

# Inverse of the label mapping in 1_homogenize_samples.ipynb, including the
# relabelling of salvage (221, 222) and canopy closing (122)
ORIGINAL_LABELS = {
    110: "0",
    231: "1",
    230: "2",
    121: "3",
    122: "3",
    120: "4",
    123: "b",
    211: "5",
    212: "6",
    243: "7",
    242: "8",
    240: "9",
    220: "a",
    221: "a",
    222: "a",
    213: "c",
    200: "e",
    245: "f",
    232: "g",
    241: "h",
}

# Sample columns which are null if empty. They are stored as "" in the campaign.
NULLABLE_SAMPLE_COLUMNS = ["s2_tile", "cluster_id"]

SAMPLES_SCHEMA = {
    "sample_id": pa.uint16(),
    "original_sample_id": pa.int64(),
    "interpreter": pa.large_string(),
    "dataset": pa.uint8(),
    "source": pa.large_string(),
    "source_description": pa.large_string(),
    "s2_tile": pa.large_string(),
    "cluster_id": pa.large_string(),
    "cluster_description": pa.large_string(),
    "comment": pa.large_string(),
    "confidence": pa.large_string(),
}

# Order of sample properties in the campaign. The first ones are passed to the
# tsbrowser and keep the order of the original campaign, so urls don't change.
PROPERTY_ORDER = [
    "confidence",
    "cluster_id",
    "source",
    "comment",
    "interpreter",
    "original_sample_id",
    "dataset",
    "source_description",
    "s2_tile",
    "cluster_description",
]

LABELS_SCHEMA = {
    "original_sample_id": pa.int64(),
    "dataset": pa.uint8(),
    "label": pa.uint16(),
    "original_label": pa.large_string(),
    "start": pa.timestamp("ms", tz="UTC"),
    "end": pa.timestamp("ms", tz="UTC"),
    "sample_id": pa.uint16(),
    "start_next_label": pa.timestamp("ms", tz="UTC"),
}

FLAG_DATE_FORMAT = "%Y-%m-%d"
EVENT_DURATION = pd.Timedelta(hours=23, minutes=59, seconds=59)


def campaign_header(classes_mapping: dict[str, str]) -> dict:
    return {
        "name": "DISFOR",
        "startDate": "2015-01-01",
        "endDate": "2025-01-01",
        "flagLabels": classes_mapping,
        "fields": [
            {
                "key": "sample_id",
                "label": "Sample ID",
                "type": "display",
                "required": True,
                "session_persistent": False,
            },
            {
                "key": "source",
                "label": "Source",
                "type": "display",
                "required": True,
                "session_persistent": False,
            },
            {
                "key": "cluster_id",
                "label": "Cluster ID",
                "type": "display",
                "required": True,
                "session_persistent": False,
            },
            {
                "key": "confidence",
                "label": "Confidence",
                "type": "select",
                "options": ["high", "medium", "low"],
                "required": True,
                "session_persistent": False,
            },
            {
                "key": "comment",
                "label": "Comment",
                "type": "text",
                "required": False,
                "session_persistent": False,
            },
            {
                "key": "interpreter",
                "label": "Interpreter",
                "type": "text",
                "required": True,
                "session_persistent": True,
            },
        ],
    }


def labels_to_flags(labels: pd.DataFrame) -> dict[str, str]:
    """Encode the labels of a single sample as tsbrowser flags.

    Events are a single flag at their start date, segments a flag at their start
    and end date.
    """
    flags = {}
    for row in labels.itertuples():
        dates = [row.start] if row.label in EVENT_LABELS else [row.start, row.end]
        for date in dates:
            key = date.strftime(FLAG_DATE_FORMAT)
            if key in flags:
                raise ValueError(
                    f"Sample {row.sample_id} has more than one flag on {key}"
                )
            flags[key] = str(row.label)
    return dict(sorted(flags.items()))


def _decode_flags(flags: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Decode tsbrowser flags of a single sample to label rows and issues.

    Segment flags open and close a segment of the same label, events may lie
    within a segment. Segments which are not closed are reported as issues and
    left out of the label rows.
    """
    rows, issues = [], []
    open_segment = None
    for key, value in sorted(flags.items()):
        date = pd.Timestamp(key, tz="UTC")
        label = int(value)
        if label in EVENT_LABELS:
            rows.append({"label": label, "start": date, "end": date + EVENT_DURATION})
        elif open_segment is None:
            open_segment = {"label": label, "start": date}
        elif open_segment["label"] == label:
            rows.append({**open_segment, "end": date})
            open_segment = None
        else:
            issues.append(
                f"segment {open_segment['label']} starting "
                f"{open_segment['start']:{FLAG_DATE_FORMAT}} has no end flag "
                f"before segment {label} starts on {key}"
            )
            open_segment = {"label": label, "start": date}
    if open_segment is not None:
        issues.append(
            f"segment {open_segment['label']} starting "
            f"{open_segment['start']:{FLAG_DATE_FORMAT}} has no end flag"
        )
    return sorted(rows, key=lambda row: row["start"]), issues


def _salvage_issues(flags: dict[str, str]) -> list[str]:
    """Report salvage flags directly following a disturbance of the other kind,
    for example a biotic salvage directly after an abiotic disturbance."""
    issues = []
    previous = None
    for key, value in sorted(flags.items()):
        label = int(value)
        if label in SALVAGE_MISMATCHES and previous is not None:
            salvage_kind, other_kind, other_labels = SALVAGE_MISMATCHES[label]
            if previous[1] in other_labels:
                issues.append(
                    f"{salvage_kind} salvage {label} on {key} directly after "
                    f"{other_kind} flag {previous[1]} on {previous[0]}"
                )
        previous = (key, label)
    return issues


def flags_to_labels(flags: dict[str, str]) -> list[dict]:
    """Decode tsbrowser flags of a single sample to label rows.

    Raises a ValueError listing all issues if a segment has no end flag.
    """
    rows, issues = _decode_flags(flags)
    if issues:
        raise ValueError("; ".join(issues))
    return rows


def validate_campaign(campaign: dict) -> list[str]:
    """Check a tsbrowser campaign before converting it with `campaign_to_tables`.

    Returns a list of all issues found: segments without an end flag, missing
    interpreter or confidence, salvage flags directly following a disturbance
    of the other kind (biotic/abiotic) and samples sharing the exact same
    coordinates.
    An empty list means the campaign is valid.
    """
    issues = []
    samples_by_coordinates = {}
    for feature in campaign["features"]:
        properties = feature["properties"]
        sample_id = properties.get("sample_id")
        flags = properties.get("flags", {})
        for issue in _decode_flags(flags)[1] + _salvage_issues(flags):
            issues.append(f"Sample {sample_id}: {issue}")
        for column in ["interpreter", "confidence"]:
            value = properties.get(column)
            if value is None or not str(value).strip():
                issues.append(f"Sample {sample_id}: missing {column}")
        coordinates = tuple(feature["geometry"]["coordinates"])
        samples_by_coordinates.setdefault(coordinates, []).append(sample_id)

    for coordinates, sample_ids in samples_by_coordinates.items():
        if len(sample_ids) > 1:
            issues.append(
                f"Samples {', '.join(map(str, sample_ids))}: duplicated point "
                f"at {coordinates}"
            )
    return issues


def tables_to_campaign(
    samples: gpd.GeoDataFrame,
    labels: pd.DataFrame,
    classes_mapping: dict[str, str],
) -> dict:
    """Convert samples and labels tables to a tsbrowser campaign.

    Inverse of `campaign_to_tables`. Label timestamps are reduced to dates and
    null comments become empty strings.
    """
    labels_by_sample = dict(list(labels.sort_values("start").groupby("sample_id")))

    features = []
    for sample in samples.sort_values("sample_id").itertuples():
        properties = {
            "sample_id": int(sample.sample_id),
            "flags": labels_to_flags(labels_by_sample[sample.sample_id]),
        }
        for column in PROPERTY_ORDER:
            value = getattr(sample, column)
            if pa.types.is_integer(SAMPLES_SCHEMA[column]):
                properties[column] = int(value)
            else:
                properties[column] = "" if pd.isna(value) else str(value)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [sample.geometry.x, sample.geometry.y],
                },
                "properties": properties,
            }
        )

    return {
        "type": "FeatureCollection",
        "campaign": campaign_header(classes_mapping),
        "features": features,
    }


def campaign_to_tables(campaign: dict) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Convert a tsbrowser campaign to samples and labels tables.

    Inverse of `tables_to_campaign`. The returned tables have the schema of
    samples.parquet and labels.parquet. Raises a ValueError listing all issues
    found by `validate_campaign`.
    """
    issues = validate_campaign(campaign)
    if issues:
        raise ValueError(
            f"Campaign has {len(issues)} issues:\n" + "\n".join(issues)
        )

    sample_rows, label_rows, geometries = [], [], []
    for feature in campaign["features"]:
        properties = feature["properties"]
        sample_rows.append({column: properties[column] for column in SAMPLES_SCHEMA})
        geometries.append(Point(feature["geometry"]["coordinates"]))
        for row in flags_to_labels(properties["flags"]):
            label_rows.append(
                {
                    "original_sample_id": properties["original_sample_id"],
                    "dataset": properties["dataset"],
                    "label": row["label"],
                    "original_label": ORIGINAL_LABELS.get(row["label"]),
                    "start": row["start"],
                    "end": row["end"],
                    "sample_id": properties["sample_id"],
                }
            )

    samples = pd.DataFrame(sample_rows, columns=list(SAMPLES_SCHEMA))
    samples[NULLABLE_SAMPLE_COLUMNS] = samples[NULLABLE_SAMPLE_COLUMNS].replace(
        "", None
    )
    samples = gpd.GeoDataFrame(
        _to_arrow_dtypes(samples, SAMPLES_SCHEMA),
        geometry=gpd.GeoSeries(geometries, crs="EPSG:4326"),
    )
    samples = samples.sort_values("sample_id").reset_index(drop=True)

    labels = pd.DataFrame(label_rows, columns=list(LABELS_SCHEMA))
    labels = labels.sort_values(["sample_id", "start"], kind="stable")
    labels["start_next_label"] = labels.groupby("sample_id")["start"].shift(-1)
    labels = _to_arrow_dtypes(labels.reset_index(drop=True), LABELS_SCHEMA)

    return samples, labels


def _to_arrow_dtypes(df: pd.DataFrame, schema: dict[str, pa.DataType]):
    return pd.DataFrame(
        {
            column: pd.Series(
                pa.array(df[column].tolist(), type=dtype, from_pandas=True),
                dtype=pd.ArrowDtype(dtype),
            )
            for column, dtype in schema.items()
        }
    )


def campaign_geojson_to_parquet(geojson_path: Path | str, out_dir: Path | str):
    """Write samples.parquet and labels.parquet from a campaign geojson."""
    with open(geojson_path) as f:
        samples, labels = campaign_to_tables(json.load(f))
    samples.to_parquet(Path(out_dir) / "samples.parquet", index=False)
    labels.to_parquet(Path(out_dir) / "labels.parquet", index=False)


def prepare_browser_urls():
    samples = gpd.read_parquet(disfor.get("samples.parquet"))
    labels = pd.read_parquet(disfor.get("labels.parquet"))
    with disfor.get("classes.json").open() as f:
        classes_mapping = json.load(f)

    campaign_setup = tables_to_campaign(samples, labels, classes_mapping)

    with open("disfor_campaign.geojson", "w") as fp:
        json.dump(campaign_setup, fp, indent=4)

    return campaign_setup


def urls_from_campaign_geojson(campaign_dict, base_url):
    query_params = {}
    campaign_schema = dict(campaign_dict["campaign"])
    query_params["start"] = campaign_schema.pop("startDate")
    query_params["end"] = campaign_schema.pop("endDate")
    campaign_schema["campaign"] = campaign_schema.pop("name")
    query_params["schema"] = json.dumps(campaign_schema, separators=(",", ":"))

    # Only pass the properties the tsbrowser uses, to keep the urls short
    url_properties = ["sample_id", "flags"] + [
        field["key"] for field in campaign_schema["fields"]
    ]

    urls = []
    for feature in campaign_dict["features"]:
        query_params["lon"] = feature["geometry"]["coordinates"][0]
        query_params["lat"] = feature["geometry"]["coordinates"][1]
        properties = {
            key: value
            for key, value in feature["properties"].items()
            if key in url_properties
        }
        query_params["sample"] = json.dumps(properties, separators=(",", ":"))
        urls.append(
            {
                "sample_id": feature["properties"]["sample_id"],
                "url": base_url + urllib.parse.urlencode(query_params),
            }
        )

    return urls


def polars_url_table(base_url):
    campaign = prepare_browser_urls()
    urls = urls_from_campaign_geojson(campaign, base_url)
    # ignoring warnings due to unknown geopolars extension
    with warnings.catch_warnings(action="ignore"):
        samples = pl.read_parquet(disfor.get("samples.parquet"))[
            [
                "sample_id",
                "dataset",
                "comment",
                "confidence",
            ]
        ]

    html_urls = pl.DataFrame(urls).with_columns(
        pl.format('<a href="{}" target="_blank">Explore!</a>', pl.col.url)
    )
    return samples.join(pl.DataFrame(html_urls), on="sample_id")
