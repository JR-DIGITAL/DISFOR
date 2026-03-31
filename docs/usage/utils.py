import json
import urllib
import warnings

import disfor
import geopandas as gpd
import polars as pl


def prepare_browser_urls():
    samples = gpd.read_parquet(disfor.get("samples.parquet"))
    samples.fillna("", inplace=True)
    labels = pl.read_parquet(disfor.get("labels.parquet")).with_columns(
        is_event=pl.col.start.dt.day() == pl.col.end.dt.day()
    )
    with disfor.get("classes.json").open() as f:
        classes_mapping = json.load(f)

    campaign_setup = {
        "type": "FeatureCollection",
        "campaign": {
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
        },
        "features": [],
    }

    events = labels.filter("is_event").select(
        "sample_id", "label", pl.col.start.alias("flag_date")
    )
    segments = (
        labels.filter(~pl.col.is_event)
        .unpivot(["start", "end"], index=["sample_id", "label"], value_name="flag_date")
        .drop("variable")
    )
    all_labels = pl.concat([events, segments])

    flags_tuple = (
        all_labels.filter(sample_id=0)
        .sort("flag_date")
        .select(pl.col.flag_date.dt.strftime("%Y-%m-%d"), "label")
        .rows_by_key("flag_date", unique=True)
    )
    flags = {k: str(v[0]) for k, v in flags_tuple.items()}

    features = []
    for row, f in samples.iterrows():
        sample_id = f["sample_id"]
        flags_tuple = (
            all_labels.filter(sample_id=sample_id)
            .sort("flag_date")
            .select(pl.col.flag_date.dt.strftime("%Y-%m-%d"), "label")
            .rows_by_key("flag_date", unique=True)
        )
        flags = {k: str(v[0]) for k, v in flags_tuple.items()}
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [f["geometry"].x, f["geometry"].y],
            },
            "properties": {
                "sample_id": sample_id,
                "flags": flags,
                "confidence": f["confidence"],
                "cluster_id": f["cluster_id"],
                "source": f["source"],
                "comment": f["comment"],
                "interpreter": f["interpreter"],
            },
        }
        features.append(feature)

    campaign_setup["features"] = features

    with open("disfor_campaign.geojson", "w") as fp:
        json.dump(campaign_setup, fp, indent=4)

    return campaign_setup


def urls_from_campaign_geojson(campaign_dict, base_url):
    query_params = {}
    campaign_schema = campaign_dict["campaign"]
    query_params["start"] = campaign_schema.pop("startDate")
    query_params["end"] = campaign_schema.pop("endDate")
    campaign_schema["campaign"] = campaign_schema.pop("name")
    query_params["schema"] = json.dumps(campaign_schema, separators=(",", ":"))

    urls = []
    for feature in campaign_dict["features"]:
        query_params["lon"] = feature["geometry"]["coordinates"][0]
        query_params["lat"] = feature["geometry"]["coordinates"][1]
        query_params["sample"] = json.dumps(
            feature["properties"], separators=(",", ":")
        )
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
