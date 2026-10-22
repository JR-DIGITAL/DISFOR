"""Share of sampled windthrow polygons per event that weren't interpreted as windthrow.

A polygon counts as windthrow if its raw flags contain a wind ("7") or a
salvage ("a") flag (see hrvpp_label_mapping in 1_homogenize_samples.ipynb).
Flags with confidence "low" are included, since these are the raw flags.

Usage (from the repository root):
    python scripts/windthrow_flag_stats.py
"""

import json
from pathlib import Path

import geopandas as gpd
import polars as pl

FLAG_DIR = Path(__file__).parent.parent / "flags"
WINDTHROW_FLAGS = {"7", "a"}


def read_flags(flag_dir):
    rows = []
    for flag_path in flag_dir.glob("flags_*.json"):
        with flag_path.open("r") as f:
            sample = json.load(f)
        rows.append(
            {
                "id": int(flag_path.stem.split("_")[-1]),
                "is_windthrow": bool(WINDTHROW_FLAGS & set(sample["flags"].values())),
                "low_confidence": (sample["confidence"] or "").lower() == "low",
            }
        )
    return pl.DataFrame(rows)


if __name__ == "__main__":
    polygons = pl.from_pandas(
        gpd.read_file(FLAG_DIR / "windthrow_flags.geojson")[
            ["id", "EventId", "StormName"]
        ]
    ).with_columns(pl.col.id.cast(pl.Int64))
    flags = read_flags(FLAG_DIR / "windthrow_flags")

    stats = (
        polygons.join(flags, on="id", how="left")
        .group_by("EventId")
        .agg(
            # Some polygons of an event have the storm name "unknown"
            pl.col.StormName.filter(pl.col.StormName != "unknown")
            .first()
            .fill_null("unknown"),
            n_polygons=pl.len(),
            n_interpreted=pl.col.is_windthrow.is_not_null().sum(),
            n_not_windthrow=(~pl.col.is_windthrow).sum(),
            n_not_windthrow_low_conf=(
                ~pl.col.is_windthrow & pl.col.low_confidence
            ).sum(),
        )
        .with_columns(
            pct_not_windthrow=(
                100 * pl.col.n_not_windthrow / pl.col.n_interpreted
            ).round(1)
        )
        .sort("EventId")
    )

    total = stats.select(
        pl.lit("Total").alias("EventId"),
        pl.lit(None, pl.String).alias("StormName"),
        pl.sum(
            "n_polygons", "n_interpreted", "n_not_windthrow", "n_not_windthrow_low_conf"
        ),
    ).with_columns(
        pct_not_windthrow=(100 * pl.col.n_not_windthrow / pl.col.n_interpreted).round(1)
    )

    with pl.Config(tbl_rows=-1, tbl_width_chars=200):
        print(pl.concat([stats, total], how="vertical_relaxed"))
