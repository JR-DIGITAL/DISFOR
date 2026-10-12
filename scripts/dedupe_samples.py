"""Remove duplicate sample points (identical geometry) from the dataset.

Duplicates stem from the evoland EFFIS polygons, where clusters 71/72 and 73/74
were sampled twice and interpreted independently. Per duplicate pair:

- Same label class sequence: keep the copy with the earliest wildfire date, then
  earliest revegetation start, then lowest sample_id. The pixel time-series of
  both copies are identical; the later dates just skip cloud flagged
  observations in which the event is already visible.
- Different label class sequence: drop both copies, they are written to
  duplicates_to_review.csv for reinterpretation.

Clusters which share a duplicate point cover the same fire polygon, so they are
merged into the cluster of the lower sample_id (72 -> 71, 74 -> 73). This way the
fire stays in one group for cluster based splits, regardless of which copy is kept.

Dropped sample_ids are also removed from id_mapping.csv, so that
1_homogenize_samples.ipynb excludes them when rebuilding the dataset.
"""

import json

import geopandas as gpd
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATA = "./data"


def label_sequence(labels: pd.DataFrame, sample_id: int) -> pd.DataFrame:
    return labels[labels.sample_id == sample_id].sort_values("start")


def format_sequence(seq: pd.DataFrame) -> str:
    return " | ".join(f"{r.label}@{r.start.date()}" for r in seq.itertuples())


def first_start(seq: pd.DataFrame, label: int) -> pd.Timestamp:
    starts = seq.loc[seq.label == label, "start"]
    return starts.iloc[0] if len(starts) else pd.Timestamp.max.tz_localize("UTC")


def resolve_pairs(samples: gpd.GeoDataFrame, labels: pd.DataFrame):
    wkt = samples.geometry.to_wkt()
    duplicated = samples[wkt.duplicated(keep=False)].assign(wkt=wkt)

    drop, review = [], []
    for _, group in duplicated.groupby("wkt"):
        ids = sorted(int(i) for i in group.sample_id)
        seqs = {i: label_sequence(labels, i) for i in ids}
        classes = {i: tuple(seqs[i].label) for i in ids}

        if len(set(classes.values())) > 1:
            drop.extend(ids)
            a, b = ids
            review.append(
                {
                    "sample_id_a": a,
                    "sample_id_b": b,
                    "labels_a": format_sequence(seqs[a]),
                    "labels_b": format_sequence(seqs[b]),
                }
            )
            continue

        keep = min(
            ids,
            key=lambda i: (first_start(seqs[i], 242), first_start(seqs[i], 123), i),
        )
        drop.extend(i for i in ids if i != keep)

    return sorted(drop), review


def cluster_merges(samples: gpd.GeoDataFrame) -> dict[str, str]:
    wkt = samples.geometry.to_wkt()
    duplicated = samples[wkt.duplicated(keep=False)].assign(wkt=wkt)

    merges = {}
    for _, group in duplicated.sort_values("sample_id").groupby("wkt"):
        target, *others = group.cluster_id
        for cluster in others:
            if cluster != target:
                merges[cluster] = target

    # resolve chains, e.g. c -> b -> a becomes c -> a
    def resolve(cluster):
        while cluster in merges:
            cluster = merges[cluster]
        return cluster

    return {cluster: resolve(cluster) for cluster in merges}


def filter_parquet(
    path: str, drop: list[int], cluster_mapping: dict[str, str] | None = None
) -> None:
    # pyarrow round trip keeps schema and geo/pandas metadata untouched
    table = pq.read_table(path)
    drop_ids = pa.array(drop, type=table.schema.field("sample_id").type)
    mask = pc.invert(pc.is_in(table["sample_id"], value_set=drop_ids))
    table = table.filter(mask)

    if cluster_mapping:
        idx = table.schema.get_field_index("cluster_id")
        clusters = table["cluster_id"].to_pylist()
        merged = pa.array(
            [cluster_mapping.get(c, c) for c in clusters],
            type=table.schema.field("cluster_id").type,
        )
        table = table.set_column(idx, table.schema.field("cluster_id"), merged)

    pq.write_table(table, path)


if __name__ == "__main__":
    samples = gpd.read_parquet(f"{DATA}/samples.parquet")
    labels = pd.read_parquet(f"{DATA}/labels.parquet")

    drop, review = resolve_pairs(samples, labels)
    cluster_mapping = cluster_merges(samples)
    print(f"Dropping {len(drop)} samples, {len(review)} conflicting pairs to review")
    print(f"Merging clusters: {cluster_mapping}")

    filter_parquet(f"{DATA}/samples.parquet", drop, cluster_mapping)
    for name in ["labels", "pixel_data"]:
        filter_parquet(f"{DATA}/{name}.parquet", drop)

    for name in ["train_ids", "val_ids"]:
        with open(f"{DATA}/{name}.json") as f:
            ids = json.load(f)
        with open(f"{DATA}/{name}.json", "w") as f:
            json.dump([i for i in ids if i not in drop], f, indent=4)

    id_mapping = pl.read_csv(f"{DATA}/id_mapping.csv")
    id_mapping.filter(~pl.col.sample_id.is_in(drop)).write_csv(
        f"{DATA}/id_mapping.csv"
    )

    pd.DataFrame(review).to_csv("./notebooks/duplicates_to_review.csv", index=False)
