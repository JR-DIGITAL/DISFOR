import json
from pathlib import Path

import polars as pl
from utils import campaign_to_tables, validate_campaign

if __name__ == "__main__":
    with Path("labels/disfor.geojson").open("r") as f:
        campaign = json.load(f)
    issues = validate_campaign(campaign)
    if issues:
        print(issues)
        print(",".join([i.split(" ")[1][:-1] for i in issues]))
    else:
        s, l = campaign_to_tables(campaign)
        s.to_parquet("data/samples.parquet")
        l.to_parquet("data/labels.parquet")

        # Update labels for pixel_data
        df = pl.read_parquet("./data/pixel_data.parquet").drop("label")
        labels_df = pl.read_parquet("./data/labels.parquet").select(
            pl.col.sample_id,
            pl.col.label,
            timestamps=pl.col.start.dt.date(),
        )
        added_labels = df.join_asof(labels_df, by="sample_id", on="timestamps")

        # Save result
        added_labels.write_parquet("./data/pixel_data.parquet")
