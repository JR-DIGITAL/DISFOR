import polars as pl


if __name__ == "__main__":
    # Concatenate final result
    df = pl.read_parquet("./data/pixel_data.parquet").drop("label")
    labels_df = pl.read_parquet("./data/labels.parquet").select(
        pl.col.sample_id,
        pl.col.label,
        timestamps=pl.col.start.dt.date(),
    )
    added_labels = df.join_asof(labels_df, by="sample_id", on="timestamps")

    # Save result
    added_labels.write_parquet("./data/pixel_data.parquet")
