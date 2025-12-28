import tifffile
from pathlib import Path
import polars as pl
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def extract_from_tif(tif: Path) -> pl.DataFrame:
    sample_id = tif.parent.stem
    # label = tif.parent.stem
    date = tif.stem

    arr = tifffile.imread(tif)
    middle = 16
    spectral_pixel_values = arr[middle, middle, :]

    # Calculate "clear" mask
    clear = np.isin(arr[:, :, -1], [2, 4, 5, 6])
    clear_4x4 = clear[middle - 2 : middle + 2, middle - 2 : middle + 2].mean()
    clear_8x8 = clear[middle - 4 : middle + 4, middle - 4 : middle + 4].mean()
    clear_16x16 = clear[middle - 8 : middle + 8, middle - 8 : middle + 8].mean()
    clear_32x32 = clear[middle - 16 : middle + 16, middle - 16 : middle + 16].mean()

    # Schema definition
    schema = {
        "B02": pl.UInt16,
        "B03": pl.UInt16,
        "B04": pl.UInt16,
        "B05": pl.UInt16,
        "B06": pl.UInt16,
        "B07": pl.UInt16,
        "B08": pl.UInt16,
        "B8A": pl.UInt16,
        "B11": pl.UInt16,
        "B12": pl.UInt16,
        "SCL": pl.UInt8,
    }

    # Build DataFrame
    return (
        pl.DataFrame(spectral_pixel_values[np.newaxis], schema, orient="row")
        .with_columns(
            sample_id=pl.lit(sample_id).cast(pl.UInt16),
            timestamps=pl.date(*[int(d) for d in date.split("-")]),
            # label=pl.lit(label).cast(pl.UInt16),
            percent_clear_4x4=clear_4x4,
            percent_clear_8x8=clear_8x8,
            percent_clear_16x16=clear_16x16,
            percent_clear_32x32=clear_32x32,
        )
        .with_columns((pl.selectors.starts_with("percent_clear") * 100).cast(pl.UInt8))
    )


def chunked_iterable(iterable, n):
    """Split iterable into n roughly equal chunks."""
    k, m = divmod(len(iterable), n)
    return (iterable[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def process_chunk(tif_list):
    return pl.concat([extract_from_tif(tif) for tif in tif_list])


if __name__ == "__main__":
    # List TIFF files
    print("Start glob")
    tifs = list(Path("./data/tiffs").glob("*/*.tif"))
    print("End glob")
    band_order = [
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

    num_workers = 16  # or os.cpu_count()
    chunks = list(chunked_iterable(tifs, 10000))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        dfs = list(
            tqdm(
                executor.map(process_chunk, chunks),
                total=len(chunks),
                desc="Processing chunks",
            )
        )

    # Concatenate final result
    df = pl.concat(dfs).with_columns(clear=pl.col.SCL.is_in([2, 4, 5, 6]))
    labels_df = pl.read_parquet("./data/labels.parquet").select(
        pl.col.sample_id,
        pl.col.label,
        timestamps=pl.col.start.dt.date(),
    )
    added_labels = df.join_asof(labels_df, by="sample_id", on="timestamps")

    # Save result
    added_labels.write_parquet("./data/pixel_data.parquet")
