from pathlib import Path
import polars as pl
import rasterio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def process_sample(args):
    sample_id, sample_id_right, source, destination = args
    source = Path(source)
    destination = Path(destination)

    sample_out = destination / str(sample_id)
    sample_out.mkdir(parents=True, exist_ok=True)

    for tiff in (source / str(sample_id_right)).glob("*.tif"):
        out_path = sample_out / tiff.name
        if out_path.exists():
            continue
        with rasterio.open(tiff) as src:
            profile = src.profile
            profile.update(compress="zstd")
            arr = src.read()
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)


def chunked_iterable(iterable, n):
    """Split iterable into n roughly equal chunks."""
    k, m = divmod(len(iterable), n)
    return (iterable[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def process_chunk(chunk):
    return [process_sample(sample) for sample in chunk]


if __name__ == "__main__":
    old = pl.read_csv(
        "https://huggingface.co/datasets/JR-DIGITAL/DISFOR/resolve/main/id_mapping.csv"
    ).filter(pl.col.dataset.ne("Evoland"))

    new = pl.read_csv(
        r"C:\Users\Jonas.Viehweger\Documents\Projects\2025\DISFOR\data\id_mapping.csv"
    ).filter(pl.col.dataset.ne("Evoland"))

    joined = new.join(old, on=["dataset", "original_sample_id"], how="inner")

    source = Path(
        r"C:\Users\Jonas.Viehweger\AppData\Local\disfor\disfor\Cache\0.1.0\tiffs"
    )
    destination = Path(
        r"C:\Users\Jonas.Viehweger\Documents\Projects\2025\DISFOR\data\tiffs"
    )

    tasks = [
        (
            row["sample_id"],
            row["sample_id_right"],
            str(source),  # Convert to string for serialization
            str(destination),  # Convert to string for serialization
        )
        for row in joined.iter_rows(named=True)
    ]
    n_workers = 32

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(
            tqdm(
                executor.map(process_sample, tasks),
                total=len(tasks),
            )
        )
