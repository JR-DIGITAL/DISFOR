import xarray as xr
from rasterio import CRS
import polars as pl
from pathlib import Path
from pyproj import Transformer
import datetime
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from typing import Tuple

# Set up logging to track progress
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_sample(args: Tuple) -> Tuple[str, str]:
    """
    Process a single sample - this function will be run in parallel.
    Args must be pickle-able for multiprocessing.
    """
    sample_dict, labels_path, desired_band_order = args
    sample_id = sample_dict["sample_id"]

    try:
        # Skip if already exists
        if Path(f"data/tiffs/{sample_id}").exists():
            logger.info(f"Skipping {sample_id} - already exists")
            return sample_id, "skipped"

        # Load labels in each process (avoid pickling large dataframes)
        labels = pl.read_parquet(labels_path)

        # Open zarr file
        ds = xr.open_zarr(
            f"data/chips/{sample_id}.zarr",
            decode_coords="all",
            mask_and_scale=False,
            chunks=None,
            use_zarr_fill_value_as_mask=False,
        )

        epsg = CRS.from_wkt(ds.spatial_ref.crs_wkt).to_epsg()
        ds = ds.drop_vars("spatial_ref")
        ds = ds.rio.write_crs(epsg)
        ds = ds[desired_band_order].drop_duplicates("time")

        # Find row/col
        transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
        xx, yy = transformer.transform(
            sample_dict["geometry_x"], sample_dict["geometry_y"]
        )
        ix = ds.indexes["x"].get_indexer([xx], method="nearest")[0]
        iy = ds.indexes["y"].get_indexer([yy], method="nearest")[0]

        # Spatial subset: 32x32
        chip_size = 32
        half_size = chip_size // 2
        small_ds = ds.isel(
            x=slice(ix - half_size, ix + half_size),
            y=slice(iy - half_size, iy + half_size),
        )

        clear_ts = small_ds.astype("uint16")

        # Slice time based on label
        sample_labels = labels.filter(pl.col("sample_id") == sample_id)

        for row in sample_labels.iter_rows(named=True):
            label = row["label"]
            start = row["start"].replace(tzinfo=None)
            end = row.get("start_next_label")
            if end is None:
                end = row.get("end")
            end = end.replace(tzinfo=None) - datetime.timedelta(days=1)

            label_ts = clear_ts.sel(time=slice(start, end))

            for time in label_ts.time:
                time_str = time.astype("datetime64[us]").item().strftime("%Y-%m-%d")
                out_file = Path(f"data/tiffs/{sample_id}/{label}/{time_str}.tif")
                out_file.parent.mkdir(exist_ok=True, parents=True)
                label_ts.sel(time=time).rio.to_raster(out_file, driver="COG")

        logger.info(f"Successfully processed {sample_id}")
        return sample_id, "success"

    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {str(e)}")
        return sample_id, f"error: {str(e)}"


def prepare_sample_dict(row):
    """Convert GeoDataFrame row to a pickle-able dictionary"""
    return {
        "sample_id": row["sample_id"],
        "geometry_x": row["geometry"].x,
        "geometry_y": row["geometry"].y,
    }


def main():
    # Open labels and samples
    samples = gpd.read_parquet("data/samples.parquet")
    labels_path = "data/labels.parquet"
    desired_band_order = [
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

    # Prepare arguments for each process
    # Convert samples to pickle-able format
    sample_args = [
        (prepare_sample_dict(row), labels_path, desired_band_order)
        for _, row in samples.iterrows()
    ]

    # Determine number of workers
    # For ProcessPoolExecutor, typically use number of CPU cores or slightly less
    max_workers = 10  # None uses cpu_count(), or set explicitly like: max_workers = 4

    logger.info(
        f"Starting processing of {len(sample_args)} samples with {max_workers or 'default'} workers"
    )

    # Process samples in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(process_sample, args): args[0]["sample_id"]
            for args in sample_args
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_sample):
            sample_id = future_to_sample[future]
            try:
                result_id, status = future.result()
                results[result_id] = status
                completed += 1
                logger.info(
                    f"Progress: {completed}/{len(sample_args)} samples completed"
                )
            except Exception as e:
                logger.error(f"Unexpected error for sample {sample_id}: {str(e)}")
                results[sample_id] = f"unexpected error: {str(e)}"
                completed += 1

    # Summary
    logger.info("Processing complete!")
    success_count = sum(1 for status in results.values() if status == "success")
    skipped_count = sum(1 for status in results.values() if status == "skipped")
    error_count = sum(1 for status in results.values() if status.startswith("error"))
    logger.info(
        f"Summary: {success_count} successful, {skipped_count} skipped, {error_count} errors"
    )

    # Log errors if any
    if error_count > 0:
        logger.warning("Samples with errors:")
        for sample_id, status in results.items():
            if status.startswith("error"):
                logger.warning(f"  {sample_id}: {status}")


if __name__ == "__main__":
    main()
