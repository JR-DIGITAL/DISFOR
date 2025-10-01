from pathlib import Path
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import xarray as xr
import rioxarray

# Does the same as in 2_get_zarrs, but a bit quicker due to not having the ipykernel overhead

samples = gpd.read_parquet("data/samples.parquet")
labels = pd.read_parquet("data/labels.parquet")

bands = {
    "blue": "B02_10m",
    "green": "B03_10m",
    "red": "B04_10m",
    "nir": "B08_10m",
    "rededge1": "B05_20m",
    "rededge2": "B06_20m",
    "rededge3": "B07_20m",
    "nir08": "B8A_20m",
    "swir16": "B11_20m",
    "swir22": "B12_20m",
    "scl": "SCL_20m",
}


def write_zarr_chip(ds, geom, sample_id):
    # Find nearest x/y indices
    x_idx = np.abs(ds.x - geom.x).argmin().item()
    y_idx = np.abs(ds.y - geom.y).argmin().item()

    # Define window size
    full_size = 128
    x_start = max(0, x_idx - (full_size // 2))
    x_end = x_start + full_size
    y_start = max(0, y_idx - (full_size // 2))
    y_end = y_start + full_size

    # Subset dataset
    subset = ds.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
    data_mask = (subset["SCL"] == 0).any(dim=("y", "x"))
    if data_mask:
        warnings.warn(
            f"Sample {sample_id} contains no valid data (SCL == 0). Skipping."
        )
        return

    get_data = subset.astype("uint16")

    # There's a bug with xarray doing over-eager conversion of timestamps (see https://github.com/pydata/xarray/issues/3942)
    # so we need to specify a time encoding
    encoding = {
        "time": {
            "units": "seconds since 2015-01-01",
            "calendar": "standard",
            "dtype": "int64",
        }
    }

    # This is to remove scale and offset, it messes with appending correct dtypes
    for data_var in get_data.data_vars:
        encoding[data_var] = {"dtype": "uint16"}
        get_data[data_var].attrs = {}

    # Write to zarr
    zarr_path = f"data/chips/{sample_id}.zarr"
    if not os.path.exists(zarr_path):
        # First time: initialize the zarr
        get_data.to_zarr(zarr_path, mode="w", zarr_format=3, encoding=encoding)
    else:
        # Next times: append along time
        get_data.drop_attrs().to_zarr(
            zarr_path, append_dim="time", mode="a", zarr_format=3
        )


def write_chip_wrapper(args):
    """Wrapper function for threading - unpacks arguments"""
    ds, geom, sample_id = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return write_zarr_chip(ds, geom, sample_id)


def load_sentinel2_bands(band_files, resolution="10m"):
    """Load Sentinel-2 bands into a properly structured xarray Dataset"""

    # Filter bands by resolution
    filtered_bands = {k: v for k, v in band_files.items() if k.endswith(resolution)}

    # Load each band
    band_arrays = {}
    for band_name, file_path in filtered_bands.items():
        # Open with rioxarray to preserve spatial reference
        da = rioxarray.open_rasterio(file_path, chunks=True, mask_and_scale=False)
        da = da.squeeze().drop_vars("band")  # Remove band dimension (it's singular)

        # Clean band name (B02, B03, etc.)
        clean_name = band_name.split("_")[0]
        band_arrays[clean_name] = da

    # Create dataset
    ds = xr.Dataset(band_arrays)

    return ds


def filter_acquisitions(acquisitions, samples_df):
    # try to load zarr from last sample_id and check which acquisitions have already been added
    last_sample = samples_df.iloc[-1]
    zarr_path = f"data/chips/{last_sample.sample_id}.zarr"
    if not os.path.exists(zarr_path):
        return acquisitions
    # getting last added acquisition and adding some tolerance due to smaller precision of time stored in zarr
    last_added_acquisition = pd.Timestamp(
        xr.open_zarr(zarr_path).time.values[-1]
    ) + pd.Timedelta(seconds=10)
    return [
        acquisition
        for acquisition in acquisitions
        if pd.Timestamp(acquisition.stem.split("_")[1]) > last_added_acquisition
    ]


def get_hvrpp_band_files(product_folder):
    band_files = {"SCL_20m": product_folder}  # Initialize with SCL_20m band
    # Searches for a match of the band name within the ID of the object
    for band in bands.values():
        if band == "SCL_20m":
            continue
        band_files[band] = Path(
            str(product_folder).replace(
                "SCENECLASSIFICATION_20M", f"TOC-{band.upper()}"
            )
        )
    return band_files


hrvpp_samples = samples.query("dataset=='HRVPP'")
tiles = list(hrvpp_samples["s2_tile"].unique())

for tile in tiles[-1:]:
    print(tile)
    acquisitions = list(
        Path("//digs110/FER/HR-VPP2/Data/TOC/v00/").glob(
            f"**/*_{tile}_SCENECLASSIFICATION*.tif"
        )
    )
    tiles_reprojected = hrvpp_samples.query("s2_tile==@tile").to_crs(
        f"EPSG:326{tile[0:2]}"
    )
    filtered_acquisitions = filter_acquisitions(acquisitions, tiles_reprojected)
    for product_folder in tqdm(filtered_acquisitions):
        timestamp = pd.Timestamp(product_folder.stem.split("_")[1])
        band_files = get_hvrpp_band_files(product_folder)

        # load data
        try:
            ds_10m = load_sentinel2_bands(band_files, "10m")
            ds_20m = load_sentinel2_bands(band_files, "20m").interp(
                x=ds_10m["x"],
                y=ds_10m["y"],
                method="nearest",
                kwargs={"fill_value": "extrapolate"},
            )
        except KeyError:
            print(f"some bands not found in manifest for tile {product_folder}")
            continue
        ds = (
            xr.merge([ds_10m, ds_20m])
            .expand_dims(dim="time")
            .assign_coords(time=[timestamp])
            .compute()
        )

        # Prepare arguments for threading
        chip_args = [
            (ds, row.geometry, row.sample_id) for _, row in tiles_reprojected.iterrows()
        ]

        # Write out chips using ThreadPoolExecutor
        max_workers = 16  # Adjust based on your system
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(write_chip_wrapper, args): args[2] for args in chip_args
            }

            # Process completed tasks with progress bar
            for future in as_completed(future_to_sample):
                sample_id = future_to_sample[future]
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as exc:
                    print(f"Sample {sample_id} generated an exception: {exc}")
