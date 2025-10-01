import warnings

import geopandas as gpd
from tqdm import tqdm
import xarray as xr
import zarr
import shutil

bands = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir08": "B8A",
    "swir16": "B11",
    "swir22": "B12",
    "scl": "SCL",
}


def move_wt_zarr(row):
    zarr_path = f"\\\\wsl.localhost\\Ubuntu\\home\\Documents\\Projects\\2025\\agent-reference-data\\sample_areas\\windthrow\\windthrow_sample_chips\\{row['cluster_id']}\\{row['original_sample_id']}.zarr"
    try:
        x = xr.open_zarr(zarr_path, mask_and_scale=False, decode_coords="all").compute()
    except FileNotFoundError:
        print(f"Zarr file not found: {row['cluster_id']} {row['original_sample_id']}")
        return
    new_size = 128
    new_time_dim = -1

    small_chip = x.isel(
        x=slice(len(x.x) // 2 - new_size // 2, len(x.x) // 2 + new_size // 2),
        y=slice(len(x.y) // 2 - new_size // 2, len(x.y) // 2 + new_size // 2),
    ).rename(bands)

    rechunked = small_chip.chunk({"time": new_time_dim, "y": new_size, "x": new_size})
    for var in rechunked:
        del rechunked[var].encoding["chunks"]

    compressors = zarr.codecs.BloscCodec(
        cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.bitshuffle
    )

    encoding = {}
    for data_var in rechunked.data_vars:
        encoding[data_var] = {"compressors": compressors}
    with warnings.catch_warnings():
        # rioxarray is warning about different scales per band, did not find a way to handle this warning
        # so we just ignore it
        warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
        rechunked.to_zarr(
            f"data/chips/{row['sample_id']}.zarr", zarr_format=3, encoding=encoding
        )
    shutil.rmtree(zarr_path)


samples = gpd.read_parquet("data/samples.parquet").query("dataset=='Windthrow'")

for _, row in tqdm(samples.iterrows()):
    move_wt_zarr(row)
