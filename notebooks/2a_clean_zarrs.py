import xarray as xr
import shutil
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Used after 2_get_zarrs to rechunk zarr files to make store more compressed

zarr_ids = [path.stem for path in Path("data/chips/").glob("*.zarr")]


def clean_zarr(zarr_id):
    try:
        x = xr.open_zarr(
            f"data/chips/{zarr_id}.zarr",
            mask_and_scale=False,
            decode_coords="all",
            chunks=None,
        )
        new_size = 128
        new_time_dim = 32
        rechunked = x.chunk({"time": new_time_dim, "y": new_size, "x": new_size})

        for var in rechunked:
            del rechunked[var].encoding["chunks"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
            rechunked.to_zarr(
                f"data/cleaned_chips/{zarr_id}.zarr", zarr_format=3, mode="w"
            )

        shutil.rmtree(f"data/chips/{zarr_id}.zarr")
    except Exception as e:
        print(f"Error processing {zarr_id}: {e}")


# Parallel execution wrapper
def process_zarr_ids_parallel(zarr_ids, max_workers=15):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(clean_zarr, zarr_id): zarr_id for zarr_id in zarr_ids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing Zarr files"
        ):
            _ = future.result()  # Errors are already printed in `clean_zarr`


if __name__ == "__main__":
    # Process all Zarr IDs in parallel
    process_zarr_ids_parallel(zarr_ids, max_workers=16)
