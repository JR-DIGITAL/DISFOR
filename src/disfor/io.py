import tarfile
from pathlib import Path
from typing import Literal

import pooch
import zstandard as zstd

registry = {
    "samples.parquet": "fea12863181d6003df2377e39f21ce2b4b94ae8ca509a385460c340610db5a9f",
    "labels.parquet": "cf742dc08ca91d412a0d5ec7726618042ad43cc87d1d75daa6fb57e972a1130f",
    "pixel_data.parquet": "0fb66cdba50a8492ab478e7a5e59ed490b84faaaf070c7735f55589bb308b69a",
    "train_ids.json": "14a9f88d14154d4136a7b50c8efe13f67f1f609213c14cdf461e1bd40ed997c2",
    "val_ids.json": "f24f28df1a902b5f1aeb775cda29e5be78157f3d6c9330927ea4e72697adf2a5",
    "classes.json": "80691cec88e4d90798fc8378032f14e6bd488a39d0a8c21ecf11c93b1fa7713b",
    "disfor-0-499.tar.zst": "ec3e920022cf579461c43902f28cae0a1c2c7bdb62fb67e99991c57ee812bbda",
    "disfor-500-999.tar.zst": "63dc9f18715a6e8da9b49c41f461246485477f41855da2d4090618cb4c9aefea",
    "disfor-1000-1499.tar.zst": "3da9166c1164e91f77591daa43034a3e2831a7472579b5b2d05c7ccdb5b63ec9",
    "disfor-1500-1999.tar.zst": "2e54ed525e1badaff92c0f6851d93c8b783de84cd10a25bb4b751cdf95fbada4",
    "disfor-2000-2499.tar.zst": "ea6be48314618e8e5fda0025f6b9cec6379c723985f8647ea808c498554b5768",
    "disfor-2500-2999.tar.zst": "c6789d2d1e7d00bb64e830ff784f3eaca01d909b74fb213c7f400b8096e01a51",
    "disfor-3000-3499.tar.zst": "f411d31da3ca82d5fc27d8a80f997955303ab19cf498cf8cf1f6bc4e072a58fd",
    "disfor-3500-3999.tar.zst": "833a03db7f83c30bbd8bcd5225dfce8f7662c86f0df7fd4b5ab72c78abc97295",
}

# Create a new friend to manage your sample data storage
_DATA_GETTER = pooch.create(
    # Folder where the data will be stored. For a sensible default, use the
    # default cache folder for your OS.
    path=pooch.os_cache("disfor"),
    # Base URL of the remote data store. Will call .format on this string
    # to insert the version (see below).
    base_url="https://huggingface.co/datasets/JR-DIGITAL/DISFOR/resolve/{version}/",
    # Change this, whenever the DATA (on huggingface or zenodo) changes
    version="0.2.1",
    # If a version as a "+XX.XXXXX" suffix, we'll assume that this is a dev
    # version and replace the version with this string.
    version_dev="main",
    registry=registry,
)


def marker_path(pooch, archive_name: str) -> Path:
    """Return the marker file path for an archive."""
    return Path(pooch.path) / f"{archive_name}.hash"


def marker_matches_registry(pooch, archive_name: str) -> bool:
    """Check whether the marker file exists and matches registry hash."""
    reg_hash = pooch.registry.get(archive_name)
    if reg_hash is None:
        return False

    mpath = marker_path(pooch, archive_name)
    if not mpath.exists():
        return False

    return mpath.read_text().strip() == reg_hash


class ExtractTarZst:
    def __call__(self, fname, action, pooch):
        fname = Path(fname)
        archive_name = fname.name

        if action in ("download", "update"):
            # Decompress and extract
            with open(fname, "rb") as compressed:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(compressed) as reader:
                    with tarfile.open(fileobj=reader, mode="r|") as tar:
                        tar.extractall(path=fname.parent)

            # Write marker file with registry hash
            reg_hash = pooch.registry.get(archive_name)
            if reg_hash is not None:
                mpath = marker_path(pooch, archive_name)
                mpath.write_text(reg_hash)

            # Delete the archive
            fname.unlink()

        return fname.parent / "tiffs"


def fetch_s2_chips():
    """
    Load S2 chips. Downloads multi-part tarball, extracts it, and cleans up
    the downloaded archive files.
    """
    processor = ExtractTarZst()
    extract_path = None

    for archive_name, archive_hash in _DATA_GETTER.registry.items():
        if not archive_name.startswith("disfor"):
            continue

        # Skip download if marker exists and matches registry
        if marker_matches_registry(_DATA_GETTER, archive_name):
            continue

        # Otherwise fetch + extract
        extract_path = _DATA_GETTER.fetch(
            archive_name,
            processor=processor,
        )

    # Return the chips directory (assumes all tarballs extract into same dir)
    if extract_path is None:
        extract_path = Path(_DATA_GETTER.path) / "tiffs"

    return extract_path


def get(
    name: Literal[
        "samples.parquet",
        "labels.parquet",
        "pixel_data.parquet",
        "tiffs",
        "train_ids.json",
        "val_ids.json",
        "classes.json",
    ],
) -> Path:
    """
    Function to get data paths

    This function fetches the paths to the raw data provided in DISFOR.
    If the data is not available locally, it uses `pooch` to fetch the data from huggingface.

    Args:
        name: Name of the file to fetch.

    Returns:
        Path to the file on local storage
    """
    if name == "tiffs":
        return fetch_s2_chips()
    return Path(_DATA_GETTER.fetch(name))
