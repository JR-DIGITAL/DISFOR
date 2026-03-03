import pooch
import tarfile
from pathlib import Path
import zstandard as zstd

# Create a new friend to manage your sample data storage
DATA_GETTER = pooch.create(
    # Folder where the data will be stored. For a sensible default, use the
    # default cache folder for your OS.
    path=pooch.os_cache("disfor"),
    # Base URL of the remote data store. Will call .format on this string
    # to insert the version (see below).
    base_url="https://huggingface.co/datasets/JR-DIGITAL/DISFOR/resolve/main/",
    # Change this, whenever the DATA (on huggingface or zenodo) changes
    version="0.1.0",
    # If a version as a "+XX.XXXXX" suffix, we'll assume that this is a dev
    # version and replace the version with this string.
    version_dev="main",
    registry={
        "samples.parquet": "1da65ad876e7269b138b5fa88be5f4a6b7f317f16f81bdab11c8eb49bd5232d7",
        "labels.parquet": "322c6d77cc8db13c4f2fb6b09c0f2c98e0f476c7315bc6412f40be4ad0b0b360",
        "pixel_data.parquet": "058fbe1c04934698929f25fe84dbf4b13d6c7e2298165d5b4856d42e37f6f368",
        "train_ids.json": "316df9ed292ddd9d4da899e4de8515df561e95b3fc8309fc930723ccbe419fa3",
        "val_ids.json": "bbda635c31cc0f7e6954fab71f0944e595ffa81270b6385fb7d2495fb0d9c794",
        "classes.json": "8c0ee798c8e2ba2fc85e98584f35647fd720f7845b2e684ad5ae14d54fc3150e",
        "disfor-0-499.tar.zst": "ec3e920022cf579461c43902f28cae0a1c2c7bdb62fb67e99991c57ee812bbda",
        "disfor-500-999.tar.zst": "63dc9f18715a6e8da9b49c41f461246485477f41855da2d4090618cb4c9aefea",
        "disfor-1000-1499.tar.zst": "3da9166c1164e91f77591daa43034a3e2831a7472579b5b2d05c7ccdb5b63ec9",
        "disfor-1500-1999.tar.zst": "2e54ed525e1badaff92c0f6851d93c8b783de84cd10a25bb4b751cdf95fbada4",
        "disfor-2000-2499.tar.zst": "ea6be48314618e8e5fda0025f6b9cec6379c723985f8647ea808c498554b5768",
        "disfor-2500-2999.tar.zst": "c6789d2d1e7d00bb64e830ff784f3eaca01d909b74fb213c7f400b8096e01a51",
        "disfor-3000-3499.tar.zst": "f411d31da3ca82d5fc27d8a80f997955303ab19cf498cf8cf1f6bc4e072a58fd",
        "disfor-3500-3999.tar.zst": "833a03db7f83c30bbd8bcd5225dfce8f7662c86f0df7fd4b5ab72c78abc97295",
    },
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

    for archive_name, archive_hash in DATA_GETTER.registry.items():
        if not archive_name.startswith("disfor"):
            continue

        # Skip download if marker exists and matches registry
        if marker_matches_registry(DATA_GETTER, archive_name):
            continue

        # Otherwise fetch + extract
        extract_path = DATA_GETTER.fetch(
            archive_name,
            processor=processor,
        )

    # Return the chips directory (assumes all tarballs extract into same dir)
    if extract_path is None:
        extract_path = Path(DATA_GETTER.path) / "tiffs"

    return extract_path
