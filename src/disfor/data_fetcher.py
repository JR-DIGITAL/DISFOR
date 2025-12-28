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
        "samples.parquet": "dc6358f7f04fa0cf56cda6efb24ff1279399a4e30f50c808bf790f7e3b766517",
        "labels.parquet": "9421e630f42da37737f2ed9f608f978b4c88af3ec51d72350d1b2169b3160108",
        "pixel_data.parquet": "294f074eea4a1a5ce32ce793c9118f9cecbd0fdd5c2b665f0d5a0145de41ed64",
        "train_ids.json": "e1aec584175ad95caf417c44feec6c808b9a45e617f5554d4495034a889d04ce",
        "val_ids.json": "c0745e6d5bc671e3906033d1d74ccca2b6989f445db9a28552d8f39ee717800b",
        "classes.json": "bf088fa6b91725a947352e75a5c6967052df1015ac0d2e655787fa835f0ea244",
        "disfor-0-499.tar.zst": "fe062336c6db5106432983a24981fa3a7e7d854bc0699aeea85a7754f42e4c73",
        "disfor-500-999.tar.zst": "c20f4110402c5eeef912d5cf1d00410927102b0423775d54ba1fbfe134c563f9",
        "disfor-1000-1499.tar.zst": "5e33c5bddc467060f603665866a795c665a94199813248376c9b21c2d5913f27",
        "disfor-1500-1999.tar.zst": "0d92b7a7d93e7c539fa88399d9f7b8f0b6f280e064dca8d344f3af8f03f31de3",
        "disfor-2000-2499.tar.zst": "ac711fb765808ddddbab1f814770eab90473b4f5a3887867c3747b91fab8513c",
        "disfor-2500-2999.tar.zst": "0a407d101d4af60fcf56a0c29be4265657515d61e77b427d0df11b7be6068e81",
        "disfor-3000-3499.tar.zst": "51b9bfa9bc98464b241fde4626ed99516459cf9a68cc436b01a2dd73b4f4bda1",
        "disfor-3500-3999.tar.zst": "fa1ea1168c857bf4fc567d329a02db785584316468ad3401f1494887af8d37e3",
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
