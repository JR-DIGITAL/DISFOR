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
        "samples.parquet": "ee22c9d590ae8043562594c77ec30623f5fc060d240384d5a3455ce2f643f0c6",
        "labels.parquet": "fdb87561e72bec626e0e2c145b84132a8f31c2be8e17770ec15bda15a074ad00",
        "pixel_data.parquet": "58b70bd8145f1afbcc27f7a04043deffa876d8bf64ea2296b364d707b86c2d24",
        "train_ids.json": "42bee12fe4615212b8a44a544639f30b0d9199cf24c75bbae4e794cbfb98b98f",
        "val_ids.json": "5d2078f4081b2e8cc536f2465437c8bcde79557cf5938714448dfd42e2809942",
        "classes.json": "0acaa13405dab6dfa6fe48422646c391d90fd6ee9bb80b7e658459ae244bf108",
        "chips": None,
    },
)

# Define the parts and their hashes
DISFOR_PARTS = {
    "disfor.tar.zst.aa": "1d96fcc95ed37bce1eb036cda62f5b2a6a67f9e87ed48bfaae20e544b0e8dadd",
    "disfor.tar.zst.ab": "d2a6de0db578137df10620de3caed109d3a26d6f3fcce8f27ddc4a46bf604cfc",
    "disfor.tar.zst.ac": "505aaddc6471e2ac09cbd5f6c36d0b71df2fba684823f1332521eb735d6c7f3b",
    "disfor.tar.zst.ad": "1c131d6e6da97a8f3d82ede0e7b897d3bbc4ef1fb160aea60e29168691dbee39",
    "disfor.tar.zst.ae": "3a74c32e694a0c76da593d899a3cdb11c72bbe06ce972ffb059bd64c874c5129",
    "disfor.tar.zst.af": "e3e9425b9f76351df05bf53cfc67bdd21cea3b30a9683ab23c2c4475f1123d76",
    "disfor.tar.zst.ag": "5609b5fc51a82a0814d0aad719bb3b84c6db0f605befd5e85446f42fe460fdb5",
    "disfor.tar.zst.ah": "28b10a172c1d5f0a29602884edca6da6d0ae5390240307d0d31a6a7103065bf6",
    "disfor.tar.zst.ai": "ffb502e8a89c2d1aa85515e7de6a5caf8eb47ecaf786647e2aac946d3db50065",
    "disfor.tar.zst.aj": "c95d8a3f04beebb80d99595c0896f5e2f1c1865ea45e286fbe2762d3791abf62",
}


class MultiPartDownloader:
    """
    Download multiple parts of a split archive.

    Parameters
    ----------
    parts : dict
        Dictionary mapping part filenames to their SHA256 hashes.
    """

    def __init__(self, parts):
        self.parts = parts

    def __call__(self, url, output_file, pooch_inst):
        """
        Download all parts of the archive.

        Parameters
        ----------
        url : str
            Base URL (will be ignored, we'll construct URLs for each part)
        output_file : str
            Path where the concatenated file will be stored
        pooch : Pooch
            The Pooch instance
        """
        output_path = Path(output_file)
        base_url = pooch_inst.base_url

        # Download each part
        part_files = []
        for part_name, expected_hash in self.parts.items():
            part_url = f"{base_url}{part_name}"
            part_path = output_path.parent / part_name

            # Download this part
            print(f"Downloading {part_name}...")
            part_path = pooch.retrieve(part_url, expected_hash, str(part_path))

            part_files.append(part_path)


class CombinedReader:
    """Helper class to read multiple files as one stream"""

    def __init__(self, filenames):
        self.filenames = filenames
        self.current_file = None
        self.file_index = 0

    def read(self, size=-1):
        data = b""
        while size != 0:
            if self.current_file is None:
                if self.file_index >= len(self.filenames):
                    break
                self.current_file = open(self.filenames[self.file_index], "rb")
                self.file_index += 1

            chunk = self.current_file.read(size if size > 0 else 8192)
            if not chunk:
                self.current_file.close()
                self.current_file = None
                continue

            data += chunk
            if size > 0:
                size -= len(chunk)

        return data


class ExtractAndCleanup:
    """
    Extract a tarball and delete the archive file.
    """

    def __init__(self, extract_dir=None):
        self.extract_dir = extract_dir

    def __call__(self, fname, action, pooch_inst):
        """ """
        fname_path = Path(fname)
        check = "tiffs_unpacked"

        extract_path = fname_path.parent
        # Only extract if we just downloaded or if extracted dir doesn't exist
        if action in ("update", "download") or not (extract_path / check).exists():
            extract_path.mkdir(parents=True, exist_ok=True)
            dctx = zstd.ZstdDecompressor()
            part_paths = [
                fname_path.parent / part_name for part_name in self.parts.keys()
            ]
            with dctx.stream_reader(CombinedReader(part_paths)) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(path=extract_path)

            # write a file at the end to signal that everything was unpacked successfully,
            # check for that file the next time this is run
            with (extract_path / check).open("w") as f:
                f.write("")

            # Delete the individual part files
            for part_path in part_paths:
                if part_path.exists():
                    part_path.unlink()

        return str(extract_path / "tiffs")


def fetch_s2_chips():
    """
    Load S2 chips. Downloads multi-part tarball, extracts it, and cleans up
    the downloaded archive files.

    Returns
    -------
    str
        Path to the extracted directory containing the chips
    """
    # Create the processor with the parts info so it can clean them up
    processor = ExtractAndCleanup()
    processor.parts = DISFOR_PARTS  # Add parts info for cleanup

    fname = DATA_GETTER.fetch(
        "chips",
        downloader=MultiPartDownloader(parts=DISFOR_PARTS),
        processor=processor,
    )
    return Path(fname)
