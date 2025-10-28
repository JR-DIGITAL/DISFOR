import zstandard as zstd
import tarfile
import glob
from pathlib import Path

import polars as pl


def generate_folds(n_folds: int, data_folder="data"):
    from sklearn.model_selection import StratifiedGroupKFold

    groups = pl.read_parquet(
        Path(data_folder) / "samples.parquet",
        columns=["sample_id", "cluster_id", "comment", "dataset", "confidence"],
        use_pyarrow=True,
    )
    # sample_ids in HRVPP not highly correlated -> use sample_id as group
    # Evoland: Group by cluster_id
    # Windthrow: Group by Wind Event
    clusters = groups.with_columns(
        cluster=pl.when(dataset="HRVPP")
        .then(pl.col.sample_id.cast(pl.String))
        .otherwise(pl.format("{}{}", pl.col.dataset, pl.col.cluster_id))
    )
    samples_w_clusters = (
        pl.read_parquet(Path(data_folder) / "labels.parquet")
        .join(clusters.select("sample_id", "cluster"), on="sample_id")
        .sort("cluster")
        .with_columns(
            pl.col.label.cast(pl.Int16),
            cluster_int=pl.col("cluster").rle_id(),
        )
    )
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    splits = sgkf.split(
        X=samples_w_clusters["label"],
        y=samples_w_clusters["label"],
        groups=samples_w_clusters["cluster_int"],
    )
    folds = {}
    sample_ids = samples_w_clusters["sample_id"].to_numpy()
    for i, (train_index, test_index) in enumerate(splits):
        folds[i] = {}
        folds[i]["train_ids"] = set(sample_ids[train_index].tolist())
        folds[i]["val_ids"] = set(sample_ids[test_index].tolist())
    return folds


def extract_multipart(pattern):
    """
    Pure Python extraction (requires zstandard package)
    Install: pip install zstandard
    """
    parts = sorted(glob.glob(pattern))

    # Create a decompressor
    dctx = zstd.ZstdDecompressor()

    # Decompress and extract in one go
    with dctx.stream_reader(CombinedReader(parts)) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall()


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
