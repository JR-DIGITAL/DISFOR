import zstandard as zstd
import tarfile
import glob


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
