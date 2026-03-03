import importlib.metadata
from .io import get

__version__ = importlib.metadata.version("disfor")

__all__ = ["get"]
